//! Whisper GGML model management: pinned model table, resumable downloads,
//! and SHA-256 verification.
//!
//! Models live in `$XDG_DATA_HOME/sublayer/models/` (see
//! [`SublayerPaths::models_dir`](sublayer_core::SublayerPaths)). A download is
//! streamed into a deterministic `<name>.bin.part` file so interrupted runs
//! resume via HTTP `Range` requests; the file is only moved to its final name
//! after its SHA-256 digest matches the pinned value.

use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::header;
use sha2::{Digest, Sha256};
use sublayer_core::SublayerPaths;

use crate::AiError;

/// Read chunk size used while hashing model files.
const HASH_BUFFER_SIZE: usize = 64 * 1024;

/// A pinned Whisper GGML model published in the `ggerganov/whisper.cpp`
/// repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Short name used on the command line and in config files.
    pub name: &'static str,
    /// File name under `models/` once downloaded.
    pub file_name: &'static str,
    /// HTTPS download URL.
    pub url: &'static str,
    /// Expected SHA-256 digest of the model file, lowercase hex.
    pub sha256: &'static str,
}

impl ModelSpec {
    /// Looks up a model by its short name.
    pub fn find(name: &str) -> Option<&'static ModelSpec> {
        MODELS.iter().find(|spec| spec.name == name)
    }
}

/// Known models. Digests are the Hugging Face LFS SHA-256 of each file.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "tiny.en",
        file_name: "ggml-tiny.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        sha256: "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
    },
    ModelSpec {
        name: "base.en",
        file_name: "ggml-base.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        sha256: "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    },
    ModelSpec {
        name: "small.en",
        file_name: "ggml-small.en.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        sha256: "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
    },
    ModelSpec {
        name: "large-v3-turbo",
        file_name: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    },
];

/// Downloads and verifies Whisper models under the Sublayer data directory.
#[derive(Debug)]
pub struct ModelManager {
    paths: SublayerPaths,
    client: reqwest::Client,
}

impl ModelManager {
    /// Creates a manager for the user's XDG directories.
    pub fn new(paths: SublayerPaths) -> Result<Self, AiError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            // Large models need long transfers; keep a generous overall bound
            // so a stalled connection still fails eventually.
            .timeout(Duration::from_secs(3 * 60 * 60))
            .user_agent(concat!("sublayer/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(AiError::ClientBuild)?;
        Ok(Self::with_client(paths, client))
    }

    /// Creates a manager with an explicit HTTP client (custom proxies,
    /// timeouts, or test servers).
    pub fn with_client(paths: SublayerPaths, client: reqwest::Client) -> Self {
        Self { paths, client }
    }

    /// The directory models are downloaded into.
    pub fn models_dir(&self) -> PathBuf {
        self.paths.models_dir()
    }

    /// Returns the path of `name`, downloading and verifying it when needed.
    ///
    /// Already-downloaded models are reused when their digest matches; a
    /// corrupt file is deleted and fetched again.
    pub async fn ensure_cached(&self, name: &str) -> Result<PathBuf, AiError> {
        let spec = ModelSpec::find(name).ok_or_else(|| AiError::UnknownModel(name.to_owned()))?;
        self.cached_path_for(spec).await
    }

    /// Like [`ModelManager::ensure_cached`], but for an arbitrary spec.
    pub(crate) async fn cached_path_for(&self, spec: &ModelSpec) -> Result<PathBuf, AiError> {
        let final_path = self.paths.model_path(spec.file_name);
        if final_path.is_file() {
            if verify_sha256(&final_path, spec.sha256)? {
                return Ok(final_path);
            }
            tracing::warn!(
                model = spec.name,
                path = %final_path.display(),
                "existing model failed hash verification; re-downloading"
            );
            fs::remove_file(&final_path)?;
        }
        self.download_spec(spec, &final_path).await?;
        Ok(final_path)
    }

    /// Downloads `spec` into `final_path` with resume support and digest
    /// verification. On a verification failure the partial file is discarded
    /// and the download retried once from scratch.
    pub(crate) async fn download_spec(
        &self,
        spec: &ModelSpec,
        final_path: &Path,
    ) -> Result<(), AiError> {
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let part_path = part_path(final_path);

        let offset = fs::metadata(&part_path).map(|meta| meta.len()).unwrap_or(0);
        self.fetch_into(spec, &part_path, offset).await?;

        let actual = sha256_hex(&part_path)?;
        if actual != spec.sha256 {
            tracing::warn!(
                model = spec.name,
                "downloaded model failed verification; retrying from scratch"
            );
            fs::remove_file(&part_path)?;
            self.fetch_into(spec, &part_path, 0).await?;
            let actual = sha256_hex(&part_path)?;
            if actual != spec.sha256 {
                return Err(AiError::Sha256Mismatch {
                    path: final_path.to_owned(),
                    expected: spec.sha256.to_owned(),
                    actual,
                });
            }
        }

        fs::rename(&part_path, final_path)?;
        Ok(())
    }

    /// Streams the model body into `part_path`, resuming at `offset` when the
    /// server honors `Range`. Servers that ignore the range (or reject it with
    /// `416`) cause one restart from zero.
    async fn fetch_into(
        &self,
        spec: &ModelSpec,
        part_path: &Path,
        mut offset: u64,
    ) -> Result<(), AiError> {
        let mut restarted = false;
        loop {
            let mut request = self.client.get(spec.url);
            if offset > 0 {
                request = request.header(header::RANGE, format!("bytes={offset}-"));
            }
            let mut response = request.send().await.map_err(|source| AiError::Download {
                name: spec.name.to_owned(),
                source,
            })?;

            match response.status() {
                reqwest::StatusCode::PARTIAL_CONTENT => {}
                reqwest::StatusCode::OK => offset = 0,
                reqwest::StatusCode::RANGE_NOT_SATISFIABLE if !restarted => {
                    offset = 0;
                    restarted = true;
                    continue;
                }
                reqwest::StatusCode::RANGE_NOT_SATISFIABLE => {
                    // The server considers the part complete; let the digest
                    // pass decide whether it is actually intact.
                    return Ok(());
                }
                status => {
                    return Err(AiError::HttpStatus {
                        name: spec.name.to_owned(),
                        status: status.as_u16(),
                    });
                }
            }

            let mut file = if offset == 0 {
                File::create(part_path)?
            } else {
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(false)
                    .open(part_path)?;
                file.seek(SeekFrom::Start(offset))?;
                file
            };

            loop {
                let chunk = response.chunk().await.map_err(|source| AiError::Download {
                    name: spec.name.to_owned(),
                    source,
                })?;
                match chunk {
                    Some(bytes) => file.write_all(&bytes)?,
                    None => break,
                }
            }
            file.sync_all()?;
            return Ok(());
        }
    }
}

/// `<name>.bin` → `<name>.bin.part`.
fn part_path(final_path: &Path) -> PathBuf {
    let mut name = final_path.as_os_str().to_owned();
    name.push(".part");
    final_path.with_file_name(name)
}

/// Computes the lowercase-hex SHA-256 digest of `path`.
pub fn sha256_hex(path: &Path) -> Result<String, AiError> {
    let mut hasher = Sha256::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = [0_u8; HASH_BUFFER_SIZE];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(&hasher.finalize()))
}

/// Lowercase-hex rendering of a digest.
fn hex_digest(digest: &[u8]) -> String {
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from_digit(u32::from(byte >> 4), 16).unwrap());
        hex.push(char::from_digit(u32::from(byte & 0x0F), 16).unwrap());
    }
    hex
}

/// Whether `path` exists and digests to `expected_hex` (case-insensitive).
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<bool, AiError> {
    if !path.is_file() {
        return Ok(false);
    }
    Ok(sha256_hex(path)?.eq_ignore_ascii_case(expected_hex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    /// Known digest of the ASCII string `hello\n`.
    const HELLO_SHA256: &str = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";

    // ---------------------------------------------------------------- hashing

    #[test]
    fn sha256_hex_matches_known_vector() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hello.txt");
        fs::write(&path, b"hello\n").unwrap();
        assert_eq!(sha256_hex(&path).unwrap(), HELLO_SHA256);
    }

    #[test]
    fn sha256_hex_reports_missing_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing.bin");
        assert!(matches!(sha256_hex(&path), Err(AiError::Io(_))));
    }

    #[test]
    fn verify_compares_case_insensitively_and_handles_missing_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hello.txt");
        fs::write(&path, b"hello\n").unwrap();

        assert!(verify_sha256(&path, &HELLO_SHA256.to_uppercase()).unwrap());
        assert!(!verify_sha256(&path, "deadbeef").unwrap());

        let missing = directory.path().join("missing.bin");
        assert!(!verify_sha256(&missing, HELLO_SHA256).unwrap());
    }

    #[test]
    fn known_models_have_sane_urls_and_digests() {
        for spec in MODELS {
            assert!(
                spec.url.starts_with("https://"),
                "bad url for {}",
                spec.name
            );
            assert_eq!(spec.sha256.len(), 64, "bad digest for {}", spec.name);
            assert!(
                spec.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "digest for {} is not hex",
                spec.name
            );
        }
        assert_eq!(ModelSpec::find("tiny.en").unwrap().name, "tiny.en");
        assert!(ModelSpec::find("nope").is_none());
    }

    // ------------------------------------------------------------ mock server

    /// Minimal HTTP/1.1 server that serves one payload and answers `Range`
    /// requests, used to exercise downloads without the network.
    struct MockServer {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
        shutdown: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start(payload: Vec<u8>, not_found: bool) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let url = format!("http://{}/model.bin", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let shutdown = Arc::new(AtomicBool::new(false));
            let handle = {
                let requests = Arc::clone(&requests);
                let shutdown = Arc::clone(&shutdown);
                thread::spawn(move || {
                    listener.set_nonblocking(true).unwrap();
                    while !shutdown.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                handle_connection(stream, &payload, not_found, &requests)
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(5));
                            }
                            Err(_) => break,
                        }
                    }
                })
            };
            Self {
                url,
                requests,
                shutdown,
                handle: Some(handle),
            }
        }

        fn request_log(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn handle_connection(
        mut stream: TcpStream,
        payload: &[u8],
        not_found: bool,
        requests: &Mutex<Vec<String>>,
    ) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let mut head = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !head.windows(4).any(|window| window == b"\r\n\r\n") && head.len() < 64 * 1024 {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => head.extend_from_slice(&buffer[..read]),
                Err(_) => break,
            }
        }
        let head = String::from_utf8_lossy(&head);
        let request_line = head.lines().next().unwrap_or_default().to_owned();
        let range = head
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("range:"))
            .map(str::to_owned);
        requests
            .lock()
            .unwrap()
            .push(format!("{request_line} | {range:?}"));

        let response = if not_found {
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
        } else {
            match range
                .as_deref()
                .and_then(|line| line.split("bytes=").nth(1))
                .and_then(|value| value.split('-').next())
                .and_then(|value| value.parse::<usize>().ok())
            {
                Some(start) if start < payload.len() => {
                    let body = &payload[start..];
                    let mut response = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: \
                         bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                        body.len(),
                        start,
                        payload.len() - 1,
                        payload.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(body);
                    response
                }
                Some(_) => b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n\
                             Connection: close\r\n\r\n"
                    .to_vec(),
                None => {
                    let mut response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        payload.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(payload);
                    response
                }
            }
        };
        let _ = stream.write_all(&response);
    }

    /// Deterministic 70 KiB payload.
    fn payload() -> Vec<u8> {
        (0..70_000)
            .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
            .collect()
    }

    /// Builds a spec pointing at the mock server, with the payload's digest.
    fn mock_spec(server: &MockServer, sha256: impl Into<String>) -> ModelSpec {
        // Tests only; leaking a small string to get 'static is fine.
        let url: &'static str = Box::leak(server.url.clone().into_boxed_str());
        let sha256: &'static str = Box::leak(sha256.into().into_boxed_str());
        ModelSpec {
            name: "mock",
            file_name: "mock.bin",
            url,
            sha256,
        }
    }

    fn digest_of(payload: &[u8]) -> String {
        hex_digest(&Sha256::digest(payload))
    }

    fn manager() -> (ModelManager, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let paths = SublayerPaths::new(
            dir.path().join("data"),
            dir.path().join("config"),
            dir.path().join("cache"),
        );
        // Bypass proxy environment variables so localhost requests stay local.
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client build");
        (ModelManager::with_client(paths, client), dir)
    }

    #[tokio::test]
    async fn fresh_download_lands_in_final_path() {
        let server = MockServer::start(payload(), false);
        let (manager, _dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));

        let path = manager.cached_path_for(&spec).await.unwrap();

        let data = fs::read(&path).unwrap();
        assert_eq!(data, payload());
        assert_eq!(path.file_name().unwrap(), "mock.bin");
        let log = server.request_log();
        assert_eq!(log.len(), 1, "expected exactly one GET: {log:?}");
        assert!(log[0].starts_with("GET /model.bin"), "{log:?}");
    }

    #[tokio::test]
    async fn interrupted_download_resumes_with_range() {
        let server = MockServer::start(payload(), false);
        let (manager, dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));
        let final_path = manager.paths.model_path("mock.bin");

        // Simulate a previously interrupted download.
        let part = dir.path().join("data").join("models").join("mock.bin.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, &payload()[..50_000]).unwrap();

        manager.download_spec(&spec, &final_path).await.unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), payload());
        let log = server.request_log();
        assert_eq!(log.len(), 1, "expected a single resumed GET: {log:?}");
        assert!(
            log[0].contains("bytes=50000-"),
            "expected a Range header, got {log:?}"
        );
    }

    #[tokio::test]
    async fn existing_valid_model_skips_download() {
        let server = MockServer::start(payload(), false);
        let (manager, _dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));
        let final_path = manager.paths.model_path("mock.bin");
        fs::create_dir_all(final_path.parent().unwrap()).unwrap();
        fs::write(&final_path, payload()).unwrap();

        let path = manager.cached_path_for(&spec).await.unwrap();

        assert_eq!(path, final_path);
        assert!(
            server.request_log().is_empty(),
            "network must not be touched"
        );
    }

    #[tokio::test]
    async fn corrupt_existing_model_is_replaced() {
        let server = MockServer::start(payload(), false);
        let (manager, _dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));
        let final_path = manager.paths.model_path("mock.bin");
        fs::create_dir_all(final_path.parent().unwrap()).unwrap();
        fs::write(&final_path, b"garbage that fails the digest").unwrap();

        let path = manager.cached_path_for(&spec).await.unwrap();

        assert_eq!(fs::read(path).unwrap(), payload());
        assert_eq!(server.request_log().len(), 1);
    }

    #[tokio::test]
    async fn corrupt_partial_file_restarts_from_scratch() {
        let server = MockServer::start(payload(), false);
        let (manager, dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));
        let final_path = manager.paths.model_path("mock.bin");

        // A stale part whose digest can never match the payload.
        let part = dir.path().join("data").join("models").join("mock.bin.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, vec![0xAB; 50_000]).unwrap();

        manager.download_spec(&spec, &final_path).await.unwrap();

        assert_eq!(fs::read(&final_path).unwrap(), payload());
        let log = server.request_log();
        assert!(
            log.len() >= 2,
            "corrupt part must trigger a full restart, got {log:?}"
        );
    }

    #[tokio::test]
    async fn digest_mismatch_after_retry_is_an_error() {
        let server = MockServer::start(payload(), false);
        let (manager, _dir) = manager();
        // Spec claims a digest that never matches the served payload.
        let spec = mock_spec(&server, digest_of(b"different payload"));

        let error = manager
            .download_spec(&spec, &manager.paths.model_path("mock.bin"))
            .await
            .unwrap_err();

        assert!(
            matches!(error, AiError::Sha256Mismatch { .. }),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn http_errors_surface_as_status() {
        let server = MockServer::start(Vec::new(), true);
        let (manager, _dir) = manager();
        let spec = mock_spec(&server, "00".repeat(32));

        let error = manager
            .download_spec(&spec, &manager.paths.model_path("mock.bin"))
            .await
            .unwrap_err();

        assert!(
            matches!(error, AiError::HttpStatus { status: 404, .. }),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn unknown_model_names_are_rejected() {
        let (manager, _dir) = manager();
        let error = manager.ensure_cached("whisper-XL").await.unwrap_err();
        assert!(matches!(error, AiError::UnknownModel(_)), "got {error:?}");
    }

    #[tokio::test]
    async fn complete_part_is_restarted_and_verified() {
        // A part file that is already complete-size gets rejected with `416` by
        // the server; the download restarts from zero and only the final
        // digest guarantees the file is intact.
        let server = MockServer::start(payload(), false);
        let (manager, dir) = manager();
        let spec = mock_spec(&server, digest_of(&payload()));
        let final_path = manager.paths.model_path("mock.bin");

        let part = dir.path().join("data").join("models").join("mock.bin.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, payload()).unwrap();

        manager.download_spec(&spec, &final_path).await.unwrap();

        let log = server.request_log();
        assert_eq!(log.len(), 2, "416 must restart the download: {log:?}");
        assert!(log[0].contains("bytes=70000-"), "{log:?}");
        assert!(
            !log[1].contains("Range"),
            "restart must be a plain GET: {log:?}"
        );
        assert_eq!(fs::read(&final_path).unwrap(), payload());
    }
}
