//! Pinned automatic downloads of the bundled translation bundles.
//!
//! The six bundled pairs (see [`crate::LOCAL_PAIRS`]) are fetched from
//! Mozilla's Remote Settings on demand and verified against SHA-256 digests
//! pinned in this source:
//!
//! - File **digests are pinned in code**, so a downloaded model is
//!   byte-identical to the one the workspace was tested against.
//! - **URLs are discovered at runtime** from the Remote Settings record
//!   collection by matching the record's `attachment.hash` to the pin, so
//!   upstream file moves never break a download.
//! - If Mozilla *replaces* a model (its hash disappears from the records),
//!   the download fails with a clear message instead of silently shipping
//!   untested weights.
//!
//! Downloads land in `<root>/<prefix>/` with the canonical record file names
//! (`model.koen.intgemm.alphas.bin`, ...), i.e. exactly the layout
//! [`crate::local::LocalBackend`] discovers.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::{LocalPair, TranslateError};

/// Remote Settings collection holding the Firefox Translations models.
const RECORDS_URL: &str = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records";
/// Attachment storage for that collection.
const CDN_BASE: &str = "https://firefox-settings-attachments.cdn.mozilla.net";
/// Bound for the whole records fetch and for each file download.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// One pinned record file: canonical name in the bundle directory + the
/// SHA-256 of the attachment it was tested with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedFile {
    pub name: String,
    pub sha256: String,
}

/// The full file set of one bundled pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedBundle {
    pub files: Vec<PinnedFile>,
}

/// `(model prefix, pinned files)`; digests were computed from the bundles the
/// workspace translated with end to end (see `local`'s ignored test).
const PIN_TABLE: &[(&str, &[(&str, &str)])] = &[
    (
        "zhen",
        &[
            (
                "model.zhen.intgemm.alphas.bin",
                "0aee91790894458f5d367551f6edcd4c9cb97852c34f221bcbf9f4701ebcf0cd",
            ),
            (
                "lex.50.50.zhen.s2t.bin",
                "aa7daf6cfc85c0cd2c10e2944d66f6da55497c9c6408789f3adfded4074c2fb1",
            ),
            (
                "srcvocab.zhen.spm",
                "5cc6a76611dbf86219f109141533606b15ecb34eee83673bb86b2c16b14734db",
            ),
            (
                "trgvocab.zhen.spm",
                "7bf002db37c10d3b114cc5588d7fdcb16c57d0fd1e2c34354c22cc9f0b6c3c29",
            ),
        ],
    ),
    (
        "jaen",
        &[
            (
                "model.jaen.intgemm.alphas.bin",
                "3a603e20bfe1be86071913f9e23ab5129075bc0a8490151020ac4821e4f17302",
            ),
            (
                "lex.50.50.jaen.s2t.bin",
                "525f412f0d210536c2933c78ae395fa0bf2b5ee6cc5dda61ebc2e79410ebaee4",
            ),
            (
                "vocab.jaen.spm",
                "5cb217758bae05877bb3f0c2f612e4e7c1e4cb03c10db11f4a47098d7ae62919",
            ),
        ],
    ),
    (
        "koen",
        &[
            (
                "model.koen.intgemm.alphas.bin",
                "7cb30cbc0a86e242084756445268a2346136771c92b884d9b4ebb077f67d5507",
            ),
            (
                "lex.50.50.koen.s2t.bin",
                "b26291f063ce216a74239275f0485534a38111203e59d42a2a56ba62d6eeb196",
            ),
            (
                "vocab.koen.spm",
                "1c72b740ab793cdc3a8f16913dd6b4e806c77421077dd2d85edeb7be38418598",
            ),
        ],
    ),
    (
        "esen",
        &[
            (
                "model.esen.intgemm.alphas.bin",
                "4aed7734152ae0045d1a69ae49c86cfda18f53c61f90e95e1d1de1c7c7c3b033",
            ),
            (
                "lex.50.50.esen.s2t.bin",
                "e2610211d3b9577d012638fe7e7e74ed7b4b708ce96b9e792e67c282a6492daa",
            ),
            (
                "vocab.esen.spm",
                "5ae254fa9b15aa182e70fd2a6186b1333c63a29a48043a9224c6aa4fcac058ad",
            ),
        ],
    ),
    (
        "daen",
        &[
            (
                "model.daen.intgemm.alphas.bin",
                "2c1ace24220c905404bbb1de14bfeda8a391ef8d6c56f3b193a9d3e340d99b12",
            ),
            (
                "lex.50.50.daen.s2t.bin",
                "e5f50fba646e0d3cedf2957575aa84c3bdca6798ce32ccd63c956fc043d774a3",
            ),
            (
                "vocab.daen.spm",
                "b9392fcd9c53b802628d33a160f6eb52ca3d25a07ec9e015ab93b6bd5c7e8033",
            ),
        ],
    ),
    (
        "ruen",
        &[
            (
                "model.ruen.intgemm.alphas.bin",
                "aa0efa194b3dbd2da129ba0b8db8ef139126cb0914e57072f62061294746f1e2",
            ),
            (
                "lex.50.50.ruen.s2t.bin",
                "c944e753f298bbdf6506388de7257fb48cbd12c83e2807e2c284fc323771895b",
            ),
            (
                "vocab.ruen.spm",
                "de8af5f87e231ee4016cec4bb482e6b584cb1ca34dcb624fc7b8398f25bb35df",
            ),
        ],
    ),
];

/// Pinned bundle for a bundled pair, if one exists.
pub fn pinned_bundle(pair: &LocalPair) -> Option<PinnedBundle> {
    PIN_TABLE
        .iter()
        .find(|(prefix, _)| *prefix == pair.model_prefix)
        .map(|(_, files)| PinnedBundle {
            files: files
                .iter()
                .map(|(name, sha256)| PinnedFile {
                    name: (*name).to_owned(),
                    sha256: (*sha256).to_owned(),
                })
                .collect(),
        })
}

/// Downloads and verifies the bundled pairs into `root` (on demand).
pub struct TranslationModelManager {
    root: PathBuf,
    records_url: String,
    cdn_base: String,
    client: reqwest::Client,
}

impl TranslationModelManager {
    /// Manager storing bundles under `root` (`<root>/<prefix>/` per pair).
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, TranslateError> {
        Self::new_with_urls(root, RECORDS_URL, CDN_BASE)
    }

    fn new_with_urls(
        root: impl Into<PathBuf>,
        records_url: &str,
        cdn_base: &str,
    ) -> Result<Self, TranslateError> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(FETCH_TIMEOUT)
            .user_agent(concat!("sublayer/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            root: root.into(),
            records_url: records_url.to_owned(),
            cdn_base: cdn_base.to_owned(),
            client,
        })
    }

    /// Ensures the pinned bundle for `pair` is present and verified,
    /// downloading it otherwise; returns the bundle directory.
    pub async fn ensure_pair(&self, pair: &LocalPair) -> Result<PathBuf, TranslateError> {
        let bundle = pinned_bundle(pair).ok_or_else(|| {
            TranslateError::Load(format!(
                "no pinned download for pair `{}`",
                pair.model_prefix
            ))
        })?;
        self.ensure_bundle(pair.model_prefix, &bundle).await
    }

    /// Ensures `bundle` under `<root>/<prefix>/`; skips the network when the
    /// files are already present and verify against the pins.
    pub async fn ensure_bundle(
        &self,
        prefix: &str,
        bundle: &PinnedBundle,
    ) -> Result<PathBuf, TranslateError> {
        let dir = self.root.join(prefix);
        if bundle_verified(&dir, bundle) {
            return Ok(dir);
        }
        let locations = self.resolve_locations(bundle).await?;
        for file in &bundle.files {
            let location = locations.get(&file.name).ok_or_else(|| {
                TranslateError::Load(format!(
                    "the bundled `{prefix}` model changed upstream: no record \
                     matches the pinned digest of `{}`; pin it to a newer \
                     bundle or revert upstream changes",
                    file.name
                ))
            })?;
            self.download_file(file, location, &dir).await?;
        }
        Ok(dir)
    }

    /// Maps every pinned file name to its attachment location by matching
    /// `attachment.hash` in the records collection.
    async fn resolve_locations(
        &self,
        bundle: &PinnedBundle,
    ) -> Result<HashMap<String, String>, TranslateError> {
        let text = self
            .client
            .get(&self.records_url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        let records: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| TranslateError::InvalidResponse(format!("{error}")))?;
        let mut locations = HashMap::new();
        for record in records
            .get("data")
            .and_then(|data| data.as_array())
            .into_iter()
            .flatten()
        {
            let Some(attachment) = record.get("attachment").and_then(|a| a.as_object()) else {
                continue;
            };
            let (Some(hash), Some(location)) = (
                attachment.get("hash").and_then(|h| h.as_str()),
                attachment.get("location").and_then(|l| l.as_str()),
            ) else {
                continue;
            };
            if let Some(file) = bundle
                .files
                .iter()
                .find(|file| file.sha256.eq_ignore_ascii_case(hash))
            {
                locations.insert(file.name.clone(), location.to_owned());
            }
        }
        Ok(locations)
    }

    /// Streams one file from the CDN and verifies it against its pin before
    /// renaming it into place (a `.part` suffix keeps partial downloads from
    /// ever looking verified).
    async fn download_file(
        &self,
        file: &PinnedFile,
        location: &str,
        dir: &Path,
    ) -> Result<(), TranslateError> {
        let url = format!("{}/{location}", self.cdn_base);
        let bytes = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        let actual = sha256_hex(&bytes);
        if !actual.eq_ignore_ascii_case(&file.sha256) {
            return Err(TranslateError::Load(format!(
                "sha256 mismatch for `{}` (expected {}, got {actual})",
                file.name, file.sha256
            )));
        }
        std::fs::create_dir_all(dir)?;
        let part_path = dir.join(format!(".{}.part", file.name));
        std::fs::write(&part_path, &bytes)?;
        std::fs::rename(&part_path, dir.join(&file.name))?;
        Ok(())
    }
}

/// All `bundle.files` exist in `dir` and match their pins.
fn bundle_verified(dir: &Path, bundle: &PinnedBundle) -> bool {
    bundle
        .files
        .iter()
        .all(|file| match sha256_file(&dir.join(&file.name)) {
            Ok(actual) => actual.eq_ignore_ascii_case(&file.sha256),
            Err(_) => false,
        })
}

/// SHA-256 of a file, streamed in chunks.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(&hasher.finalize()))
}

/// SHA-256 of a byte slice as lowercase hex.
fn sha256_hex(data: &[u8]) -> String {
    hex_digest(&Sha256::digest(data))
}

/// Lowercase hex of a digest's bytes.
fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TranslationBackend;
    use std::collections::HashMap;
    use std::io::{BufRead, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};

    fn hash_of(data: &[u8]) -> String {
        hex_digest(&Sha256::digest(data))
    }

    /// Fixture bundle: build the pinned set + record routes from payloads.
    fn fixture(files: &[(&str, &[u8])]) -> (PinnedBundle, HashMap<String, Vec<u8>>) {
        let mut bundle = PinnedBundle { files: Vec::new() };
        let mut routes = HashMap::new();
        let mut records: Vec<serde_json::Value> = Vec::new();
        for (name, payload) in files {
            let hash = hash_of(payload);
            bundle.files.push(PinnedFile {
                name: (*name).to_owned(),
                sha256: hash.clone(),
            });
            routes.insert(format!("att/{name}"), payload.to_vec());
            records.push(serde_json::json!({
                "name": name,
                "attachment": {
                    "hash": hash,
                    "size": payload.len(),
                    "filename": name,
                    "location": format!("att/{name}"),
                },
            }));
        }
        routes.insert(
            "records".to_owned(),
            serde_json::to_vec(&serde_json::json!({ "data": records })).unwrap(),
        );
        (bundle, routes)
    }

    /// Tiny HTTP server mapping path → bytes; records every request path.
    struct MockServer {
        addr: SocketAddr,
        requests: Arc<Mutex<Vec<String>>>,
        shutdown: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start(routes: HashMap<String, Vec<u8>>) -> Self {
            let routes = Arc::new(routes);
            let requests = Arc::new(Mutex::new(Vec::new()));
            let shutdown = Arc::new(AtomicBool::new(false));
            let (listener, addr) = loop {
                match TcpListener::bind("127.0.0.1:0") {
                    Ok(listener) => {
                        let addr = listener.local_addr().unwrap();
                        break (listener, addr);
                    }
                    Err(_) => std::thread::yield_now(),
                }
            };
            listener.set_nonblocking(true).unwrap();
            let thread = {
                let routes = Arc::clone(&routes);
                let requests = Arc::clone(&requests);
                let shutdown = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    while !shutdown.load(AtomicOrdering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                let routes = Arc::clone(&routes);
                                let requests = Arc::clone(&requests);
                                std::thread::spawn(move || {
                                    let Some(path) = read_path(&stream) else {
                                        return;
                                    };
                                    requests.lock().unwrap().push(path.clone());
                                    let (status, body) = match routes.get(&path) {
                                        Some(body) => (200, body.clone()),
                                        None => (404, b"not found".to_vec()),
                                    };
                                    let _ = write_response(&stream, status, &body);
                                });
                            }
                            Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
                        }
                    }
                })
            };
            Self {
                addr,
                requests,
                shutdown,
                thread: Some(thread),
            }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.shutdown.store(true, AtomicOrdering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn read_path(stream: &TcpStream) -> Option<String> {
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .ok()?;
        let mut reader = std::io::BufReader::new(stream.try_clone().ok()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).ok()?;
        request_line
            .split_whitespace()
            .nth(1)
            .map(|path| path.trim_start_matches('/').to_owned())
    }

    fn write_response(stream: &TcpStream, status: u16, body: &[u8]) -> std::io::Result<()> {
        let reason = if status == 200 { "OK" } else { "Not Found" };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.set_write_timeout(Some(std::time::Duration::from_secs(10)))?;
        let mut writer = stream.try_clone()?;
        writer.write_all(response.as_bytes())?;
        writer.write_all(body)
    }

    fn manager(urls: (&str, &str)) -> (TranslationModelManager, tempfile::TempDir) {
        let root = tempfile::tempdir().unwrap();
        let manager = TranslationModelManager::new_with_urls(root.path(), urls.0, urls.1).unwrap();
        (manager, root)
    }

    #[tokio::test]
    async fn downloads_all_files_verified_and_reuses_them() {
        let (bundle, routes) = fixture(&[
            ("model.koen.intgemm.alphas.bin", b"MODEL".as_slice()),
            ("lex.50.50.koen.s2t.bin", b"LEX".as_slice()),
            ("vocab.koen.spm", b"VOCAB".as_slice()),
        ]);
        let server = MockServer::start(routes);
        let (manager, _root) = manager((
            &format!("{}/records", server.base_url()),
            &server.base_url(),
        ));

        let dir = manager.ensure_bundle("koen", &bundle).await.unwrap();
        assert_eq!(
            std::fs::read(dir.join("model.koen.intgemm.alphas.bin")).unwrap(),
            b"MODEL"
        );
        assert_eq!(
            std::fs::read(dir.join("lex.50.50.koen.s2t.bin")).unwrap(),
            b"LEX"
        );
        assert_eq!(std::fs::read(dir.join("vocab.koen.spm")).unwrap(), b"VOCAB");
        assert_eq!(
            server.requests(),
            [
                "records",
                "att/model.koen.intgemm.alphas.bin",
                "att/lex.50.50.koen.s2t.bin",
                "att/vocab.koen.spm"
            ]
        );

        // A second ensure must verify locally and touch no network.
        let before = server.requests().len();
        let again = manager.ensure_bundle("koen", &bundle).await.unwrap();
        assert_eq!(again, dir);
        assert_eq!(
            server.requests().len(),
            before,
            "verified bundles skip the network"
        );
    }

    #[tokio::test]
    async fn pre_verified_directories_never_touch_the_network() {
        let (bundle, routes) = fixture(&[("model.koen.intgemm.alphas.bin", b"MODEL".as_slice())]);
        let server = MockServer::start(routes);
        let (manager, _root) = manager((
            &format!("{}/records", server.base_url()),
            &server.base_url(),
        ));

        // Pre-place a valid file (and an unrelated stale one that must stay).
        let dir = manager.root.join("koen");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.koen.intgemm.alphas.bin"), b"MODEL").unwrap();
        std::fs::write(dir.join("leftover.part"), b"x").unwrap();

        manager.ensure_bundle("koen", &bundle).await.unwrap();
        assert!(
            server.requests().is_empty(),
            "no HTTP when already verified"
        );
        assert!(dir.join("leftover.part").exists(), "unrelated files stay");
    }

    #[tokio::test]
    async fn upstream_replacements_fail_with_a_clear_message() {
        let (bundle, _) = fixture(&[("model.koen.intgemm.alphas.bin", b"MODEL".as_slice())]);
        // Records exist but the attachment hash was "updated" away.
        let routes = HashMap::from([(
            "records".to_owned(),
            serde_json::to_vec(&serde_json::json!({
                "data": [{ "name": "model.koen.intgemm.alphas.bin", "attachment": {
                    "hash": "deadbeef", "location": "att/model.koen.intgemm.alphas.bin" } }]
            }))
            .unwrap(),
        )]);
        let server = MockServer::start(routes);
        let (manager, _root) = manager((
            &format!("{}/records", server.base_url()),
            &server.base_url(),
        ));

        let error = manager.ensure_bundle("koen", &bundle).await.unwrap_err();
        let message = error.to_string();
        assert!(message.contains("changed upstream"), "got: {message}");
        assert!(message.contains("model.koen.intgemm.alphas.bin"));
    }

    #[tokio::test]
    async fn corrupt_downloads_never_leave_a_verified_file() {
        let (bundle, mut routes) =
            fixture(&[("model.koen.intgemm.alphas.bin", b"MODEL".as_slice())]);
        // The server returns different bytes than the pin.
        routes.insert(
            "att/model.koen.intgemm.alphas.bin".to_owned(),
            b"TAMPERED".to_vec(),
        );
        routes.insert(
            "att/model.koen.intgemm.alphas.bin".to_owned(),
            b"TAMPERED".to_vec(),
        );
        let server = MockServer::start(routes);
        let (manager, _root) = manager((
            &format!("{}/records", server.base_url()),
            &server.base_url(),
        ));

        let error = manager.ensure_bundle("koen", &bundle).await.unwrap_err();
        assert!(
            error.to_string().contains("sha256 mismatch"),
            "got: {error}"
        );
        let dir = manager.root.join("koen");
        assert!(
            !dir.join("model.koen.intgemm.alphas.bin").exists(),
            "no final file on mismatch"
        );
    }

    #[test]
    fn the_six_bundled_pairs_have_complete_pins() {
        for pair in crate::LOCAL_PAIRS {
            let bundle = pinned_bundle(pair).expect("every bundled pair is pinnable");
            assert!(
                bundle.files.len() >= 3,
                "{}: model + lex + vocab",
                pair.model_prefix
            );
            for file in &bundle.files {
                assert_eq!(file.sha256.len(), 64, "{}: hex sha256", file.name);
                assert!(file.name.ends_with(".bin") || file.name.ends_with(".spm"));
            }
            let names: Vec<&str> = bundle.files.iter().map(|file| file.name.as_str()).collect();
            assert!(
                names
                    .iter()
                    .any(|name| name.ends_with(".intgemm.alphas.bin"))
            );
            assert!(names.iter().any(|name| name.ends_with(".s2t.bin")));
        }
        let zh = pinned_bundle(&crate::LOCAL_PAIRS[0]).unwrap();
        let zh_names: Vec<String> = zh.files.iter().map(|file| file.name.clone()).collect();
        assert!(zh_names.contains(&"srcvocab.zhen.spm".to_owned()));
        assert!(zh_names.contains(&"trgvocab.zhen.spm".to_owned()));
    }

    #[test]
    fn hex_digest_is_lowercase_and_padded() {
        assert_eq!(hex_digest(&[0xAB, 0x0F]), "ab0f");
        assert_eq!(hex_digest(&[]), "");
    }

    /// Real end-to-end against Mozilla's Remote Settings: downloads the
    /// Korean bundle into a temp directory and translates with it.
    ///
    /// ```text
    /// cargo test -p sublayer-translate --features download -- \
    ///   --ignored downloads_koen_from_the_live_collection
    /// ```
    #[ignore]
    #[tokio::test]
    async fn downloads_koen_from_the_live_collection() {
        let pair = crate::find_local_pair("ko").expect("ko is bundled");
        let root = tempfile::tempdir().unwrap();
        let manager = TranslationModelManager::new(root.path()).unwrap();
        let dir = manager
            .ensure_pair(pair)
            .await
            .expect("live download must succeed");
        eprintln!("downloaded to {}", dir.display());

        let backend = crate::local::LocalBackend::new(&dir, pair.language).unwrap();
        let out = backend
            .translate(&["오늘 날씨가 아주 좋아요.".to_owned()])
            .unwrap();
        eprintln!("translation: {:?}", out);
        assert_eq!(out.len(), 1);
        assert!(!out[0].trim().is_empty());
    }
}
