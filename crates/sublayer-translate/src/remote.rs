//! Remote translation backend over the OpenAI-compatible chat-completions
//! API.
//!
//! Points at any server speaking that protocol, from hosted providers
//! (OpenAI, Groq, ...) to local ones (LM Studio, Ollama, llama.cpp, vLLM),
//! which keeps the backend usable both online and fully on-premises. The API
//! key is optional so keyless local servers work out of the box; when set it
//! is sent as a `Bearer` token.
//!
//! Sentences are translated one request at a time (each gets its own user
//! message) so outputs stay independent and order-stable. Every request is
//! bounded by a connect and a total timeout.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{TranslateError, TranslationBackend};

/// Timeouts for every translation request (connect + whole round trip).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Longest response body excerpt kept in errors.
const BODY_EXCERPT_CHARS: usize = 500;

/// Configuration for the OpenAI-compatible backend.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteTranslateConfig {
    /// Server base URL; the path is normalized:
    /// `"https://host/v1"` and `"https://host/v1/"` both post to
    /// `https://host/v1/chat/completions`, and an already full
    /// `.../chat/completions` URL is used as-is.
    pub base_url: String,
    /// Optional `Authorization: Bearer <token>`; `None` for keyless local
    /// servers.
    pub api_key: Option<String>,
    /// Model name reported in the request body (e.g. `"gpt-4o-mini"`,
    /// `"qwen2.5:7b"`).
    pub model: String,
    /// Target language, used in the system prompt (a name works best for
    /// small models, e.g. `"German"`).
    pub target_language: String,
}

/// OpenAI-compatible chat-completions translation backend.
#[derive(Clone, Debug)]
pub struct RemoteBackend {
    client: reqwest::blocking::Client,
    config: RemoteTranslateConfig,
    endpoint: String,
    system_prompt: String,
}

impl RemoteBackend {
    /// Builds the HTTP client (with timeouts) and normalizes the endpoint.
    pub fn new(config: RemoteTranslateConfig) -> Result<Self, TranslateError> {
        if config.base_url.trim().is_empty() {
            return Err(TranslateError::Load(
                "base_url must not be empty".to_owned(),
            ));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;
        let system_prompt = format!(
            "You translate subtitles into {}. Reply with the translation \
             only: no quotes, no notes, no explanations.",
            config.target_language
        );
        Ok(Self {
            client,
            endpoint: chat_endpoint(&config.base_url),
            config,
            system_prompt,
        })
    }

    fn translate_one(&self, sentence: &str) -> Result<String, TranslateError> {
        if sentence.trim().is_empty() {
            return Ok(String::new());
        }
        let request = ChatRequest {
            model: &self.config.model,
            temperature: 0,
            messages: vec![
                Message {
                    role: "system",
                    content: &self.system_prompt,
                },
                Message {
                    role: "user",
                    content: sentence,
                },
            ],
        };
        let mut builder = self.client.post(&self.endpoint).json(&request);
        if let Some(api_key) = &self.config.api_key {
            builder = builder.bearer_auth(api_key);
        }
        let response = builder.send().map_err(TranslateError::Http)?;
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            return Err(TranslateError::HttpStatus {
                status: status.as_u16(),
                detail: excerpt(&body),
            });
        }
        let parsed: ChatResponse = serde_json::from_str(&body)
            .map_err(|error| TranslateError::InvalidResponse(format!("{error}: {body}")))?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .ok_or_else(|| TranslateError::InvalidResponse(format!("no choices in {body}")))
    }
}

impl TranslationBackend for RemoteBackend {
    fn name(&self) -> &str {
        "openai-compatible"
    }

    fn source_language(&self) -> Option<&str> {
        // The chat model accepts any source language.
        None
    }

    fn target_language(&self) -> Option<&str> {
        Some(&self.config.target_language)
    }

    fn translate(&self, sentences: &[String]) -> Result<Vec<String>, TranslateError> {
        let mut out = Vec::with_capacity(sentences.len());
        for sentence in sentences {
            out.push(self.translate_one(sentence)?);
        }
        Ok(out)
    }
}

/// Normalizes a base URL to the chat-completions endpoint.
fn chat_endpoint(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/chat/completions") {
        base.to_owned()
    } else {
        format!("{base}/chat/completions")
    }
}

/// Keeps the tail of a response body for error messages.
fn excerpt(body: &str) -> String {
    let body = body.trim();
    let characters = body.chars().count();
    if characters <= BODY_EXCERPT_CHARS {
        return body.to_owned();
    }
    body.chars().skip(characters - BODY_EXCERPT_CHARS).collect()
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    temperature: u8,
    messages: Vec<Message<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageContent,
}

#[derive(Deserialize)]
struct MessageContent {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    /// One recorded HTTP request.
    #[derive(Debug, Clone)]
    struct Request {
        method: String,
        path: String,
        headers: HashMap<String, String>,
        body: String,
    }

    /// Tiny HTTP server for the tests: answers one request per connection
    /// with the closure's response, records every request, and shuts down on
    /// drop.
    struct MockServer {
        addr: SocketAddr,
        requests: Arc<Mutex<Vec<Request>>>,
        shutdown: Arc<std::sync::atomic::AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start(respond: impl Fn(&Request) -> (u16, String) + Send + Sync + 'static) -> Self {
            let respond = Arc::new(respond);
            let requests = Arc::new(Mutex::new(Vec::new()));
            let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
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
                let requests = Arc::clone(&requests);
                let shutdown = Arc::clone(&shutdown);
                std::thread::spawn(move || {
                    while !shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                let requests = Arc::clone(&requests);
                                let respond = Arc::clone(&respond);
                                std::thread::spawn(move || {
                                    if let Ok(Some(request)) = read_request(&stream) {
                                        requests.lock().unwrap().push(request.clone());
                                        let (status, body) = respond(&request);
                                        let _ = write_response(&stream, status, &body);
                                    }
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
            format!("http://{}/v1", self.addr)
        }

        fn request_log(&self) -> Vec<Request> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.shutdown
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    /// Reads one HTTP request (headers + body) from the stream.
    fn read_request(stream: &TcpStream) -> std::io::Result<Option<Request>> {
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
        let mut reader = stream.try_clone()?;
        let mut head = Vec::new();
        let mut byte = [0_u8; 1];
        let mut header_end = None;
        // Bounded header scan (64 KiB is far beyond any real request here).
        while head.len() < 64 * 1024 && reader.read(&mut byte)? == 1 {
            head.push(byte[0]);
            if head.len() >= 4 && head[head.len() - 4..] == *b"\r\n\r\n" {
                header_end = Some(head.len());
                break;
            }
        }
        let Some(header_end) = header_end else {
            return Ok(None);
        };
        let header_text = String::from_utf8_lossy(&head[..header_end]).into_owned();
        let mut lines = header_text.split("\r\n");
        let request_line = lines.next().unwrap_or_default();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_owned();
        let path = parts.next().unwrap_or_default().to_owned();
        let mut headers = HashMap::new();
        let mut content_length = 0_usize;
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim().to_owned();
                if name == "content-length" {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.insert(name, value);
            }
        }
        let mut body = vec![0_u8; content_length];
        reader.read_exact(&mut body)?;
        Ok(Some(Request {
            method,
            path,
            headers,
            body: String::from_utf8_lossy(&body).into_owned(),
        }))
    }

    /// Writes `HTTP/1.1 {status}` with the body and closes the connection.
    fn write_response(stream: &TcpStream, status: u16, body: &str) -> std::io::Result<()> {
        let reason = match status {
            200 => "OK",
            500 => "Internal Server Error",
            _ => "Error",
        };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.set_write_timeout(Some(std::time::Duration::from_secs(10)))?;
        stream.try_clone()?.write_all(response.as_bytes())
    }

    fn config(base_url: String) -> RemoteTranslateConfig {
        RemoteTranslateConfig {
            base_url,
            api_key: Some("secret".to_owned()),
            model: "test-model".to_owned(),
            target_language: "English".to_owned(),
        }
    }

    /// Translates by echoing `user` content with a prefix, so tests can prove
    /// both the request shape and the output order.
    fn echo_translation(request: &Request) -> (u16, String) {
        let value: serde_json::Value = serde_json::from_str(&request.body).unwrap();
        let sentence = value["messages"][1]["content"].as_str().unwrap_or_default();
        (
            200,
            format!(
                r#"{{"choices":[{{"message":{{"role":"assistant","content":"TR: {sentence}"}}}}]}}"#
            ),
        )
    }

    #[test]
    fn translates_sentences_in_order_and_shapes_the_request() {
        let server = MockServer::start(echo_translation);
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();

        let out = backend
            .translate(&["Hallo".to_owned(), "Welt".to_owned()])
            .unwrap();
        assert_eq!(out, ["TR: Hallo", "TR: Welt"]);

        let log = server.request_log();
        assert_eq!(log.len(), 2);
        for request in &log {
            assert_eq!(request.method, "POST");
            assert_eq!(request.path, "/v1/chat/completions");
            assert_eq!(
                request.headers.get("authorization").unwrap(),
                "Bearer secret"
            );
            let value: serde_json::Value = serde_json::from_str(&request.body).unwrap();
            assert_eq!(value["model"], "test-model");
            assert_eq!(value["temperature"], 0);
            assert_eq!(value["messages"][0]["role"], "system");
            assert!(
                value["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("English")
            );
        }
        assert_eq!(backend.target_language(), Some("English"));
        assert_eq!(backend.source_language(), None, "remote accepts any source");
        assert!(
            log[1].body.contains("Welt"),
            "second request carries the second sentence: {}",
            log[1].body
        );
    }

    #[test]
    fn api_key_is_optional_for_keyless_servers() {
        let server = MockServer::start(echo_translation);
        let mut cfg = config(server.base_url());
        cfg.api_key = None;
        let backend = RemoteBackend::new(cfg).unwrap();

        backend.translate(&["Hallo".to_owned()]).unwrap();
        let log = server.request_log();
        assert!(!log[0].headers.contains_key("authorization"));
    }

    #[test]
    fn non_success_status_surfaces_with_a_body_excerpt() {
        let server = MockServer::start(|_| (500, "backend exploded".to_owned()));
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();

        let error = backend
            .translate(&["Hallo".to_owned()])
            .expect_err("500 must fail");
        match error {
            TranslateError::HttpStatus { status, detail } => {
                assert_eq!(status, 500);
                assert_eq!(detail, "backend exploded");
            }
            other => panic!("expected HttpStatus, got {other:?}"),
        }
    }

    #[test]
    fn malformed_bodies_are_invalid_responses() {
        let server = MockServer::start(|_| (200, "not json at all".to_owned()));
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();

        let error = backend
            .translate(&["Hallo".to_owned()])
            .expect_err("garbage must fail");
        assert!(matches!(error, TranslateError::InvalidResponse(_)));
    }

    #[test]
    fn empty_choices_are_invalid_responses() {
        let server = MockServer::start(|_| (200, r#"{"choices":[]}"#.to_owned()));
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();

        let error = backend
            .translate(&["Hallo".to_owned()])
            .expect_err("no choices must fail");
        assert!(matches!(error, TranslateError::InvalidResponse(_)));
    }

    #[test]
    fn blank_sentences_are_skipped_without_requests() {
        let server = MockServer::start(echo_translation);
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();

        let out = backend
            .translate(&["".to_owned(), " ".to_owned(), "Hallo".to_owned()])
            .unwrap();
        assert_eq!(out, ["", "", "TR: Hallo"]);
        assert_eq!(server.request_log().len(), 1);
    }

    #[test]
    fn empty_batches_are_noops() {
        let server = MockServer::start(echo_translation);
        let backend = RemoteBackend::new(config(server.base_url())).unwrap();
        assert!(backend.translate(&[]).unwrap().is_empty());
        assert!(server.request_log().is_empty());
    }

    #[test]
    fn endpoint_normalization_handles_slashes_and_full_urls() {
        assert_eq!(
            chat_endpoint("https://host/v1"),
            "https://host/v1/chat/completions"
        );
        assert_eq!(
            chat_endpoint("https://host/v1/"),
            "https://host/v1/chat/completions"
        );
        assert_eq!(
            chat_endpoint("https://host/v1/chat/completions"),
            "https://host/v1/chat/completions"
        );
    }

    #[test]
    fn empty_base_url_is_rejected() {
        let error = RemoteBackend::new(config(String::new())).expect_err("must fail");
        assert!(matches!(error, TranslateError::Load(_)));
    }

    #[test]
    fn long_error_bodies_are_bounded() {
        let long = "x".repeat(BODY_EXCERPT_CHARS + 50);
        assert_eq!(excerpt(&long).chars().count(), BODY_EXCERPT_CHARS);
    }
}
