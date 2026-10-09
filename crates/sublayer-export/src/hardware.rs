//! Hardware encoder probing.
//!
//! Detection is intentionally conservative and dependency-free: a backend is
//! only reported when both the kernel device node exists *and* the local
//! FFmpeg build lists the encoder. That mirrors how the render will actually
//! behave, instead of promising acceleration that fails at spawn time.

use std::path::{Path, PathBuf};

use sublayer_media::ffmpeg;

use crate::ExportError;

/// Environment variable overriding the encoder selection.
pub const ENCODER_ENV: &str = "SUBLAYER_ENCODER";

/// Directory holding the DRM render nodes used by VA-API.
const DRI_DIR: &str = "/dev/dri";

/// Video encoder used for the burn-in render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HardwareEncoder {
    /// Intel & AMD hardware encoding through VA-API (`h264_vaapi`).
    Vaapi,
    /// NVIDIA hardware encoding through NVENC (`h264_nvenc`).
    Nvenc,
    /// Mesa hardware encoding through the Vulkan video extension
    /// (`h264_vulkan`); built only with the `encode_vulkan` feature, since
    /// unstable drivers have been observed to reset the GPU on encode.
    #[cfg(feature = "encode_vulkan")]
    Vulkan,
    /// Software encoding through `libx264`.
    #[default]
    Cpu,
}

impl HardwareEncoder {
    /// Human-readable name used in logs and status messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Vaapi => "VA-API",
            Self::Nvenc => "NVENC",
            #[cfg(feature = "encode_vulkan")]
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU (libx264)",
        }
    }

    /// The FFmpeg encoder name this backend maps to.
    pub fn encoder_name(self) -> &'static str {
        match self {
            Self::Vaapi => "h264_vaapi",
            Self::Nvenc => "h264_nvenc",
            #[cfg(feature = "encode_vulkan")]
            Self::Vulkan => "h264_vulkan",
            Self::Cpu => "libx264",
        }
    }

    /// Parses an encoder name (`vaapi`, `va-api`, `nvenc`, `vulkan`, `cpu`,
    /// `x264`, or any of the FFmpeg encoder names). `vulkan` resolves only
    /// when built with the `encode_vulkan` feature.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "vaapi" | "va-api" | "va" | "h264_vaapi" => Some(Self::Vaapi),
            "nvenc" | "nvidia" | "h264_nvenc" => Some(Self::Nvenc),
            #[cfg(feature = "encode_vulkan")]
            "vulkan" | "h264_vulkan" => Some(Self::Vulkan),
            "cpu" | "x264" | "libx264" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Encoder choice requested by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderPreference {
    /// Pick the best available backend (VA-API, then NVENC, then CPU; plus
    /// Vulkan when built with the `encode_vulkan` feature).
    #[default]
    Auto,
    /// Force a specific backend; failing when it is unavailable.
    Explicit(HardwareEncoder),
}

impl EncoderPreference {
    /// Parses a preference value; `auto` maps to [`EncoderPreference::Auto`].
    pub fn parse(value: &str) -> Option<Self> {
        if value.trim().eq_ignore_ascii_case("auto") {
            return Some(Self::Auto);
        }
        HardwareEncoder::parse(value).map(Self::Explicit)
    }

    /// Reads `SUBLAYER_ENCODER`; `Ok(None)` when unset or empty.
    pub fn from_env() -> Result<Option<Self>, ExportError> {
        let Some(value) = std::env::var(ENCODER_ENV).ok().filter(|v| !v.is_empty()) else {
            return Ok(None);
        };
        Self::parse(&value)
            .map(Some)
            .ok_or(ExportError::UnknownEncoder(value))
    }

    /// Label shown in status messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Explicit(encoder) => encoder.label(),
        }
    }
}

/// Encoders detected on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HardwareProbe {
    /// Whether VA-API can be used (render node plus `h264_vaapi`).
    pub vaapi: bool,
    /// Whether NVENC can be used (NVIDIA node plus `h264_nvenc`).
    pub nvenc: bool,
    /// Whether Mesa Vulkan video encoding can be used (render node plus
    /// `h264_vulkan`), only compiled with `encode_vulkan`. RADV needs a DRM
    /// node like VA-API does, and the render-node condition conveniently
    /// scopes this to Mesa platforms: NVIDIA GPUs (no render node) keep
    /// NVENC as their only probe.
    #[cfg(feature = "encode_vulkan")]
    pub vulkan: bool,
    /// VA-API render node a render should upload to, when present.
    pub vaapi_device: Option<PathBuf>,
}

impl HardwareProbe {
    /// Picks the encoder for `preference`.
    ///
    /// Auto-detection falls back down the chain VA-API → NVENC → CPU (plus
    /// Vulkan in `encode_vulkan` builds); an explicit choice that is
    /// unavailable is reported instead of being silently downgraded. Runtime
    /// failures of a probed encoder are handled by [`HardwareProbe::chain`]
    /// plus [`crate::ExportOptions::fallbacks`].
    pub fn select(&self, preference: EncoderPreference) -> Result<HardwareEncoder, ExportError> {
        self.chain(preference)?
            .into_iter()
            .next()
            .ok_or(ExportError::EncoderUnavailable("cpu"))
    }

    /// Encoders to attempt for `preference`, best first.
    ///
    /// `Auto` yields the whole available chain ending in CPU, so a render can
    /// retry when a *device* turns out not to support encoding even though the
    /// FFmpeg build lists the encoder — probing cannot see driver profile
    /// gaps, and failing there would otherwise abort a render the CPU could
    /// finish. An explicit choice yields exactly one encoder: `--encoder vaapi`
    /// never silently renders on the CPU.
    pub fn chain(
        &self,
        preference: EncoderPreference,
    ) -> Result<Vec<HardwareEncoder>, ExportError> {
        match preference {
            EncoderPreference::Auto => {
                let mut chain = Vec::new();
                if self.vaapi {
                    chain.push(HardwareEncoder::Vaapi);
                }
                if self.nvenc {
                    chain.push(HardwareEncoder::Nvenc);
                }
                #[cfg(feature = "encode_vulkan")]
                if self.vulkan {
                    chain.push(HardwareEncoder::Vulkan);
                }
                chain.push(HardwareEncoder::Cpu);
                Ok(chain)
            }
            EncoderPreference::Explicit(HardwareEncoder::Vaapi) if !self.vaapi => {
                Err(ExportError::EncoderUnavailable("vaapi"))
            }
            EncoderPreference::Explicit(HardwareEncoder::Nvenc) if !self.nvenc => {
                Err(ExportError::EncoderUnavailable("nvenc"))
            }
            #[cfg(feature = "encode_vulkan")]
            EncoderPreference::Explicit(HardwareEncoder::Vulkan) if !self.vulkan => {
                Err(ExportError::EncoderUnavailable("vulkan"))
            }
            EncoderPreference::Explicit(encoder) => Ok(vec![encoder]),
        }
    }
}

/// Probes the local machine for VA-API, NVENC, and Vulkan support.
///
/// The FFmpeg encoder list is fetched once; any failure to run FFmpeg is
/// reported, since a render could not start either.
pub async fn probe_hardware() -> Result<HardwareProbe, ExportError> {
    let encoders = ffmpeg_encoders().await?;
    let render_node = first_render_node(Path::new(DRI_DIR));
    Ok(probe_from(
        &encoders,
        render_node.as_deref(),
        nvidia_device_present(),
    ))
}

/// Pure combination step of [`probe_hardware`], split out for tests.
fn probe_from(
    encoder_list: &str,
    render_node: Option<&Path>,
    nvidia_device: bool,
) -> HardwareProbe {
    let has_encoder = |name: &str| encoder_list.split_whitespace().any(|token| token == name);
    let vaapi = render_node.is_some() && has_encoder("h264_vaapi");
    #[cfg(feature = "encode_vulkan")]
    let vulkan = render_node.is_some() && has_encoder("h264_vulkan");
    HardwareProbe {
        vaapi,
        nvenc: nvidia_device && has_encoder("h264_nvenc"),
        #[cfg(feature = "encode_vulkan")]
        vulkan,
        vaapi_device: if vaapi {
            render_node.map(Path::to_path_buf)
        } else {
            None
        },
    }
}

/// Runs `ffmpeg -encoders` and returns its stdout.
async fn ffmpeg_encoders() -> Result<String, ExportError> {
    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    let mut command = ffmpeg::command(&program);
    command.args(["-hide_banner", "-loglevel", "error", "-encoders"]);
    let output = ffmpeg::run(ffmpeg::FFMPEG, &mut command).await?;
    if !output.status.success() {
        return Err(ffmpeg::failure(ffmpeg::FFMPEG, output.status, &output.stderr).into());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Lowest-numbered `/dev/dri/renderD*` node, if the directory exists.
fn first_render_node(dir: &Path) -> Option<PathBuf> {
    let mut nodes: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("renderD"))
        })
        .collect();
    nodes.sort();
    nodes.into_iter().next()
}

/// Whether an NVIDIA kernel device is exposed to this process.
fn nvidia_device_present() -> bool {
    Path::new("/dev/nvidiactl").exists() || Path::new("/dev/nvidia0").exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCODER_LIST: &str = "Encoders:\n V..... h264_vaapi   VAAPI H.264\n V..... h264_nvenc   NVIDIA NVENC H.264\n V..... h264_vulkan  Vulkan H.264\n V..... libx264      libx264 H.264\n";

    #[test]
    fn encoder_names_round_trip_through_parse() {
        #[allow(unused_mut)]
        let mut encoders = vec![
            HardwareEncoder::Vaapi,
            HardwareEncoder::Nvenc,
            HardwareEncoder::Cpu,
        ];
        #[cfg(feature = "encode_vulkan")]
        encoders.push(HardwareEncoder::Vulkan);
        for encoder in encoders {
            assert_eq!(
                HardwareEncoder::parse(encoder.encoder_name()),
                Some(encoder)
            );
        }
        assert_eq!(
            HardwareEncoder::parse("VA-API"),
            Some(HardwareEncoder::Vaapi)
        );
        #[cfg(feature = "encode_vulkan")]
        {
            assert_eq!(
                HardwareEncoder::parse("vulkan"),
                Some(HardwareEncoder::Vulkan)
            );
            assert_eq!(
                HardwareEncoder::parse("h264_vulkan"),
                Some(HardwareEncoder::Vulkan)
            );
        }
        assert_eq!(HardwareEncoder::parse("x264"), Some(HardwareEncoder::Cpu));
        assert_eq!(HardwareEncoder::parse("av1"), None);
        assert_eq!(
            EncoderPreference::parse("auto"),
            Some(EncoderPreference::Auto)
        );
        assert_eq!(
            EncoderPreference::parse("nvenc"),
            Some(EncoderPreference::Explicit(HardwareEncoder::Nvenc))
        );
    }

    #[cfg(not(feature = "encode_vulkan"))]
    #[test]
    fn vulkan_is_not_parsed_without_the_feature() {
        assert_eq!(HardwareEncoder::parse("vulkan"), None);
        assert_eq!(HardwareEncoder::parse("h264_vulkan"), None);
    }

    #[test]
    fn probe_requires_both_device_and_encoder() {
        let node = Path::new("/dev/dri/renderD128");
        let both = probe_from(ENCODER_LIST, Some(node), true);
        assert!(both.vaapi && both.nvenc);
        #[cfg(feature = "encode_vulkan")]
        assert!(both.vulkan);

        // No render node: VA-API (and Vulkan, when compiled in) are out even
        // though FFmpeg supports them (Mesa drivers need a DRM node); NVENC
        // stays.
        let no_node = probe_from(ENCODER_LIST, None, true);
        assert!(!no_node.vaapi && no_node.nvenc);
        #[cfg(feature = "encode_vulkan")]
        assert!(!no_node.vulkan);

        // Old FFmpeg build without the encoders.
        let no_encoders = probe_from("Encoders:\n V..... libx264 H.264\n", Some(node), true);
        assert!(!no_encoders.vaapi && !no_encoders.nvenc);
        #[cfg(feature = "encode_vulkan")]
        assert!(!no_encoders.vulkan);

        // No NVIDIA device.
        let no_nvidia = probe_from(ENCODER_LIST, Some(node), false);
        assert!(no_nvidia.vaapi && !no_nvidia.nvenc);
        #[cfg(feature = "encode_vulkan")]
        assert!(no_nvidia.vulkan);
    }

    #[test]
    fn selection_falls_back_to_cpu_and_reports_explicit_misses() {
        let none = HardwareProbe::default();
        assert_eq!(
            none.select(EncoderPreference::Auto).unwrap(),
            HardwareEncoder::Cpu
        );
        assert!(matches!(
            none.select(EncoderPreference::Explicit(HardwareEncoder::Vaapi)),
            Err(ExportError::EncoderUnavailable("vaapi"))
        ));
        assert_eq!(
            none.select(EncoderPreference::Explicit(HardwareEncoder::Cpu))
                .unwrap(),
            HardwareEncoder::Cpu
        );

        let vaapi_only = HardwareProbe {
            vaapi: true,
            nvenc: false,
            #[cfg(feature = "encode_vulkan")]
            vulkan: false,
            vaapi_device: Some(PathBuf::from("/dev/dri/renderD128")),
        };
        assert_eq!(
            vaapi_only.select(EncoderPreference::Auto).unwrap(),
            HardwareEncoder::Vaapi
        );
        assert_eq!(
            vaapi_only.vaapi_device.as_deref(),
            Some(Path::new("/dev/dri/renderD128"))
        );

        let nvenc_only = HardwareProbe {
            vaapi: false,
            nvenc: true,
            #[cfg(feature = "encode_vulkan")]
            vulkan: false,
            vaapi_device: None,
        };
        assert_eq!(
            nvenc_only.select(EncoderPreference::Auto).unwrap(),
            HardwareEncoder::Nvenc
        );

        // Vulkan is the rescue when VA-API is missing entirely.
        #[cfg(feature = "encode_vulkan")]
        {
            let vulkan_only = HardwareProbe {
                vaapi: false,
                nvenc: false,
                vulkan: true,
                vaapi_device: None,
            };
            assert_eq!(
                vulkan_only.select(EncoderPreference::Auto).unwrap(),
                HardwareEncoder::Vulkan
            );
            assert!(matches!(
                vulkan_only.select(EncoderPreference::Explicit(HardwareEncoder::Vaapi)),
                Err(ExportError::EncoderUnavailable("vaapi"))
            ));
            assert!(matches!(
                HardwareProbe::default()
                    .select(EncoderPreference::Explicit(HardwareEncoder::Vulkan)),
                Err(ExportError::EncoderUnavailable("vulkan"))
            ));
        }
    }

    #[test]
    fn chain_orders_auto_and_pins_explicit_choices() {
        let all = HardwareProbe {
            vaapi: true,
            nvenc: true,
            #[cfg(feature = "encode_vulkan")]
            vulkan: true,
            vaapi_device: Some(PathBuf::from("/dev/dri/renderD128")),
        };
        let mut expected = vec![HardwareEncoder::Vaapi, HardwareEncoder::Nvenc];
        #[cfg(feature = "encode_vulkan")]
        expected.push(HardwareEncoder::Vulkan);
        expected.push(HardwareEncoder::Cpu);
        assert_eq!(all.chain(EncoderPreference::Auto).unwrap(), expected);
        // Explicit choices never carry fallbacks: the user asked for one backend.
        assert_eq!(
            all.chain(EncoderPreference::Explicit(HardwareEncoder::Nvenc))
                .unwrap(),
            vec![HardwareEncoder::Nvenc]
        );

        let nvenc_only = HardwareProbe {
            vaapi: false,
            nvenc: true,
            #[cfg(feature = "encode_vulkan")]
            vulkan: false,
            vaapi_device: None,
        };
        assert_eq!(
            nvenc_only.chain(EncoderPreference::Auto).unwrap(),
            vec![HardwareEncoder::Nvenc, HardwareEncoder::Cpu]
        );

        #[cfg(feature = "encode_vulkan")]
        {
            let vulkan_only = HardwareProbe {
                vaapi: false,
                nvenc: false,
                vulkan: true,
                vaapi_device: None,
            };
            assert_eq!(
                vulkan_only.chain(EncoderPreference::Auto).unwrap(),
                vec![HardwareEncoder::Vulkan, HardwareEncoder::Cpu]
            );
        }

        let none = HardwareProbe::default();
        assert_eq!(
            none.chain(EncoderPreference::Auto).unwrap(),
            vec![HardwareEncoder::Cpu]
        );
        assert!(matches!(
            none.chain(EncoderPreference::Explicit(HardwareEncoder::Nvenc)),
            Err(ExportError::EncoderUnavailable("nvenc"))
        ));
        #[cfg(feature = "encode_vulkan")]
        assert!(matches!(
            none.chain(EncoderPreference::Explicit(HardwareEncoder::Vulkan)),
            Err(ExportError::EncoderUnavailable("vulkan"))
        ));
    }

    #[test]
    fn render_node_picks_the_lowest_numbered_node() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("card0"), b"").unwrap();
        std::fs::write(dir.path().join("renderD129"), b"").unwrap();
        std::fs::write(dir.path().join("renderD128"), b"").unwrap();

        let node = first_render_node(dir.path()).unwrap();
        assert_eq!(node.file_name().unwrap(), "renderD128");
        assert_eq!(first_render_node(Path::new("/nonexistent-dri")), None);
    }
}
