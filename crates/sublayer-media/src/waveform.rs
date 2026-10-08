//! Timeline waveform decimation and `.waveform` cache files.
//!
//! The cache stores fixed-rate min/max/RMS buckets (50 per second by default)
//! as a compact little-endian binary file, so the timeline can render long
//! clips without re-reading the WAV.
//!
//! Layout (`<i16>`: a `u16`, `[...]`: an array):
//!
//! ```text
//! offset  size  field
//!      0     4  magic `SLWF`
//!      4     2  layout version
//!      6     4  WAV sample rate in Hz
//!     10     4  buckets per second
//!     14     8  mono frame count
//!     22     8  bucket count
//!     30  12*n  buckets: f32 min, f32 max, f32 rms
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::MediaError;

/// Buckets per second used by the timeline unless a caller asks otherwise.
pub const DEFAULT_BUCKETS_PER_SEC: u32 = 50;

/// Magic bytes at the start of every `.waveform` cache file.
const CACHE_MAGIC: [u8; 4] = *b"SLWF";

/// Cache layout version written by this build.
const CACHE_VERSION: u16 = 1;

/// Size of the cache header preceding the bucket array.
const CACHE_HEADER_LEN: u64 = 30;

/// Size of one serialized bucket.
const CACHE_BUCKET_LEN: u64 = 12;

/// Full-scale divisor for a signed integer sample of the given bit depth.
fn full_scale(bits: u16) -> f32 {
    (1_u64 << (bits - 1)) as f32
}

/// Peak and RMS levels of one time bucket, normalized to `-1.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveformBucket {
    /// Most negative sample in the bucket.
    pub min_amplitude: f32,
    /// Most positive sample in the bucket.
    pub max_amplitude: f32,
    /// Root-mean-square level across the bucket.
    pub rms: f32,
}

impl WaveformBucket {
    /// Bucket representing digital silence.
    pub const fn silent() -> Self {
        Self {
            min_amplitude: 0.0,
            max_amplitude: 0.0,
            rms: 0.0,
        }
    }

    /// Largest absolute amplitude in the bucket.
    pub fn peak_amplitude(&self) -> f32 {
        self.max_amplitude.max(-self.min_amplitude)
    }
}

/// A decimated waveform plus the WAV properties needed to map time to buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformCache {
    sample_rate: u32,
    samples_per_sec: u32,
    frame_count: u64,
    buckets: Vec<WaveformBucket>,
}

impl WaveformCache {
    /// Assembles a cache from already-decimated buckets.
    ///
    /// # Panics
    ///
    /// Panics when `sample_rate` or `samples_per_sec` is zero, since the
    /// time↔bucket mapping would be undefined.
    pub fn new(
        sample_rate: u32,
        samples_per_sec: u32,
        frame_count: u64,
        buckets: Vec<WaveformBucket>,
    ) -> Self {
        assert!(
            sample_rate > 0 && samples_per_sec > 0,
            "waveform cache needs non-zero rates"
        );
        Self {
            sample_rate,
            samples_per_sec,
            frame_count,
            buckets,
        }
    }

    /// Sample rate of the decoded WAV.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Buckets per second of timeline time.
    pub fn samples_per_sec(&self) -> u32 {
        self.samples_per_sec
    }

    /// Number of mono frames the buckets were computed from.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// All buckets, ordered by start time.
    pub fn buckets(&self) -> &[WaveformBucket] {
        &self.buckets
    }

    /// Whether the cache holds no buckets.
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Audio duration covered by the buckets, in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        (u128::from(self.frame_count) * 1_000 / u128::from(self.sample_rate)) as u64
    }

    /// Buckets covering `[start_ms, end_ms)`, clamped to the cached range.
    pub fn slice_ms(&self, start_ms: u64, end_ms: u64) -> &[WaveformBucket] {
        let buckets_per_sec = u64::from(self.samples_per_sec);
        let len = self.buckets.len() as u64;
        let first = (start_ms.saturating_mul(buckets_per_sec) / 1_000).min(len);
        let last = (end_ms.saturating_mul(buckets_per_sec) / 1_000).min(len);
        if last <= first {
            return &[];
        }
        &self.buckets[first as usize..last as usize]
    }

    /// Serializes the cache to `path`, creating parent directories as needed.
    pub fn save(&self, path: &Path) -> Result<(), MediaError> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }

        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&CACHE_MAGIC)?;
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;
        writer.write_all(&self.sample_rate.to_le_bytes())?;
        writer.write_all(&self.samples_per_sec.to_le_bytes())?;
        writer.write_all(&self.frame_count.to_le_bytes())?;
        writer.write_all(&(self.buckets.len() as u64).to_le_bytes())?;
        for bucket in &self.buckets {
            writer.write_all(&bucket.min_amplitude.to_le_bytes())?;
            writer.write_all(&bucket.max_amplitude.to_le_bytes())?;
            writer.write_all(&bucket.rms.to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Reads a cache written by [`WaveformCache::save`].
    ///
    /// Malformed headers are rejected before allocating, so corrupt files
    /// cannot trigger huge allocations.
    pub fn load(path: &Path) -> Result<Self, MediaError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        let magic = read_array::<4>(&mut reader)?;
        if magic != CACHE_MAGIC {
            return Err(MediaError::InvalidCache("bad magic bytes".to_owned()));
        }
        let version = u16::from_le_bytes(read_array(&mut reader)?);
        if version > CACHE_VERSION {
            return Err(MediaError::UnsupportedCacheVersion {
                found: version,
                supported: CACHE_VERSION,
            });
        }
        let sample_rate = u32::from_le_bytes(read_array(&mut reader)?);
        let samples_per_sec = u32::from_le_bytes(read_array(&mut reader)?);
        let frame_count = u64::from_le_bytes(read_array(&mut reader)?);
        let bucket_count = u64::from_le_bytes(read_array(&mut reader)?);

        if sample_rate == 0 || samples_per_sec == 0 {
            return Err(MediaError::InvalidCache(
                "header declares a zero sample rate".to_owned(),
            ));
        }
        let declared_len = bucket_count
            .saturating_mul(CACHE_BUCKET_LEN)
            .saturating_add(CACHE_HEADER_LEN);
        if declared_len > file_len {
            return Err(MediaError::InvalidCache(format!(
                "header declares {bucket_count} buckets but the file is only {file_len} bytes"
            )));
        }

        let mut buckets = Vec::with_capacity(bucket_count as usize);
        for _ in 0..bucket_count {
            let bucket = WaveformBucket {
                min_amplitude: f32::from_le_bytes(read_array(&mut reader)?),
                max_amplitude: f32::from_le_bytes(read_array(&mut reader)?),
                rms: f32::from_le_bytes(read_array(&mut reader)?),
            };
            if !bucket.min_amplitude.is_finite()
                || !bucket.max_amplitude.is_finite()
                || !bucket.rms.is_finite()
            {
                return Err(MediaError::InvalidCache(
                    "bucket contains a non-finite amplitude".to_owned(),
                ));
            }
            buckets.push(bucket);
        }

        Ok(Self {
            sample_rate,
            samples_per_sec,
            frame_count,
            buckets,
        })
    }
}

/// Reads `reader` fully into an `N`-byte array, reporting truncation as
/// [`MediaError::InvalidCache`].
fn read_array<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], MediaError> {
    let mut buffer = [0_u8; N];
    reader
        .read_exact(&mut buffer)
        .map_err(|error| MediaError::InvalidCache(format!("truncated cache file: {error}")))?;
    Ok(buffer)
}

/// Reads `wav_path` and decimates it into a [`WaveformCache`].
///
/// Blocking call; async callers should prefer [`generate_waveform_cache`].
pub fn waveform_from_wav(
    wav_path: &Path,
    samples_per_sec: u32,
) -> Result<WaveformCache, MediaError> {
    if samples_per_sec == 0 {
        return Err(MediaError::InvalidBucketRate);
    }

    let mut reader = WavReader::open(wav_path)?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Err(MediaError::InvalidWavSpec(
            "WAV declares a zero sample rate".to_owned(),
        ));
    }

    let frames = mono_frames(&mut reader)?;
    let (buckets, frame_count) = decimate_mono(frames, spec.sample_rate, samples_per_sec)?;
    Ok(WaveformCache::new(
        spec.sample_rate,
        samples_per_sec,
        frame_count,
        buckets,
    ))
}

/// Decimates `wav_path` and writes the cache to `cache_path`.
///
/// Returns the buckets so the caller can render immediately without a cache
/// round-trip. The decode/decimate pass runs on the blocking thread pool.
pub async fn generate_waveform_cache(
    wav_path: &Path,
    cache_path: &Path,
    samples_per_sec: u32,
) -> Result<Vec<WaveformBucket>, MediaError> {
    let wav_path = wav_path.to_owned();
    let cache_path = cache_path.to_owned();
    let task = tokio::task::spawn_blocking(move || -> Result<WaveformCache, MediaError> {
        let cache = waveform_from_wav(&wav_path, samples_per_sec)?;
        cache.save(&cache_path)?;
        Ok(cache)
    });

    let cache = task
        .await
        .map_err(|error| MediaError::TaskJoin(error.to_string()))??;
    Ok(cache.buckets().to_vec())
}

/// Iterator over mono `f32` frames of a WAV file.
type SampleIter<'a> = Box<dyn Iterator<Item = Result<f32, MediaError>> + 'a>;

/// Wraps a WAV reader into an iterator of mono frames normalized to `-1.0..=1.0`.
///
/// Integer samples are scaled by their full-scale value, multichannel frames are
/// averaged, and a trailing partial frame is dropped.
fn mono_frames(
    reader: &mut WavReader<BufReader<File>>,
) -> Result<FrameDownmix<SampleIter<'_>>, MediaError> {
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(MediaError::InvalidWavSpec(
            "WAV declares zero channels".to_owned(),
        ));
    }

    let samples: SampleIter<'_> = match (spec.sample_format, spec.bits_per_sample) {
        // 24-bit data is read through `i32`; hound preserves the raw 24-bit
        // magnitude, so the divisor follows the file's bit depth.
        (SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let divisor = full_scale(bits);
            Box::new(reader.samples::<i32>().map(move |sample| {
                sample
                    .map(|value| value as f32 / divisor)
                    .map_err(MediaError::from)
            }))
        }
        (SampleFormat::Float, 32) => Box::new(
            reader
                .samples::<f32>()
                .map(|sample| sample.map_err(MediaError::from)),
        ),
        (sample_format, bits_per_sample) => {
            return Err(MediaError::UnsupportedSampleFormat {
                sample_format,
                bits_per_sample,
            });
        }
    };

    Ok(FrameDownmix { samples, channels })
}

/// Averages interleaved frames down to mono.
struct FrameDownmix<I> {
    samples: I,
    channels: usize,
}

impl<I> Iterator for FrameDownmix<I>
where
    I: Iterator<Item = Result<f32, MediaError>>,
{
    type Item = Result<f32, MediaError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut sum = 0.0_f32;
        for _ in 0..self.channels {
            match self.samples.next()? {
                Ok(sample) => sum += sample,
                Err(error) => return Some(Err(error)),
            }
        }
        Some(Ok(sum / self.channels as f32))
    }
}

/// Running min/max/RMS accumulator for a single bucket.
#[derive(Debug, Clone, Copy)]
struct BucketAccumulator {
    min: f32,
    max: f32,
    sum_of_squares: f64,
    count: u64,
}

impl BucketAccumulator {
    fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            sum_of_squares: 0.0,
            count: 0,
        }
    }

    fn push(&mut self, sample: f32) {
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
        self.sum_of_squares += f64::from(sample) * f64::from(sample);
        self.count += 1;
    }

    fn finish(self) -> WaveformBucket {
        debug_assert!(self.count > 0, "finished an empty bucket");
        let rms = (self.sum_of_squares / self.count as f64).sqrt() as f32;
        WaveformBucket {
            min_amplitude: self.min,
            max_amplitude: self.max,
            rms,
        }
    }
}

/// Decimates mono frames into `samples_per_sec` buckets.
///
/// Returns the buckets plus the number of frames consumed. A bucket index is
/// derived from the frame index (`frame * buckets_per_sec / sample_rate`), so
/// non-divisible rates stay aligned with the timeline; buckets that received no
/// frames are filled with silence, which only happens when the caller asks for
/// more buckets per second than the source has samples per second.
fn decimate_mono<I>(
    frames: I,
    sample_rate: u32,
    samples_per_sec: u32,
) -> Result<(Vec<WaveformBucket>, u64), MediaError>
where
    I: Iterator<Item = Result<f32, MediaError>>,
{
    debug_assert!(sample_rate > 0 && samples_per_sec > 0);
    let samples_per_sec_u64 = u64::from(samples_per_sec);
    let sample_rate_u64 = u64::from(sample_rate);

    let mut buckets = Vec::new();
    let mut accumulator = BucketAccumulator::new();
    let mut current_bucket = 0_u64;
    let mut frame_count = 0_u64;
    let mut started = false;

    for (index, frame) in frames.enumerate() {
        let frame = frame?;
        frame_count = index as u64 + 1;

        let bucket = index as u64 * samples_per_sec_u64 / sample_rate_u64;
        if started && bucket != current_bucket {
            buckets.push(accumulator.finish());
            for _ in 0..bucket - current_bucket - 1 {
                buckets.push(WaveformBucket::silent());
            }
            accumulator = BucketAccumulator::new();
        }
        current_bucket = bucket;
        started = true;
        accumulator.push(frame);
    }
    if started {
        buckets.push(accumulator.finish());
    }

    Ok((buckets, frame_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};

    /// Writes `interleaved` samples as a 16-bit PCM WAV.
    fn write_wav(path: &Path, sample_rate: u32, channels: u16, interleaved: &[f32]) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in interleaved {
            let scaled = (sample * f32::from(i16::MAX)).round() as i16;
            writer.write_sample(scaled).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn silence(count: usize) -> Vec<f32> {
        vec![0.0; count]
    }

    #[test]
    fn decimation_emits_fixed_rate_buckets() {
        let frames = (0..1_000).map(|_| Ok(0.5_f32));
        let (buckets, frame_count) = decimate_mono(frames, 1_000, 100).unwrap();

        assert_eq!(frame_count, 1_000);
        assert_eq!(buckets.len(), 100);
        for bucket in &buckets {
            assert!((bucket.min_amplitude - 0.5).abs() < 1e-6);
            assert!((bucket.max_amplitude - 0.5).abs() < 1e-6);
            assert!((bucket.rms - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn decimation_keeps_the_partial_tail_bucket() {
        let frames = (0..25).map(|_| Ok(1.0_f32));
        let (buckets, frame_count) = decimate_mono(frames, 1_000, 100).unwrap();

        assert_eq!(frame_count, 25);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[2].max_amplitude, 1.0);
    }

    #[test]
    fn decimation_captures_peaks_and_rms() {
        let samples = [1.0_f32, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0];
        let (buckets, _) = decimate_mono(samples.into_iter().map(Ok), 8, 1).unwrap();

        assert_eq!(buckets.len(), 1);
        let bucket = buckets[0];
        assert_eq!(bucket.min_amplitude, -1.0);
        assert_eq!(bucket.max_amplitude, 1.0);
        assert!((bucket.rms - 0.3125_f32.sqrt()).abs() < 1e-6);
        assert_eq!(bucket.peak_amplitude(), 1.0);
    }

    #[test]
    fn decimation_of_empty_input_yields_nothing() {
        let (buckets, frame_count) =
            decimate_mono(std::iter::empty::<Result<f32, MediaError>>(), 48_000, 50).unwrap();

        assert!(buckets.is_empty());
        assert_eq!(frame_count, 0);
    }

    #[test]
    fn decimation_fills_skipped_buckets_with_silence() {
        // Two frames at a 2 Hz source and 10 buckets/s: index 0 maps to bucket
        // 0, index 1 maps to bucket 5, so buckets 1..=4 must be silent.
        let frames = [Ok(1.0_f32), Ok(1.0)];
        let (buckets, frame_count) = decimate_mono(frames.into_iter(), 2, 10).unwrap();

        assert_eq!(frame_count, 2);
        assert_eq!(buckets.len(), 6);
        assert_eq!(buckets[0].max_amplitude, 1.0);
        assert!(
            buckets[1..5]
                .iter()
                .all(|bucket| *bucket == WaveformBucket::silent())
        );
        assert_eq!(buckets[5].max_amplitude, 1.0);
    }

    #[test]
    fn decimation_reports_hound_errors() {
        let frames = vec![Ok(0.1_f32), Ok(0.2), Ok(0.3)];
        let failing = frames.into_iter().enumerate().map(|(index, frame)| {
            if index == 1 {
                Err(MediaError::Wav(hound::Error::InvalidSampleFormat))
            } else {
                frame
            }
        });
        assert!(matches!(
            decimate_mono(failing, 8, 1),
            Err(MediaError::Wav(_))
        ));
    }

    #[test]
    fn waveform_matches_the_sine_envelope() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sine.wav");
        let sample_rate = 16_000_u32;
        let amplitude = 0.9_f32;
        let samples: Vec<f32> = (0..sample_rate)
            .map(|index| {
                amplitude
                    * (2.0 * std::f32::consts::PI * 440.0 * index as f32 / sample_rate as f32).sin()
            })
            .collect();
        write_wav(&path, sample_rate, 1, &samples);

        let cache = waveform_from_wav(&path, DEFAULT_BUCKETS_PER_SEC).unwrap();

        assert_eq!(cache.sample_rate(), 16_000);
        assert_eq!(cache.samples_per_sec(), DEFAULT_BUCKETS_PER_SEC);
        assert_eq!(cache.frame_count(), u64::from(sample_rate));
        assert_eq!(cache.buckets().len(), 50);
        assert_eq!(cache.duration_ms(), 1_000);

        let peak = cache
            .buckets()
            .iter()
            .map(WaveformBucket::peak_amplitude)
            .fold(0.0_f32, f32::max);
        assert!((peak - amplitude).abs() < 0.01, "peak {peak}");

        let mean_rms = cache.buckets().iter().map(|bucket| bucket.rms).sum::<f32>() / 50.0;
        let expected_rms = amplitude / 2.0_f32.sqrt();
        assert!(
            (mean_rms - expected_rms).abs() < 0.01,
            "rms {mean_rms}, expected {expected_rms}"
        );
    }

    #[test]
    fn waveform_downmixes_multichannel_frames() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("antiphase.wav");
        let mut interleaved = Vec::new();
        for _ in 0..400 {
            interleaved.push(1.0_f32);
            interleaved.push(-1.0_f32);
        }
        write_wav(&path, 400, 2, &interleaved);

        let cache = waveform_from_wav(&path, 1).unwrap();

        assert_eq!(cache.frame_count(), 400);
        assert_eq!(cache.buckets().len(), 1);
        assert!(cache.buckets()[0].peak_amplitude() < 1e-6);
    }

    #[test]
    fn waveform_scales_24_bit_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("24bit.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        let amplitude = 0.5_f32 * 8_388_607.0;
        for _ in 0..800 {
            writer.write_sample(amplitude.round() as i32).unwrap();
        }
        writer.finalize().unwrap();

        let cache = waveform_from_wav(&path, 50).unwrap();

        // 800 frames at 8 kHz is 100 ms, i.e. five 20 ms buckets.
        assert_eq!(cache.buckets().len(), 5);
        let peak = cache.buckets()[0].peak_amplitude();
        assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn waveform_rejects_zero_bucket_rate() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tone.wav");
        write_wav(&path, 8_000, 1, &silence(16));

        assert!(matches!(
            waveform_from_wav(&path, 0),
            Err(MediaError::InvalidBucketRate)
        ));
    }

    #[test]
    fn waveform_rejects_unsupported_bit_depths() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("8bit.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 8,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        writer.write_sample(0_i8).unwrap();
        writer.finalize().unwrap();

        assert!(matches!(
            waveform_from_wav(&path, 50),
            Err(MediaError::UnsupportedSampleFormat {
                bits_per_sample: 8,
                ..
            })
        ));
    }

    #[test]
    fn waveform_accepts_float_wavs() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("float.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        for _ in 0..800 {
            writer.write_sample(0.25_f32).unwrap();
        }
        writer.finalize().unwrap();

        let cache = waveform_from_wav(&path, 50).unwrap();

        assert_eq!(cache.frame_count(), 800);
        assert!((cache.buckets()[0].max_amplitude - 0.25).abs() < 1e-6);
    }

    #[test]
    fn waveform_cache_roundtrips_through_disk() {
        let directory = tempfile::tempdir().unwrap();
        let wav = directory.path().join("tone.wav");
        let samples: Vec<f32> = (0..8_000)
            .map(|index| 0.5 * (index as f32 / 64.0).sin())
            .collect();
        write_wav(&wav, 8_000, 1, &samples);

        let cache = waveform_from_wav(&wav, DEFAULT_BUCKETS_PER_SEC).unwrap();
        let cache_path = directory.path().join("caches").join("tone.waveform");
        cache.save(&cache_path).unwrap();

        let loaded = WaveformCache::load(&cache_path).unwrap();
        assert_eq!(loaded, cache);
    }

    #[test]
    fn waveform_cache_slices_by_time() {
        let buckets = vec![WaveformBucket::silent(); 100];
        // 100 buckets at 50/s cover 2 seconds, backed by 2 seconds of 8 kHz audio.
        let cache = WaveformCache::new(8_000, 50, 16_000, buckets);

        assert_eq!(cache.duration_ms(), 2_000);
        assert_eq!(cache.slice_ms(0, 500).len(), 25);
        assert_eq!(cache.slice_ms(500, 1_000).len(), 25);
        assert_eq!(cache.slice_ms(1_900, 5_000).len(), 5);
        assert!(cache.slice_ms(1_500, 1_500).is_empty());
        assert!(cache.slice_ms(5_000, 6_000).is_empty());
    }

    #[test]
    fn waveform_cache_rejects_corrupt_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.waveform");
        let cache = WaveformCache::new(8_000, 50, 800, vec![WaveformBucket::silent()]);
        cache.save(&path).unwrap();
        let valid = std::fs::read(&path).unwrap();

        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        std::fs::write(&path, &bad_magic).unwrap();
        assert!(matches!(
            WaveformCache::load(&path),
            Err(MediaError::InvalidCache(_))
        ));

        let truncated = &valid[..20];
        std::fs::write(&path, truncated).unwrap();
        assert!(matches!(
            WaveformCache::load(&path),
            Err(MediaError::InvalidCache(_))
        ));

        let mut future_version = valid.clone();
        future_version[4..6].copy_from_slice(&99_u16.to_le_bytes());
        std::fs::write(&path, &future_version).unwrap();
        assert!(matches!(
            WaveformCache::load(&path),
            Err(MediaError::UnsupportedCacheVersion { found: 99, .. })
        ));

        let mut absurd_bucket_count = valid.clone();
        absurd_bucket_count[22..30].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &absurd_bucket_count).unwrap();
        assert!(matches!(
            WaveformCache::load(&path),
            Err(MediaError::InvalidCache(_))
        ));
    }

    #[tokio::test]
    async fn generated_cache_is_written_and_returned() {
        let directory = tempfile::tempdir().unwrap();
        let wav = directory.path().join("tone.wav");
        write_wav(&wav, 16_000, 1, &vec![0.25; 16_000]);
        let cache_path = directory.path().join("tone.waveform");

        let buckets = generate_waveform_cache(&wav, &cache_path, DEFAULT_BUCKETS_PER_SEC)
            .await
            .unwrap();

        assert_eq!(buckets.len(), 50);
        assert_eq!(buckets[0].max_amplitude, 0.25);
        let loaded = WaveformCache::load(&cache_path).unwrap();
        assert_eq!(loaded.buckets(), buckets.as_slice());

        let error = generate_waveform_cache(&wav, &cache_path, 0)
            .await
            .unwrap_err();
        assert!(matches!(error, MediaError::InvalidBucketRate));
    }
}
