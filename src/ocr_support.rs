use crate::frame_processor::BoundingBox;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tesseract::{PageSegMode, Tesseract};

const MAX_OCR_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_OCR_CACHE_ENTRIES: usize = 2_048;

#[derive(Default)]
struct OcrCacheState {
    entries: HashMap<u64, OcrCacheEntry>,
    bytes: usize,
    access_clock: u64,
}

struct OcrCacheEntry {
    width: usize,
    height: usize,
    rgb_crop: Vec<u8>,
    text: String,
    last_access: u64,
}

#[derive(Default)]
pub(crate) struct OcrCache {
    state: Mutex<OcrCacheState>,
}

impl OcrCache {
    fn get(&self, width: usize, height: usize, rgb_crop: &[u8]) -> Option<String> {
        if rgb_crop.len() > MAX_OCR_CACHE_BYTES {
            return None;
        }
        let key = ocr_cache_key(width, height, rgb_crop);
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.access_clock = state.access_clock.wrapping_add(1);
        let access_clock = state.access_clock;
        let entry = state.entries.get_mut(&key)?;
        if entry.width != width || entry.height != height || entry.rgb_crop != rgb_crop {
            return None;
        }
        entry.last_access = access_clock;
        Some(entry.text.clone())
    }

    fn insert(&self, width: usize, height: usize, rgb_crop: Vec<u8>, text: String) {
        let entry_bytes = rgb_crop.len().saturating_add(text.len());
        if entry_bytes > MAX_OCR_CACHE_BYTES {
            return;
        }
        let key = ocr_cache_key(width, height, &rgb_crop);
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());

        if let Some(entry) = state.entries.get(&key) {
            if entry.width == width && entry.height == height && entry.rgb_crop == rgb_crop {
                return;
            }
        }
        if let Some(replaced) = state.entries.remove(&key) {
            state.bytes -= replaced.rgb_crop.len() + replaced.text.len();
        }

        while state.entries.len() >= MAX_OCR_CACHE_ENTRIES
            || state.bytes.saturating_add(entry_bytes) > MAX_OCR_CACHE_BYTES
        {
            let oldest_key = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(key, _)| *key);
            let Some(oldest_key) = oldest_key else {
                return;
            };
            if let Some(oldest) = state.entries.remove(&oldest_key) {
                state.bytes -= oldest.rgb_crop.len() + oldest.text.len();
            }
        }

        state.access_clock = state.access_clock.wrapping_add(1);
        let last_access = state.access_clock;
        state.bytes += entry_bytes;
        state.entries.insert(
            key,
            OcrCacheEntry {
                width,
                height,
                rgb_crop,
                text,
                last_access,
            },
        );
    }
}

fn ocr_cache_key(width: usize, height: usize, rgb_crop: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    width.hash(&mut hasher);
    height.hash(&mut hasher);
    rgb_crop.hash(&mut hasher);
    hasher.finish()
}

enum OcrBackend {
    Process,
    Library(Option<Tesseract>),
}

pub(crate) struct OcrSession<'cache> {
    backend: OcrBackend,
    cache: &'cache OcrCache,
}

impl<'cache> OcrSession<'cache> {
    pub(crate) fn new(use_tesseract_library: bool, cache: &'cache OcrCache) -> Self {
        let backend = if use_tesseract_library {
            OcrBackend::Library(None)
        } else {
            OcrBackend::Process
        };
        Self { backend, cache }
    }

    pub(crate) fn recognize(
        &mut self,
        frame: &[u8],
        frame_width: usize,
        frame_height: usize,
        crop: &BoundingBox,
        frame_number: u64,
    ) -> String {
        ocr_result_or_empty(
            recognize_text_inner(
                frame,
                frame_width,
                frame_height,
                crop,
                &mut self.backend,
                self.cache,
            ),
            frame_number,
        )
    }
}

fn ocr_result_or_empty(result: io::Result<String>, frame_number: u64) -> String {
    result.unwrap_or_else(|error| {
        eprintln!("OCR failed on frame {frame_number}: {error}");
        String::new()
    })
}

fn recognize_text_inner(
    frame: &[u8],
    frame_width: usize,
    frame_height: usize,
    crop: &BoundingBox,
    backend: &mut OcrBackend,
    ocr_cache: &OcrCache,
) -> io::Result<String> {
    if !crop_is_large_enough(crop) {
        return Ok(String::new());
    }
    let rgb_crop = crop_rgba_to_rgb(frame, frame_width, frame_height, crop)?;
    if let Some(text) = ocr_cache.get(crop.width, crop.height, &rgb_crop) {
        return Ok(text);
    }
    let text = match backend {
        OcrBackend::Process => recognize_text_with_process(&rgb_crop, crop),
        OcrBackend::Library(engine) => recognize_text_with_library(engine, &rgb_crop, crop),
    }?;
    ocr_cache.insert(crop.width, crop.height, rgb_crop, text.clone());
    Ok(text)
}

fn crop_is_large_enough(crop: &BoundingBox) -> bool {
    crop.width >= 20 && crop.height >= 5
}

fn recognize_text_with_process(rgb_crop: &[u8], crop: &BoundingBox) -> io::Result<String> {
    let mut tesseract = Command::new("tesseract")
        .args(["stdin", "stdout", "--psm", "7"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = tesseract.stdin.take().ok_or_else(|| {
        io::Error::new(io::ErrorKind::BrokenPipe, "failed to open Tesseract stdin")
    })?;
    write!(stdin, "P6\n{} {}\n255\n", crop.width, crop.height)?;
    stdin.write_all(rgb_crop)?;
    drop(stdin);

    let output = tesseract.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Tesseract exited with status {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn recognize_text_with_library(
    engine: &mut Option<Tesseract>,
    rgb_crop: &[u8],
    crop: &BoundingBox,
) -> io::Result<String> {
    let frame_width = i32::try_from(crop.width)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    let frame_height = i32::try_from(crop.height)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
    let bytes_per_line = frame_width
        .checked_mul(3)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "OCR crop row is too large"))?;
    let engine_instance = match engine.take() {
        Some(engine_instance) => engine_instance,
        None => Tesseract::new(None, Some("eng"))
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?,
    };
    let result = (|| {
        let mut engine_instance = engine_instance
            .set_frame(rgb_crop, frame_width, frame_height, 3, bytes_per_line)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?;
        engine_instance.set_page_seg_mode(PageSegMode::PsmSingleLine);
        let mut engine_instance = engine_instance
            .recognize()
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?;
        let text = engine_instance
            .get_text()
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error.to_string()))?;
        Ok((engine_instance, text))
    })();

    match result {
        Ok((engine_instance, text)) => {
            *engine = Some(engine_instance);
            Ok(text)
        }
        Err(error) => Err(error),
    }
}

fn crop_rgba_to_rgb(
    frame: &[u8],
    frame_width: usize,
    frame_height: usize,
    crop: &BoundingBox,
) -> io::Result<Vec<u8>> {
    let frame_len = frame_width
        .checked_mul(frame_height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame dimensions overflow"))?;
    let crop_right = crop
        .left
        .checked_add(crop.width)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "crop dimensions overflow"))?;
    let crop_bottom = crop
        .top
        .checked_add(crop.height)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "crop dimensions overflow"))?;
    if crop.width == 0
        || crop.height == 0
        || frame.len() < frame_len
        || crop_right > frame_width
        || crop_bottom > frame_height
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "OCR crop is outside the RGBA frame",
        ));
    }

    let rgb_len = crop
        .width
        .checked_mul(crop.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "crop dimensions overflow"))?;
    let mut rgb = Vec::with_capacity(rgb_len);
    for y in crop.top..crop_bottom {
        for x in crop.left..crop_right {
            let pixel = (y * frame_width + x) * 4;
            rgb.extend_from_slice(&frame[pixel..pixel + 3]);
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::{
        crop_is_large_enough, crop_rgba_to_rgb, ocr_result_or_empty, recognize_text_inner,
        OcrBackend, OcrCache, OcrSession, MAX_OCR_CACHE_ENTRIES,
    };
    use crate::frame_processor::BoundingBox;

    #[test]
    fn caches_only_identical_rgb_crops_with_matching_dimensions() {
        let cache = OcrCache::default();
        cache.insert(5, 5, vec![1, 2, 3], "cached".to_string());

        assert_eq!(cache.get(5, 5, &[1, 2, 3]), Some("cached".to_string()));
        assert_eq!(cache.get(3, 5, &[1, 2, 3]), None);
        assert_eq!(cache.get(5, 5, &[1, 2, 4]), None);
    }

    #[test]
    fn evicts_least_recently_used_crop_when_cache_is_full() {
        let cache = OcrCache::default();
        let crop_bytes = |index: u64| {
            let mut bytes = vec![0; 75];
            bytes[..8].copy_from_slice(&index.to_le_bytes());
            bytes
        };

        cache.insert(5, 5, crop_bytes(0), "zero".to_string());
        cache.insert(5, 5, crop_bytes(1), "one".to_string());
        assert_eq!(cache.get(5, 5, &crop_bytes(0)), Some("zero".to_string()));
        for index in 2..MAX_OCR_CACHE_ENTRIES as u64 {
            cache.insert(5, 5, crop_bytes(index), index.to_string());
        }

        cache.insert(
            5,
            5,
            crop_bytes(MAX_OCR_CACHE_ENTRIES as u64),
            "newest".to_string(),
        );

        assert_eq!(cache.get(5, 5, &crop_bytes(0)), Some("zero".to_string()));
        assert_eq!(cache.get(5, 5, &crop_bytes(1)), None);
        assert_eq!(
            cache.get(5, 5, &crop_bytes(MAX_OCR_CACHE_ENTRIES as u64)),
            Some("newest".to_string())
        );
    }

    #[test]
    fn skips_crops_narrower_than_twenty_pixels_without_ocr_error() {
        let frame = vec![255; 5 * 5 * 4];
        let crop = BoundingBox {
            left: 0,
            top: 0,
            width: 4,
            height: 5,
        };
        let cache = OcrCache::default();
        let mut session = OcrSession::new(false, &cache);
        let mut backend = OcrBackend::Process;

        assert!(session.recognize(&frame, 5, 5, &crop, 1).is_empty());
        assert!(
            recognize_text_inner(&frame, 5, 5, &crop, &mut backend, &cache)
                .unwrap()
                .is_empty()
        );
        assert!(!crop_is_large_enough(&crop));
        assert!(!crop_is_large_enough(&BoundingBox {
            width: 19,
            height: 5,
            ..crop
        }));
        assert!(!crop_is_large_enough(&BoundingBox {
            width: 20,
            height: 4,
            ..crop
        }));
        assert!(crop_is_large_enough(&BoundingBox {
            width: 20,
            height: 5,
            ..crop
        }));
    }

    #[test]
    fn converts_rgba_crop_to_rgb_for_tesseract() {
        let frame: Vec<u8> = (0..24).collect();
        let crop = BoundingBox {
            left: 1,
            top: 0,
            width: 2,
            height: 2,
        };

        assert_eq!(
            crop_rgba_to_rgb(&frame, 3, 2, &crop).unwrap(),
            vec![4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22]
        );
    }

    #[test]
    fn does_not_propagate_ocr_failures() {
        let text = ocr_result_or_empty(
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "simulated Tesseract failure",
            )),
            7,
        );

        assert!(text.is_empty());
    }
}
