use crate::ghost::{
    find_combat1_crop, find_diamond_symbols, find_location1_symbols, find_location_symbols,
    find_story_crop, horizontal_span_is_centered, text_crop_between_symbols,
    title_crop_below_symbols, Location1Side,
};
use std::collections::HashMap;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tesseract::{PageSegMode, Tesseract};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BoundingBox {
    pub(crate) left: usize,
    pub(crate) top: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

pub(crate) struct ProcessingState {
    pub(crate) current_frame: u64,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) frame_size: usize,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
    pub(crate) check_story_line_thickness: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TextAct {
    timestamp_ns: u128,
    actnumber: String,
    acttitle: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Location2 {
    timestamp_ns: u128,
    location2: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Location1 {
    timestamp_ns: u128,
    location1: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Story {
    timestamp_ns: u128,
    text: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Combat1 {
    timestamp_ns: u128,
    text: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum HistoryEntry {
    TextAct(TextAct),
    Location1(Location1),
    Location2(Location2),
    Story(Story),
    Combat1(Combat1),
}

pub(crate) type History = Vec<HistoryEntry>;

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

pub(crate) fn process_frame(
    state: &mut ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotate_frame: bool,
    use_tesseract_library: bool,
    ocr_cache: &OcrCache,
    history: &mut History,
) -> io::Result<Option<Vec<u8>>> {
    state.current_frame += 1;
    let elapsed_ns = frame_timestamp_ns(state);
    let elapsed_seconds = elapsed_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let nanoseconds = elapsed_ns % 1_000_000_000;
    let mut ocr_backend = if use_tesseract_library {
        OcrBackend::Library(None)
    } else {
        OcrBackend::Process
    };
    writeln!(
        output_file,
        "Frame {} at {:02}:{:02}:{:02}.{:09}: {}x{} ({} RGBA bytes)",
        state.current_frame,
        hours,
        minutes,
        seconds,
        nanoseconds,
        state.frame_width,
        state.frame_height,
        frame.len()
    )?;

    let symbols = find_diamond_symbols(state, frame);
    let mut annotated_frame = annotate_frame.then(|| frame.to_vec());
    for symbol in &symbols {
        if let Some(annotated_frame) = &mut annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, symbol);
        }
        writeln!(
            output_file,
            "Diamond symbol: x={} y={} width={} height={}",
            symbol.left, symbol.top, symbol.width, symbol.height
        )?;
    }
    for pair in symbols.windows(2) {
        if !horizontal_span_is_centered(
            pair[0].left,
            pair[1].left.saturating_add(pair[1].width),
            state.frame_width,
        ) {
            continue;
        }
        let mut actnumber = None;
        let mut acttitle = None;
        if let Some(crop) =
            text_crop_between_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        {
            let text = recognize_text(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
                &mut ocr_backend,
                ocr_cache,
            );
            if !text.trim().is_empty() {
                actnumber = Some(text.trim().to_string());
            }
        }
        if let Some(crop) =
            title_crop_below_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        {
            let title = recognize_text(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
                &mut ocr_backend,
                ocr_cache,
            );
            if !title.trim().is_empty() {
                acttitle = Some(title.trim().to_string());
            }
        }
        if let (Some(actnumber), Some(acttitle)) = (actnumber, acttitle) {
            add_textact(history, elapsed_ns, actnumber, acttitle);
        }
    }

    let location_symbols = find_location_symbols(state, frame);
    for symbol in &location_symbols {
        if let Some(annotated_frame) = &mut annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, symbol);
        }
        writeln!(
            output_file,
            "Location2 symbol: x={} y={} width={} height={}",
            symbol.left, symbol.top, symbol.width, symbol.height
        )?;
    }
    for pair in location_symbols.windows(2) {
        if let Some(crop) =
            text_crop_between_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        {
            let location2 = recognize_text(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
                &mut ocr_backend,
                ocr_cache,
            );
            let location2 = location2.trim();
            if !location2.is_empty() {
                add_location2(history, elapsed_ns, location2.to_string());
            }
        }
    }

    let location1_symbols = find_location1_symbols(state, frame);
    for symbol in &location1_symbols {
        if let Some(annotated_frame) = &mut annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, &symbol.bounds);
        }
        writeln!(
            output_file,
            "Location1 symbol: x={} y={} width={} height={}",
            symbol.bounds.left, symbol.bounds.top, symbol.bounds.width, symbol.bounds.height
        )?;
    }
    for pair in location1_symbols.windows(2) {
        if pair[0].side != Location1Side::Left || pair[1].side != Location1Side::Right {
            continue;
        }
        if let Some(crop) = text_crop_between_symbols(
            &pair[0].bounds,
            &pair[1].bounds,
            state.frame_width,
            state.frame_height,
        ) {
            let location1 = recognize_text(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
                &mut ocr_backend,
                ocr_cache,
            );
            let location1 = location1.trim();
            if !location1.is_empty() {
                add_location1(history, elapsed_ns, location1.to_string());
            }
        }
    }

    if let Some(crop) = find_story_crop(
        frame,
        state.frame_width,
        state.frame_height,
        state.check_story_line_thickness,
    ) {
        writeln!(
            output_file,
            "Story panel: x={} y={} width={} height={}",
            crop.left, crop.top, crop.width, crop.height
        )?;
        let story = recognize_text(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
            &mut ocr_backend,
            ocr_cache,
        );
        let story = story.trim();
        if !story.is_empty() {
            add_story(history, elapsed_ns, story.to_string());
        }
    }

    if let Some(crop) = find_combat1_crop(frame, state.frame_width, state.frame_height) {
        writeln!(
            output_file,
            "Combat1 lines: x={} y={} width={} height={}",
            crop.left, crop.top, crop.width, crop.height
        )?;
        let text = recognize_text(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
            &mut ocr_backend,
            ocr_cache,
        );
        let text = text.trim();
        if !text.is_empty() {
            add_combat1(history, elapsed_ns, text.to_string());
        }
    }
    Ok(annotated_frame)
}

fn frame_timestamp_ns(state: &ProcessingState) -> u128 {
    u128::from(state.current_frame - 1) * u128::from(state.frame_rate_den) * 1_000_000_000
        / u128::from(state.frame_rate_num)
}

fn add_textact(history: &mut History, timestamp_ns: u128, actnumber: String, acttitle: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::TextAct(textact) => {
            textact.actnumber == actnumber
                && textact.acttitle == acttitle
                && timestamp_ns.saturating_sub(textact.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
        HistoryEntry::Location2(_) => false,
        HistoryEntry::Location1(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::TextAct(TextAct {
            timestamp_ns,
            actnumber,
            acttitle,
        }));
    }
}

fn add_location2(history: &mut History, timestamp_ns: u128, location2: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::TextAct(_) => false,
        HistoryEntry::Location1(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
        HistoryEntry::Location2(previous) => {
            previous.location2 == location2
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
    });
    if !is_duplicate {
        history.push(HistoryEntry::Location2(Location2 {
            timestamp_ns,
            location2,
        }));
    }
}

fn add_location1(history: &mut History, timestamp_ns: u128, location1: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Location1(previous) => {
            previous.location1 == location1
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
        HistoryEntry::TextAct(_) | HistoryEntry::Location2(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Location1(Location1 {
            timestamp_ns,
            location1,
        }));
    }
}

fn add_story(history: &mut History, timestamp_ns: u128, text: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Story(previous) => {
            previous.text == text
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Combat1(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Story(Story { timestamp_ns, text }));
    }
}

fn add_combat1(history: &mut History, timestamp_ns: u128, text: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Combat1(previous) => {
            previous.text == text
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Story(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Combat1(Combat1 { timestamp_ns, text }));
    }
}

pub(crate) fn merge_history(history: &mut History, frame_history: History) {
    for entry in frame_history {
        match entry {
            HistoryEntry::TextAct(textact) => add_textact(
                history,
                textact.timestamp_ns,
                textact.actnumber,
                textact.acttitle,
            ),
            HistoryEntry::Location1(location1) => {
                add_location1(history, location1.timestamp_ns, location1.location1)
            }
            HistoryEntry::Location2(location2) => {
                add_location2(history, location2.timestamp_ns, location2.location2)
            }
            HistoryEntry::Story(story) => add_story(history, story.timestamp_ns, story.text),
            HistoryEntry::Combat1(combat1) => {
                add_combat1(history, combat1.timestamp_ns, combat1.text)
            }
        }
    }
}

pub(crate) fn write_history(history: &History, output_file: &mut File) -> io::Result<()> {
    for entry in history {
        match entry {
            HistoryEntry::TextAct(textact) => {
                let elapsed_seconds = textact.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = textact.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "TextAct at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Text between symbols: {}", textact.actnumber)?;
                writeln!(output_file, "Title below symbols: {}", textact.acttitle)?;
            }
            HistoryEntry::Location2(location2) => {
                let elapsed_seconds = location2.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = location2.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Location2 at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Location2: {}", location2.location2)?;
            }
            HistoryEntry::Location1(location1) => {
                let elapsed_seconds = location1.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = location1.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Location1 at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Location1: {}", location1.location1)?;
            }
            HistoryEntry::Story(story) => {
                let elapsed_seconds = story.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = story.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Story at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Story: {}", story.text)?;
            }
            HistoryEntry::Combat1(combat1) => {
                let elapsed_seconds = combat1.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = combat1.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Combat1 at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Combat1: {}", combat1.text)?;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn write_history_single_line(
    history: &History,
    output_file: &mut impl Write,
) -> io::Result<()> {
    for entry in history {
        match entry {
            HistoryEntry::TextAct(textact) => {
                let elapsed_seconds = textact.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(
                    output_file,
                    "{timestamp} act: {}, {}",
                    textact.actnumber, textact.acttitle
                )?;
            }
            HistoryEntry::Location2(location2) => {
                let elapsed_seconds = location2.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(
                    output_file,
                    "{timestamp} location2: {}",
                    location2.location2
                )?;
            }
            HistoryEntry::Location1(location1) => {
                let elapsed_seconds = location1.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(
                    output_file,
                    "{timestamp} location1: {}",
                    location1.location1
                )?;
            }
            HistoryEntry::Story(story) => {
                let elapsed_seconds = story.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(output_file, "{timestamp} story: {}", story.text)?;
            }
            HistoryEntry::Combat1(combat1) => {
                let elapsed_seconds = combat1.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(output_file, "{timestamp} combat1: {}", combat1.text)?;
            }
        }
    }
    Ok(())
}

fn format_history_timestamp(hours: u128, minutes: u128, seconds: u128) -> String {
    match (hours, minutes) {
        (0, 0) => seconds.to_string(),
        (0, minutes) => format!("{minutes}:{seconds:02}"),
        (hours, minutes) => format!("{hours}:{minutes:02}:{seconds:02}"),
    }
}

fn draw_bounding_box(frame: &mut [u8], frame_width: usize, bounds: &BoundingBox) {
    let right = bounds.left + bounds.width;
    let bottom = bounds.top + bounds.height;
    for inset in 0..2 {
        for x in bounds.left + inset..right - inset {
            set_red_pixel(frame, frame_width, x, bounds.top + inset);
            set_red_pixel(frame, frame_width, x, bottom - 1 - inset);
        }
        for y in bounds.top + inset..bottom - inset {
            set_red_pixel(frame, frame_width, bounds.left + inset, y);
            set_red_pixel(frame, frame_width, right - 1 - inset, y);
        }
    }
}

fn set_red_pixel(frame: &mut [u8], frame_width: usize, x: usize, y: usize) {
    let pixel = (y * frame_width + x) * 4;
    frame[pixel..pixel + 4].copy_from_slice(&[255, 0, 0, 255]);
}

fn recognize_text(
    frame: &[u8],
    frame_width: usize,
    frame_height: usize,
    crop: &BoundingBox,
    frame_number: u64,
    backend: &mut OcrBackend,
    ocr_cache: &OcrCache,
) -> String {
    ocr_result_or_empty(
        recognize_text_inner(frame, frame_width, frame_height, crop, backend, ocr_cache),
        frame_number,
    )
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
    stdin.write_all(&rgb_crop)?;
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
        add_combat1, add_location1, add_location2, add_story, add_textact,
        write_history_single_line, BoundingBox, HistoryEntry, OcrBackend, OcrCache,
    };

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
        for index in 2..super::MAX_OCR_CACHE_ENTRIES as u64 {
            cache.insert(5, 5, crop_bytes(index), index.to_string());
        }

        cache.insert(
            5,
            5,
            crop_bytes(super::MAX_OCR_CACHE_ENTRIES as u64),
            "newest".to_string(),
        );

        assert_eq!(cache.get(5, 5, &crop_bytes(0)), Some("zero".to_string()));
        assert_eq!(cache.get(5, 5, &crop_bytes(1)), None);
        assert_eq!(
            cache.get(5, 5, &crop_bytes(super::MAX_OCR_CACHE_ENTRIES as u64)),
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
        let mut backend = OcrBackend::Process;
        let cache = OcrCache::default();

        assert!(
            super::recognize_text_inner(&frame, 5, 5, &crop, &mut backend, &cache)
                .unwrap()
                .is_empty()
        );
        assert!(!super::crop_is_large_enough(&crop));
        assert!(!super::crop_is_large_enough(&BoundingBox {
            width: 19,
            height: 5,
            ..crop
        }));
        assert!(!super::crop_is_large_enough(&BoundingBox {
            width: 20,
            height: 4,
            ..crop
        }));
        assert!(super::crop_is_large_enough(&BoundingBox {
            width: 20,
            height: 5,
            ..crop
        }));
    }

    #[test]
    fn writes_history_entries_on_single_lines() {
        let mut history = Vec::new();
        add_textact(
            &mut history,
            3_661_000_000_000,
            "A12".to_string(),
            "Opening".to_string(),
        );
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"1:01:01 act: A12, Opening\n");
    }

    #[test]
    fn omits_zero_leading_timestamp_fields() {
        assert_eq!(super::format_history_timestamp(0, 1, 2), "1:02");
        assert_eq!(super::format_history_timestamp(0, 0, 2), "2");
    }

    #[test]
    fn writes_location2_entries_on_single_lines() {
        let mut history = Vec::new();
        add_location2(
            &mut history,
            2_000_000_000,
            "Loyal Friend's Grave".to_string(),
        );
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"2 location2: Loyal Friend's Grave\n");
    }

    #[test]
    fn writes_location1_entries_on_single_lines() {
        let mut history = Vec::new();
        add_location1(&mut history, 2_000_000_000, "Kin Prefecture".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"2 location1: Kin Prefecture\n");
    }

    #[test]
    fn writes_story_entries_on_single_lines() {
        let mut history = Vec::new();
        add_story(&mut history, 2_000_000_000, "From the darkness".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"2 story: From the darkness\n");
    }

    #[test]
    fn writes_combat1_entries_on_single_lines() {
        let mut history = Vec::new();
        add_combat1(&mut history, 2_000_000_000, "RYUZO".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"2 combat1: RYUZO\n");
    }

    #[test]
    fn deduplicates_matching_textacts_within_twenty_seconds() {
        let mut history = Vec::new();
        add_textact(
            &mut history,
            5_000_000_000,
            "A12".to_string(),
            "Opening".to_string(),
        );
        add_textact(
            &mut history,
            10_000_000_000,
            "B34".to_string(),
            "Different".to_string(),
        );
        add_textact(
            &mut history,
            25_000_000_000,
            "A12".to_string(),
            "Opening".to_string(),
        );
        add_textact(
            &mut history,
            25_000_000_001,
            "A12".to_string(),
            "Opening".to_string(),
        );

        let timestamps: Vec<u128> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::TextAct(textact) => textact.timestamp_ns,
                HistoryEntry::Location1(_) => panic!("unexpected Location1 entry"),
                HistoryEntry::Location2(_) => panic!("unexpected Location2 entry"),
                HistoryEntry::Story(_) => panic!("unexpected Story entry"),
                HistoryEntry::Combat1(_) => panic!("unexpected Combat1 entry"),
            })
            .collect();
        assert_eq!(
            timestamps,
            vec![5_000_000_000, 10_000_000_000, 25_000_000_001]
        );
    }

    #[test]
    fn deduplicates_matching_story_entries_within_twenty_seconds() {
        let mut history = Vec::new();
        add_story(&mut history, 1_000_000_000, "From the darkness".to_string());
        add_story(
            &mut history,
            20_000_000_000,
            "From the darkness".to_string(),
        );
        add_story(
            &mut history,
            21_000_000_001,
            "From the darkness".to_string(),
        );

        let stories: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::TextAct(_) => panic!("unexpected TextAct entry"),
                HistoryEntry::Location1(_) => panic!("unexpected Location1 entry"),
                HistoryEntry::Location2(_) => panic!("unexpected Location2 entry"),
                HistoryEntry::Story(story) => story.text.as_str(),
                HistoryEntry::Combat1(_) => panic!("unexpected Combat1 entry"),
            })
            .collect();
        assert_eq!(stories, vec!["From the darkness", "From the darkness"]);
    }

    #[test]
    fn deduplicates_matching_location2_entries_within_twenty_seconds() {
        let mut history = Vec::new();
        add_location2(
            &mut history,
            1_000_000_000,
            "Loyal Friend's Grave".to_string(),
        );
        add_location2(
            &mut history,
            20_000_000_000,
            "Loyal Friend's Grave".to_string(),
        );
        add_location2(
            &mut history,
            21_000_000_001,
            "Loyal Friend's Grave".to_string(),
        );

        let locations: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::TextAct(_) => panic!("unexpected TextAct entry"),
                HistoryEntry::Location1(_) => panic!("unexpected Location1 entry"),
                HistoryEntry::Location2(location) => location.location2.as_str(),
                HistoryEntry::Story(_) => panic!("unexpected Story entry"),
                HistoryEntry::Combat1(_) => panic!("unexpected Combat1 entry"),
            })
            .collect();
        assert_eq!(
            locations,
            vec!["Loyal Friend's Grave", "Loyal Friend's Grave"]
        );
    }

    #[test]
    fn deduplicates_matching_location1_entries_within_twenty_seconds() {
        let mut history = Vec::new();
        add_location1(&mut history, 1_000_000_000, "Kin Prefecture".to_string());
        add_location1(&mut history, 20_000_000_000, "Kin Prefecture".to_string());
        add_location1(&mut history, 21_000_000_001, "Kin Prefecture".to_string());

        let locations: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::TextAct(_) => panic!("unexpected TextAct entry"),
                HistoryEntry::Location1(location) => location.location1.as_str(),
                HistoryEntry::Location2(_) => panic!("unexpected Location2 entry"),
                HistoryEntry::Story(_) => panic!("unexpected Story entry"),
                HistoryEntry::Combat1(_) => panic!("unexpected Combat1 entry"),
            })
            .collect();
        assert_eq!(locations, vec!["Kin Prefecture", "Kin Prefecture"]);
    }

    #[test]
    fn draws_red_outline_on_copied_frame() {
        let width = 10;
        let mut frame = vec![255; width * 10 * 4];
        let bounds = BoundingBox {
            left: 2,
            top: 2,
            width: 6,
            height: 6,
        };

        super::draw_bounding_box(&mut frame, width, &bounds);

        let top_edge = (2 * width + 2) * 4;
        let inner_pixel = (4 * width + 4) * 4;
        assert_eq!(&frame[top_edge..top_edge + 4], &[255, 0, 0, 255]);
        assert_eq!(&frame[inner_pixel..inner_pixel + 4], &[255, 255, 255, 255]);
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
            super::crop_rgba_to_rgb(&frame, 3, 2, &crop).unwrap(),
            vec![4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22]
        );
    }

    #[test]
    fn does_not_propagate_ocr_failures() {
        let text = super::ocr_result_or_empty(
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "simulated Tesseract failure",
            )),
            7,
        );

        assert!(text.is_empty());
    }
}
