use crate::ghost::{
    find_combat1_crop, find_diamond_symbols, find_grace_detection, find_location1_symbols,
    find_location_symbols, find_story_crop, horizontal_span_is_centered, text_crop_between_symbols,
    title_crop_below_symbols, DetectorScratch, Location1Side,
};
use crate::ocr_support::OcrSession;
use std::fs::File;
use std::io::{self, Write};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Game {
    #[default]
    GhostofTsushima,
    EldenRing,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BoundingBox {
    pub(crate) left: usize,
    pub(crate) top: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

pub(crate) struct ProcessingState {
    pub(crate) game: Game,
    pub(crate) current_frame: u64,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) frame_size: usize,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
    pub(crate) check_story_line_thickness: bool,
    pub(crate) verbose: bool,
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
pub(crate) struct Grace {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum HistoryEntry {
    TextAct(TextAct),
    Location1(Location1),
    Location2(Location2),
    Story(Story),
    Combat1(Combat1),
    Grace(Grace),
}

pub(crate) type History = Vec<HistoryEntry>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Detector {
    Act,
    Location2,
    Location1,
    Story,
    Combat1,
}

pub(crate) struct DetectorPriority {
    order: Mutex<[Detector; 5]>,
}

impl Default for DetectorPriority {
    fn default() -> Self {
        Self {
            order: Mutex::new([
                Detector::Act,
                Detector::Location2,
                Detector::Location1,
                Detector::Story,
                Detector::Combat1,
            ]),
        }
    }
}

impl DetectorPriority {
    fn order(&self) -> [Detector; 5] {
        *self.order.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn promote(&self, detector: Detector) {
        let mut order = self.order.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(index) = order.iter().position(|candidate| *candidate == detector) {
            order[..=index].rotate_right(1);
        }
    }
}

pub(crate) fn process_frame(
    state: &mut ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotate_frame: bool,
    ocr_session: &mut OcrSession<'_>,
    detector_scratch: &mut DetectorScratch,
    detector_priority: &DetectorPriority,
    history: &mut History,
) -> io::Result<Option<Vec<u8>>> {
    state.current_frame += 1;
    let elapsed_ns = frame_timestamp_ns(state);
    let elapsed_seconds = elapsed_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let nanoseconds = elapsed_ns % 1_000_000_000;
    if state.verbose {
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
    }

    let mut annotated_frame = annotate_frame.then(|| frame.to_vec());
    if state.game == Game::EldenRing {
        process_grace_detector(
            state,
            frame,
            output_file,
            &mut annotated_frame,
            ocr_session,
            history,
            elapsed_ns,
        )?;
        return Ok(annotated_frame);
    }

    for detector in detector_priority.order() {
        let matched = match detector {
            Detector::Act => process_act_detector(
                state,
                frame,
                output_file,
                &mut annotated_frame,
                ocr_session,
                history,
                elapsed_ns,
                detector_scratch,
            )?,
            Detector::Location2 => process_location2_detector(
                state,
                frame,
                output_file,
                &mut annotated_frame,
                ocr_session,
                history,
                elapsed_ns,
                detector_scratch,
            )?,
            Detector::Location1 => process_location1_detector(
                state,
                frame,
                output_file,
                &mut annotated_frame,
                ocr_session,
                history,
                elapsed_ns,
                detector_scratch,
            )?,
            Detector::Story => process_story_detector(
                state,
                frame,
                output_file,
                &mut annotated_frame,
                ocr_session,
                history,
                elapsed_ns,
            )?,
            Detector::Combat1 => process_combat1_detector(
                state,
                frame,
                output_file,
                &mut annotated_frame,
                ocr_session,
                history,
                elapsed_ns,
            )?,
        };
        if matched {
            detector_priority.promote(detector);
            break;
        }
    }
    Ok(annotated_frame)
}

fn process_grace_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<()> {
    let Some(detection) = find_grace_detection(frame, state.frame_width, state.frame_height) else {
        return Ok(());
    };

    if let Some(annotated_frame) = annotated_frame {
        draw_bounding_box(annotated_frame, state.frame_width, &detection.symbol);
    }
    writeln!(
        output_file,
        "Grace symbol: x={} y={} width={} height={}",
        detection.symbol.left,
        detection.symbol.top,
        detection.symbol.width,
        detection.symbol.height
    )?;
    let name = ocr_session
        .recognize(
            frame,
            state.frame_width,
            state.frame_height,
            &detection.label,
            state.current_frame,
        )
        .trim()
        .to_string();
    if !name.is_empty() {
        add_grace(history, elapsed_ns, name);
    }
    Ok(())
}

fn process_act_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
    scratch: &mut DetectorScratch,
) -> io::Result<bool> {
    let symbols = find_diamond_symbols(state, frame, scratch);
    let pairs = symbols
        .windows(2)
        .filter(|pair| {
            horizontal_span_is_centered(
                pair[0].left,
                pair[1].left.saturating_add(pair[1].width),
                state.frame_width,
            )
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return Ok(false);
    }

    for symbol in &symbols {
        if let Some(annotated_frame) = annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, symbol);
        }
        writeln!(
            output_file,
            "Diamond symbol: x={} y={} width={} height={}",
            symbol.left, symbol.top, symbol.width, symbol.height
        )?;
    }
    for pair in pairs {
        let mut actnumber = None;
        let mut acttitle = None;
        if let Some(crop) =
            text_crop_between_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        {
            let text = ocr_session.recognize(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
            );
            if !text.trim().is_empty() {
                actnumber = Some(text.trim().to_string());
            }
        }
        if let Some(crop) =
            title_crop_below_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        {
            let title = ocr_session.recognize(
                frame,
                state.frame_width,
                state.frame_height,
                &crop,
                state.current_frame,
            );
            if !title.trim().is_empty() {
                acttitle = Some(title.trim().to_string());
            }
        }
        if let (Some(actnumber), Some(acttitle)) = (actnumber, acttitle) {
            add_textact(history, elapsed_ns, actnumber, acttitle);
        }
    }
    Ok(true)
}

fn process_location2_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
    scratch: &mut DetectorScratch,
) -> io::Result<bool> {
    let symbols = find_location_symbols(state, frame, scratch);
    let crops = symbols
        .windows(2)
        .filter_map(|pair| {
            text_crop_between_symbols(&pair[0], &pair[1], state.frame_width, state.frame_height)
        })
        .collect::<Vec<_>>();
    if crops.is_empty() {
        return Ok(false);
    }

    for symbol in &symbols {
        if let Some(annotated_frame) = annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, symbol);
        }
        writeln!(
            output_file,
            "Location2 symbol: x={} y={} width={} height={}",
            symbol.left, symbol.top, symbol.width, symbol.height
        )?;
    }
    for crop in crops {
        let location2 = ocr_session.recognize(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
        );
        let location2 = location2.trim();
        if !location2.is_empty() {
            add_location2(history, elapsed_ns, location2.to_string());
        }
    }
    Ok(true)
}

fn process_location1_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
    scratch: &mut DetectorScratch,
) -> io::Result<bool> {
    let symbols = find_location1_symbols(state, frame, scratch);
    let crops = symbols
        .windows(2)
        .filter(|pair| pair[0].side == Location1Side::Left && pair[1].side == Location1Side::Right)
        .filter_map(|pair| {
            text_crop_between_symbols(
                &pair[0].bounds,
                &pair[1].bounds,
                state.frame_width,
                state.frame_height,
            )
        })
        .collect::<Vec<_>>();
    if crops.is_empty() {
        return Ok(false);
    }

    for symbol in &symbols {
        if let Some(annotated_frame) = annotated_frame {
            draw_bounding_box(annotated_frame, state.frame_width, &symbol.bounds);
        }
        writeln!(
            output_file,
            "Location1 symbol: x={} y={} width={} height={}",
            symbol.bounds.left, symbol.bounds.top, symbol.bounds.width, symbol.bounds.height
        )?;
    }
    for crop in crops {
        let location1 = ocr_session.recognize(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
        );
        let location1 = location1.trim();
        if !location1.is_empty() {
            add_location1(history, elapsed_ns, location1.to_string());
        }
    }
    Ok(true)
}

fn process_story_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    _annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(crop) = find_story_crop(
        frame,
        state.frame_width,
        state.frame_height,
        state.check_story_line_thickness,
    ) else {
        return Ok(false);
    };
    writeln!(
        output_file,
        "Story panel: x={} y={} width={} height={}",
        crop.left, crop.top, crop.width, crop.height
    )?;
    let story = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &crop,
        state.current_frame,
    );
    let story = story.trim();
    if !story.is_empty() {
        add_story(history, elapsed_ns, story.to_string());
    }
    Ok(true)
}

fn process_combat1_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    _annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(crop) = find_combat1_crop(frame, state.frame_width, state.frame_height) else {
        return Ok(false);
    };
    writeln!(
        output_file,
        "Combat1 lines: x={} y={} width={} height={}",
        crop.left, crop.top, crop.width, crop.height
    )?;
    let text = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &crop,
        state.current_frame,
    );
    let text = text.trim();
    if !text.is_empty() {
        add_combat1(history, elapsed_ns, text.to_string());
    }
    Ok(true)
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
        HistoryEntry::Grace(_) => false,
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
        HistoryEntry::Grace(_) => false,
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
        HistoryEntry::Grace(_) => false,
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
        | HistoryEntry::Combat1(_)
        | HistoryEntry::Grace(_) => false,
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
        | HistoryEntry::Story(_)
        | HistoryEntry::Grace(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Combat1(Combat1 { timestamp_ns, text }));
    }
}

fn add_grace(history: &mut History, timestamp_ns: u128, name: String) {
    const DUPLICATE_WINDOW_NS: u128 = 20_000_000_000;

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Grace(previous) => {
            previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= DUPLICATE_WINDOW_NS
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Story(_)
        | HistoryEntry::Combat1(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Grace(Grace { timestamp_ns, name }));
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
            HistoryEntry::Grace(grace) => add_grace(history, grace.timestamp_ns, grace.name),
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
            HistoryEntry::Grace(grace) => {
                let elapsed_seconds = grace.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = grace.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Grace at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Grace: {}", grace.name)?;
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
            HistoryEntry::Grace(grace) => {
                let elapsed_seconds = grace.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let timestamp = format_history_timestamp(hours, minutes, seconds);
                writeln!(output_file, "{timestamp} grace: {}", grace.name)?;
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

#[cfg(test)]
mod tests {
    use super::{
        add_combat1, add_grace, add_location1, add_location2, add_story, add_textact,
        write_history_single_line, BoundingBox, Detector, DetectorPriority, HistoryEntry,
    };

    #[test]
    fn moves_detected_detector_to_front_and_preserves_remaining_order() {
        let priority = DetectorPriority::default();

        priority.promote(Detector::Story);
        assert_eq!(
            priority.order(),
            [
                Detector::Story,
                Detector::Act,
                Detector::Location2,
                Detector::Location1,
                Detector::Combat1,
            ]
        );

        priority.promote(Detector::Location2);
        assert_eq!(
            priority.order(),
            [
                Detector::Location2,
                Detector::Story,
                Detector::Act,
                Detector::Location1,
                Detector::Combat1,
            ]
        );
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
    fn writes_grace_entries_on_single_lines() {
        let mut history = Vec::new();
        add_grace(&mut history, 2_000_000_000, "Road to the Manor".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output).unwrap();

        assert_eq!(output, b"2 grace: Road to the Manor\n");
    }

    #[test]
    fn deduplicates_matching_grace_entries_within_twenty_seconds() {
        let mut history = Vec::new();
        add_grace(&mut history, 1_000_000_000, "Road to the Manor".to_string());
        add_grace(
            &mut history,
            20_000_000_000,
            "Road to the Manor".to_string(),
        );
        add_grace(
            &mut history,
            21_000_000_001,
            "Road to the Manor".to_string(),
        );

        let graces: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::Grace(grace) => grace.name.as_str(),
                _ => panic!("unexpected history entry"),
            })
            .collect();
        assert_eq!(graces, vec!["Road to the Manor", "Road to the Manor"]);
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
                HistoryEntry::Grace(_) => panic!("unexpected Grace entry"),
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
                HistoryEntry::Grace(_) => panic!("unexpected Grace entry"),
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
                HistoryEntry::Grace(_) => panic!("unexpected Grace entry"),
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
                HistoryEntry::Grace(_) => panic!("unexpected Grace entry"),
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
}
