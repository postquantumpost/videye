use crate::elden::{
    find_boss_detection, find_grace_detection, find_lost_grace_discovered_detection,
    find_newitem_detection, find_place2_detection, find_place3_detection, find_place_detection,
    find_region2_detection, place_banner_text_image, PlaceBannerDetection,
};
use crate::ghost::{
    find_combat1_crop, find_diamond_symbols, find_location1_symbols, find_location_symbols,
    find_story_crop, horizontal_span_is_centered, text_crop_between_symbols,
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
pub(crate) struct Place {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Place2 {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Place3 {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Region {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LostGraceDiscovered {
    timestamp_ns: u128,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Boss {
    timestamp_ns: u128,
    name: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NewItem {
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
    Place(Place),
    Place2(Place2),
    Place3(Place3),
    LostGraceDiscovered(LostGraceDiscovered),
    Region(Region),
    Boss(Boss),
    NewItem(NewItem),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EldenDetector {
    Grace,
    Place2,
    Place3,
    Place,
    LostGraceDiscovered,
    Boss,
    NewItem,
}

pub(crate) struct DetectorPriority {
    order: Mutex<[Detector; 5]>,
    elden_order: Mutex<[EldenDetector; 7]>,
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
            elden_order: Mutex::new([
                EldenDetector::Grace,
                EldenDetector::Place2,
                EldenDetector::Place3,
                EldenDetector::Place,
                EldenDetector::LostGraceDiscovered,
                EldenDetector::Boss,
                EldenDetector::NewItem,
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

    fn elden_order(&self) -> [EldenDetector; 7] {
        *self
            .elden_order
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn promote_elden(&self, detector: EldenDetector) {
        let mut order = self
            .elden_order
            .lock()
            .unwrap_or_else(|error| error.into_inner());
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
    match state.game {
        Game::EldenRing => {
            let mut matched = false;
            for detector in detector_priority.elden_order() {
                let detector_matched = match detector {
                    EldenDetector::Grace => process_grace_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::Place2 => process_place2_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::Place3 => process_place3_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::Place => process_place_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::LostGraceDiscovered => process_lost_grace_discovered_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::Boss => process_boss_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                    EldenDetector::NewItem => process_newitem_detector(
                        state,
                        frame,
                        output_file,
                        &mut annotated_frame,
                        ocr_session,
                        history,
                        elapsed_ns,
                    )?,
                };
                if detector_matched {
                    detector_priority.promote_elden(detector);
                    matched = true;
                    break;
                }
            }
            if !matched {
                process_region2_detector(
                    state,
                    frame,
                    output_file,
                    &mut annotated_frame,
                    ocr_session,
                    history,
                    elapsed_ns,
                )?;
            }
        }
        Game::GhostofTsushima => {
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
) -> io::Result<bool> {
    let Some(detection) = find_grace_detection(frame, state.frame_width, state.frame_height) else {
        return Ok(false);
    };

    if let Some(annotated_frame) = annotated_frame {
        draw_bounding_box(annotated_frame, state.frame_width, &detection.symbol);
    }
    if state.verbose {
        writeln!(
            output_file,
            "Grace symbol: x={} y={} width={} height={}",
            detection.symbol.left,
            detection.symbol.top,
            detection.symbol.width,
            detection.symbol.height
        )?;
    }
    let Some(name) = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.label,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    add_grace(history, elapsed_ns, name);
    Ok(true)
}

fn process_place_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(detection) = find_place_detection(frame, state.frame_width, state.frame_height) else {
        return Ok(false);
    };

    if let Some(annotated_frame) = annotated_frame {
        for x in detection.line.left..detection.line.left + detection.line.width {
            set_red_pixel(annotated_frame, state.frame_width, x, detection.line.top);
        }
    }
    if state.verbose {
        writeln!(
            output_file,
            "Place underline: x={} y={} width={}",
            detection.line.left, detection.line.top, detection.line.width
        )?;
    }
    let recognized_text = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.text,
        state.current_frame,
    );
    let Some(recognized_text) = recognized_text else {
        return Ok(false);
    };
    let name = normalize_place_name(&recognized_text);
    if name.is_empty() {
        return Ok(false);
    }
    add_place(history, elapsed_ns, name);
    Ok(true)
}

fn process_place2_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let detection = find_place2_detection(frame, state.frame_width, state.frame_height);
    process_place_banner(
        state,
        frame,
        output_file,
        annotated_frame,
        ocr_session,
        history,
        elapsed_ns,
        detection,
        "Place2",
        add_place2,
    )
}

fn process_place3_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let detection = find_place3_detection(frame, state.frame_width, state.frame_height);
    process_place_banner(
        state,
        frame,
        output_file,
        annotated_frame,
        ocr_session,
        history,
        elapsed_ns,
        detection,
        "Place3",
        add_place3,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_place_banner(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
    detection: Option<PlaceBannerDetection>,
    title: &str,
    add_entry: fn(&mut History, u128, String),
) -> io::Result<bool> {
    let Some(detection) = detection else {
        return Ok(false);
    };
    let text_image = place_banner_text_image(frame, state.frame_width, &detection);
    let Some(recognized_text) = ocr_session.recognize(
        &text_image,
        detection.text.width,
        detection.text.height,
        &BoundingBox {
            left: 0,
            top: 0,
            width: detection.text.width,
            height: detection.text.height,
        },
        state.current_frame,
    ) else {
        return Ok(false);
    };
    let name = normalize_place_name(&recognized_text);
    if name.is_empty() {
        return Ok(false);
    }

    if let Some(annotated_frame) = annotated_frame {
        for x in detection.line.left..detection.line.left + detection.line.width {
            set_red_pixel(annotated_frame, state.frame_width, x, detection.line.top);
        }
        draw_bounding_box(annotated_frame, state.frame_width, &detection.text);
    }
    if state.verbose {
        writeln!(
            output_file,
            "{} banner: top={} underline={} text x={} width={}",
            title,
            detection.banner.top,
            detection.line.top,
            detection.text.left,
            detection.text.width
        )?;
    }
    add_entry(history, elapsed_ns, name);
    Ok(true)
}

fn process_region2_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(detection) = find_region2_detection(frame, state.frame_width, state.frame_height)
    else {
        return Ok(false);
    };
    let Some(recognized_text) = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.text,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    let name = normalize_region_name(&recognized_text);
    if name.is_empty() {
        return Ok(false);
    }

    if let Some(annotated_frame) = annotated_frame {
        draw_bounding_box(annotated_frame, state.frame_width, &detection.banner);
    }
    if state.verbose {
        writeln!(
            output_file,
            "Region2 banner: x={} y={} width={} height={}",
            detection.banner.left,
            detection.banner.top,
            detection.banner.width,
            detection.banner.height
        )?;
    }
    add_region(history, elapsed_ns, name);
    Ok(true)
}

fn process_lost_grace_discovered_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(detection) =
        find_lost_grace_discovered_detection(frame, state.frame_width, state.frame_height)
    else {
        return Ok(false);
    };

    if let Some(annotated_frame) = annotated_frame {
        draw_bounding_box(annotated_frame, state.frame_width, &detection.first_l);
        draw_bounding_box(annotated_frame, state.frame_width, &detection.final_d);
    }
    if state.verbose {
        writeln!(
            output_file,
            "Lost grace anchors: L at x={} y={}, D at x={} y={}",
            detection.first_l.left,
            detection.first_l.top,
            detection.final_d.left,
            detection.final_d.top
        )?;
    }
    let Some(recognized_text) = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.text,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    let recognized_text = recognized_text.to_ascii_uppercase();
    if recognized_text.contains("LOST")
        && recognized_text.contains("GRACE")
        && recognized_text.contains("DISCOVERED")
    {
        add_lost_grace_discovered(history, elapsed_ns);
        Ok(true)
    } else {
        Ok(false)
    }
}

fn process_boss_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(detection) = find_boss_detection(frame, state.frame_width, state.frame_height) else {
        return Ok(false);
    };

    if let Some(annotated_frame) = annotated_frame {
        draw_bounding_box(annotated_frame, state.frame_width, &detection.text);
    }
    if state.verbose {
        writeln!(
            output_file,
            "Boss health bar: x={} y={} width={}",
            detection.line.left, detection.line.top, detection.line.width
        )?;
    }
    let recognized_text = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.text,
        state.current_frame,
    );
    let Some(recognized_text) = recognized_text else {
        return Ok(false);
    };
    let name = normalize_boss_name(&recognized_text);
    if name.is_empty() {
        return Ok(false);
    }
    add_boss(history, elapsed_ns, name);
    Ok(true)
}

fn process_newitem_detector(
    state: &ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotated_frame: &mut Option<Vec<u8>>,
    ocr_session: &mut OcrSession<'_>,
    history: &mut History,
    elapsed_ns: u128,
) -> io::Result<bool> {
    let Some(detection) = find_newitem_detection(frame, state.frame_width, state.frame_height) else {
        return Ok(false);
    };
    if state.verbose {
        writeln!(
            output_file,
            "Newitem separators: y={}, {}, {}",
            detection.lines[0].top, detection.lines[1].top, detection.lines[2].top
        )?;
    }
    if ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.label,
        state.current_frame,
    ).as_deref() != Some("NEW") {
        return Ok(false);
    }
    let Some(name) = ocr_session.recognize_item_name(
        frame,
        state.frame_width,
        state.frame_height,
        &detection.name,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    if let Some(annotated_frame) = annotated_frame {
        for line in &detection.lines {
            for column in line.left..line.left + line.width {
                set_red_pixel(annotated_frame, state.frame_width, column, line.top);
            }
        }
        draw_bounding_box(annotated_frame, state.frame_width, &detection.label);
        draw_bounding_box(annotated_frame, state.frame_width, &detection.name);
    }
    add_newitem_with_window(history, elapsed_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
    Ok(true)
}

fn normalize_place_name(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_region_name(text: &str) -> String {
    let name = normalize_place_name(text);
    let words = name.split_whitespace().collect::<Vec<_>>();
    let alphanumeric_count = name
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count();
    // TODO
    if words.is_empty()
        || alphanumeric_count < 6
        || (words.len() == 1 && words[0].chars().count() < 3)
    {
        String::new()
    } else {
        name
    }
}

fn normalize_boss_name(text: &str) -> String {
    let name = normalize_place_name(text);
    let alphanumeric_count = name
        .chars()
        .filter(|character| character.is_alphanumeric())
        .count();
    // TODO
    if alphanumeric_count < 5 {
        String::new()
    } else {
        name
    }
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
        if state.verbose {
            writeln!(
                output_file,
                "Diamond symbol: x={} y={} width={} height={}",
                symbol.left, symbol.top, symbol.width, symbol.height
            )?;
        }
    }
    let mut matched = false;
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
            if let Some(text) = text {
                actnumber = Some(text);
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
            if let Some(title) = title {
                acttitle = Some(title);
            }
        }
        if let (Some(actnumber), Some(acttitle)) = (actnumber, acttitle) {
            add_textact(history, elapsed_ns, actnumber, acttitle);
            matched = true;
        }
    }
    Ok(matched)
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
        if state.verbose {
            writeln!(
                output_file,
                "Location2 symbol: x={} y={} width={} height={}",
                symbol.left, symbol.top, symbol.width, symbol.height
            )?;
        }
    }
    let mut matched = false;
    for crop in crops {
        let Some(location2) = ocr_session.recognize(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
        ) else {
            continue;
        };
        add_location2(history, elapsed_ns, location2);
        matched = true;
    }
    Ok(matched)
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
        if state.verbose {
            writeln!(
                output_file,
                "Location1 symbol: x={} y={} width={} height={}",
                symbol.bounds.left, symbol.bounds.top, symbol.bounds.width, symbol.bounds.height
            )?;
        }
    }
    let mut matched = false;
    for crop in crops {
        let Some(location1) = ocr_session.recognize(
            frame,
            state.frame_width,
            state.frame_height,
            &crop,
            state.current_frame,
        ) else {
            continue;
        };
        add_location1(history, elapsed_ns, location1);
        matched = true;
    }
    Ok(matched)
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
    if state.verbose {
        writeln!(
            output_file,
            "Story panel: x={} y={} width={} height={}",
            crop.left, crop.top, crop.width, crop.height
        )?;
    }
    let Some(story) = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &crop,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    add_story(history, elapsed_ns, story);
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
    if state.verbose {
        writeln!(
            output_file,
            "Combat1 lines: x={} y={} width={} height={}",
            crop.left, crop.top, crop.width, crop.height
        )?;
    }
    let Some(text) = ocr_session.recognize(
        frame,
        state.frame_width,
        state.frame_height,
        &crop,
        state.current_frame,
    ) else {
        return Ok(false);
    };
    add_combat1(history, elapsed_ns, text);
    Ok(true)
}

fn frame_timestamp_ns(state: &ProcessingState) -> u128 {
    u128::from(state.current_frame - 1) * u128::from(state.frame_rate_den) * 1_000_000_000
        / u128::from(state.frame_rate_num)
}

const DEFAULT_DEDUPE_WINDOW_NS: u128 = 120_000_000_000;

fn add_textact(history: &mut History, timestamp_ns: u128, actnumber: String, acttitle: String) {
    add_textact_with_window(
        history,
        timestamp_ns,
        actnumber,
        acttitle,
        DEFAULT_DEDUPE_WINDOW_NS,
    );
}

fn add_textact_with_window(
    history: &mut History,
    timestamp_ns: u128,
    actnumber: String,
    acttitle: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::TextAct(textact) => {
            textact.actnumber == actnumber
                && textact.acttitle == acttitle
                && timestamp_ns.saturating_sub(textact.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::Location2(_) => false,
        HistoryEntry::Location1(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
        HistoryEntry::Grace(_) => false,
        HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => false,
        HistoryEntry::LostGraceDiscovered(_) => false,
        HistoryEntry::Region(_) => false,
        HistoryEntry::Boss(_) | HistoryEntry::NewItem(_) => false,
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
    add_location2_with_window(history, timestamp_ns, location2, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_location2_with_window(
    history: &mut History,
    timestamp_ns: u128,
    location2: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::TextAct(_) => false,
        HistoryEntry::Location1(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
        HistoryEntry::Grace(_) => false,
        HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => false,
        HistoryEntry::LostGraceDiscovered(_) => false,
        HistoryEntry::Region(_) => false,
        HistoryEntry::Boss(_) | HistoryEntry::NewItem(_) => false,
        HistoryEntry::Location2(previous) => {
            previous.location2 == location2
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
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
    add_location1_with_window(history, timestamp_ns, location1, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_location1_with_window(
    history: &mut History,
    timestamp_ns: u128,
    location1: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Location1(previous) => {
            previous.location1 == location1
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::TextAct(_) | HistoryEntry::Location2(_) => false,
        HistoryEntry::Story(_) => false,
        HistoryEntry::Combat1(_) => false,
        HistoryEntry::Grace(_) => false,
        HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => false,
        HistoryEntry::LostGraceDiscovered(_) => false,
        HistoryEntry::Region(_) => false,
        HistoryEntry::Boss(_) | HistoryEntry::NewItem(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Location1(Location1 {
            timestamp_ns,
            location1,
        }));
    }
}

fn add_story(history: &mut History, timestamp_ns: u128, text: String) {
    add_story_with_window(history, timestamp_ns, text, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_story_with_window(
    history: &mut History,
    timestamp_ns: u128,
    text: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Story(previous) => {
            previous.text == text
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Combat1(_)
        | HistoryEntry::Grace(_)
        | HistoryEntry::Place(_)
        | HistoryEntry::Place2(_)
        | HistoryEntry::Place3(_)
        | HistoryEntry::LostGraceDiscovered(_)
        | HistoryEntry::Region(_)
        | HistoryEntry::Boss(_)
        | HistoryEntry::NewItem(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Story(Story { timestamp_ns, text }));
    }
}

fn add_combat1(history: &mut History, timestamp_ns: u128, text: String) {
    add_combat1_with_window(history, timestamp_ns, text, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_combat1_with_window(
    history: &mut History,
    timestamp_ns: u128,
    text: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Combat1(previous) => {
            previous.text == text
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Story(_)
        | HistoryEntry::Grace(_)
        | HistoryEntry::Place(_)
        | HistoryEntry::Place2(_)
        | HistoryEntry::Place3(_)
        | HistoryEntry::LostGraceDiscovered(_)
        | HistoryEntry::Region(_)
        | HistoryEntry::Boss(_)
        | HistoryEntry::NewItem(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Combat1(Combat1 { timestamp_ns, text }));
    }
}

fn add_grace(history: &mut History, timestamp_ns: u128, name: String) {
    add_grace_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_grace_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Grace(previous) => {
            previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Story(_)
        | HistoryEntry::Combat1(_)
        | HistoryEntry::Place(_)
        | HistoryEntry::Place2(_)
        | HistoryEntry::Place3(_)
        | HistoryEntry::LostGraceDiscovered(_)
        | HistoryEntry::Region(_)
        | HistoryEntry::Boss(_)
        | HistoryEntry::NewItem(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Grace(Grace { timestamp_ns, name }));
    }
}

fn add_place(history: &mut History, timestamp_ns: u128, name: String) {
    add_place_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_place_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {
    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::Place(previous)
            if previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::Place(Place { timestamp_ns, name }));
    }
}

fn add_place2(history: &mut History, timestamp_ns: u128, name: String) {
    add_place2_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_place2_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {
    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::Place2(previous)
            if previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::Place2(Place2 { timestamp_ns, name }));
    }
}

fn add_place3(history: &mut History, timestamp_ns: u128, name: String) {
    add_place3_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_place3_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {
    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::Place3(previous)
            if previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::Place3(Place3 { timestamp_ns, name }));
    }
}

fn add_region(history: &mut History, timestamp_ns: u128, name: String) {
    add_region_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_region_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| match entry {
        HistoryEntry::Region(previous) => {
            previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns
        }
        HistoryEntry::TextAct(_)
        | HistoryEntry::Location1(_)
        | HistoryEntry::Location2(_)
        | HistoryEntry::Story(_)
        | HistoryEntry::Combat1(_)
        | HistoryEntry::Grace(_)
        | HistoryEntry::Place(_)
        | HistoryEntry::Place2(_)
        | HistoryEntry::Place3(_)
        | HistoryEntry::Boss(_)
        | HistoryEntry::NewItem(_)
        | HistoryEntry::LostGraceDiscovered(_) => false,
    });
    if !is_duplicate {
        history.push(HistoryEntry::Region(Region { timestamp_ns, name }));
    }
}

fn add_lost_grace_discovered(history: &mut History, timestamp_ns: u128) {
    add_lost_grace_discovered_with_window(history, timestamp_ns, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_lost_grace_discovered_with_window(
    history: &mut History,
    timestamp_ns: u128,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::LostGraceDiscovered(previous)
            if timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::LostGraceDiscovered(LostGraceDiscovered {
            timestamp_ns,
        }));
    }
}

fn add_boss(history: &mut History, timestamp_ns: u128, name: String) {
    add_boss_with_window(history, timestamp_ns, name, DEFAULT_DEDUPE_WINDOW_NS);
}

fn add_boss_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {

    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::Boss(previous)
            if previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::Boss(Boss { timestamp_ns, name }));
    }
}

fn add_newitem_with_window(
    history: &mut History,
    timestamp_ns: u128,
    name: String,
    dedupe_window_ns: u128,
) {
    let is_duplicate = history.iter().any(|entry| {
        matches!(entry, HistoryEntry::NewItem(previous)
            if previous.name == name
                && timestamp_ns.saturating_sub(previous.timestamp_ns) <= dedupe_window_ns)
    });
    if !is_duplicate {
        history.push(HistoryEntry::NewItem(NewItem { timestamp_ns, name }));
    }
}

pub(crate) fn merge_history(
    history: &mut History,
    frame_history: History,
    dedupe_window_ns: u128,
) {
    for entry in frame_history {
        match entry {
            HistoryEntry::TextAct(textact) => add_textact_with_window(
                history,
                textact.timestamp_ns,
                textact.actnumber,
                textact.acttitle,
                dedupe_window_ns,
            ),
            HistoryEntry::Location1(location1) => {
                add_location1_with_window(
                    history,
                    location1.timestamp_ns,
                    location1.location1,
                    dedupe_window_ns,
                )
            }
            HistoryEntry::Location2(location2) => {
                add_location2_with_window(
                    history,
                    location2.timestamp_ns,
                    location2.location2,
                    dedupe_window_ns,
                )
            }
            HistoryEntry::Story(story) => {
                add_story_with_window(history, story.timestamp_ns, story.text, dedupe_window_ns)
            }
            HistoryEntry::Combat1(combat1) => {
                add_combat1_with_window(
                    history,
                    combat1.timestamp_ns,
                    combat1.text,
                    dedupe_window_ns,
                )
            }
            HistoryEntry::Grace(grace) => {
                add_grace_with_window(history, grace.timestamp_ns, grace.name, dedupe_window_ns)
            }
            HistoryEntry::Place(place) => {
                add_place_with_window(history, place.timestamp_ns, place.name, dedupe_window_ns)
            }
            HistoryEntry::Place2(place) => {
                add_place2_with_window(history, place.timestamp_ns, place.name, dedupe_window_ns)
            }
            HistoryEntry::Place3(place) => {
                add_place3_with_window(history, place.timestamp_ns, place.name, dedupe_window_ns)
            }
            HistoryEntry::LostGraceDiscovered(event) => {
                add_lost_grace_discovered_with_window(history, event.timestamp_ns, dedupe_window_ns)
            }
            HistoryEntry::Region(region) => {
                add_region_with_window(history, region.timestamp_ns, region.name, dedupe_window_ns)
            }
            HistoryEntry::Boss(boss) => {
                add_boss_with_window(history, boss.timestamp_ns, boss.name, dedupe_window_ns)
            }
            HistoryEntry::NewItem(item) => {
                add_newitem_with_window(history, item.timestamp_ns, item.name, dedupe_window_ns)
            }
        }
    }
}

fn write_place_verbose(
    output_file: &mut File,
    title: &str,
    timestamp_ns: u128,
    name: &str,
) -> io::Result<()> {
    let elapsed_seconds = timestamp_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let nanoseconds = timestamp_ns % 1_000_000_000;
    writeln!(
        output_file,
        "{title} at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
    )?;
    writeln!(output_file, "{title}: {name}")
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
            HistoryEntry::Place(place) => {
                write_place_verbose(output_file, "Place", place.timestamp_ns, &place.name)?;
            }
            HistoryEntry::Place2(place) => {
                write_place_verbose(output_file, "Place2", place.timestamp_ns, &place.name)?;
            }
            HistoryEntry::Place3(place) => {
                write_place_verbose(output_file, "Place3", place.timestamp_ns, &place.name)?;
            }
            HistoryEntry::LostGraceDiscovered(event) => {
                let elapsed_seconds = event.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = event.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "LostGraceDiscovered at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
            }
            HistoryEntry::Region(region) => {
                let elapsed_seconds = region.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = region.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Region at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Region: {}", region.name)?;
            }
            HistoryEntry::Boss(boss) => {
                let elapsed_seconds = boss.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = boss.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Boss at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Boss: {}", boss.name)?;
            }
            HistoryEntry::NewItem(item) => {
                let elapsed_seconds = item.timestamp_ns / 1_000_000_000;
                let hours = elapsed_seconds / 3_600;
                let minutes = (elapsed_seconds / 60) % 60;
                let seconds = elapsed_seconds % 60;
                let nanoseconds = item.timestamp_ns % 1_000_000_000;
                writeln!(
                    output_file,
                    "Newitem at {hours:02}:{minutes:02}:{seconds:02}.{nanoseconds:09}"
                )?;
                writeln!(output_file, "Newitem: {}", item.name)?;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn write_history_single_line(
    history: &History,
    output_file: &mut impl Write,
    frame_rate_num: u32,
    frame_rate_den: u32,
) -> io::Result<()> {
    for entry in history {
        match entry {
            HistoryEntry::TextAct(textact) => {
                let timestamp =
                    format_history_timestamp(textact.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(
                    output_file,
                    "{timestamp} act: {}, {}",
                    textact.actnumber, textact.acttitle
                )?;
            }
            HistoryEntry::Location2(location2) => {
                let timestamp = format_history_timestamp(
                    location2.timestamp_ns,
                    frame_rate_num,
                    frame_rate_den,
                );
                writeln!(
                    output_file,
                    "{timestamp} location2: {}",
                    location2.location2
                )?;
            }
            HistoryEntry::Location1(location1) => {
                let timestamp = format_history_timestamp(
                    location1.timestamp_ns,
                    frame_rate_num,
                    frame_rate_den,
                );
                writeln!(
                    output_file,
                    "{timestamp} location1: {}",
                    location1.location1
                )?;
            }
            HistoryEntry::Story(story) => {
                let timestamp =
                    format_history_timestamp(story.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} story: {}", story.text)?;
            }
            HistoryEntry::Combat1(combat1) => {
                let timestamp = format_history_timestamp(
                    combat1.timestamp_ns,
                    frame_rate_num,
                    frame_rate_den,
                );
                writeln!(output_file, "{timestamp} combat1: {}", combat1.text)?;
            }
            HistoryEntry::Grace(grace) => {
                let timestamp =
                    format_history_timestamp(grace.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} grace: {}", grace.name)?;
            }
            HistoryEntry::Place(place) => {
                let timestamp =
                    format_history_timestamp(place.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} place: {}", place.name)?;
            }
            HistoryEntry::Place2(place) => {
                let timestamp =
                    format_history_timestamp(place.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} place2: {}", place.name)?;
            }
            HistoryEntry::Place3(place) => {
                let timestamp =
                    format_history_timestamp(place.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} place3: {}", place.name)?;
            }
            HistoryEntry::LostGraceDiscovered(event) => {
                let timestamp =
                    format_history_timestamp(event.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} lost_grace_discovered")?;
            }
            HistoryEntry::Region(region) => {
                let timestamp =
                    format_history_timestamp(region.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} region: {}", region.name)?;
            }
            HistoryEntry::Boss(boss) => {
                let timestamp =
                    format_history_timestamp(boss.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} boss: {}", boss.name)?;
            }
            HistoryEntry::NewItem(item) => {
                let timestamp =
                    format_history_timestamp(item.timestamp_ns, frame_rate_num, frame_rate_den);
                writeln!(output_file, "{timestamp} newitem: {}", item.name)?;
            }
        }
    }
    Ok(())
}

fn format_history_timestamp(
    timestamp_ns: u128,
    frame_rate_num: u32,
    frame_rate_den: u32,
) -> String {
    let elapsed_seconds = timestamp_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let frame_denominator = 1_000_000_000 * u128::from(frame_rate_den);
    let frame_remainder = timestamp_ns % 1_000_000_000;
    let frame =
        (frame_remainder * u128::from(frame_rate_num) + frame_denominator / 2) / frame_denominator;
    let time = match (hours, minutes) {
        (0, 0) => format!("0:{seconds:02}"),
        (0, minutes) => format!("{minutes}:{seconds:02}"),
        (hours, minutes) => format!("{hours}:{minutes:02}:{seconds:02}"),
    };
    format!("{time} (:{frame})")
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
        add_boss, add_combat1, add_grace, add_location1, add_location2, add_lost_grace_discovered,
        add_place, add_place2, add_place3, add_region, add_story, add_textact, merge_history, BoundingBox, Detector,
        DetectorPriority, EldenDetector, HistoryEntry, Place,
    };

    fn write_history_single_line(history: &super::History, output: &mut Vec<u8>) {
        super::write_history_single_line(history, output, 30, 1).unwrap();
    }

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
    fn moves_detected_elden_detector_to_front_independently() {
        let priority = DetectorPriority::default();

        priority.promote_elden(EldenDetector::Boss);
        assert_eq!(
            priority.elden_order(),
            [
                EldenDetector::Boss,
                EldenDetector::Grace,
                EldenDetector::Place2,
                EldenDetector::Place3,
                EldenDetector::Place,
                EldenDetector::LostGraceDiscovered,
                EldenDetector::NewItem,
            ]
        );
        assert_eq!(
            priority.order(),
            [
                Detector::Act,
                Detector::Location2,
                Detector::Location1,
                Detector::Story,
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

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"1:01:01 (:0) act: A12, Opening\n");
    }

    #[test]
    fn writes_subsecond_frame_number_on_single_line() {
        let mut history = Vec::new();
        add_textact(
            &mut history,
            2_500_000_000,
            "A12".to_string(),
            "Opening".to_string(),
        );
        let mut output = Vec::new();

        super::write_history_single_line(&history, &mut output, 30, 1).unwrap();

        assert_eq!(output, b"0:02 (:15) act: A12, Opening\n");
    }

    #[test]
    fn omits_zero_leading_timestamp_fields() {
        assert_eq!(
            super::format_history_timestamp(61_000_000_000, 30, 1),
            "1:01 (:0)"
        );
        assert_eq!(
            super::format_history_timestamp(2_500_000_000, 30, 1),
            "0:02 (:15)"
        );
        assert_eq!(
            super::format_history_timestamp(3_661_100_000_000, 30, 1),
            "1:01:01 (:3)"
        );
        assert_eq!(
            super::format_history_timestamp(966_666_666, 30, 1),
            "0:00 (:29)"
        );
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

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) location2: Loyal Friend's Grave\n");
    }

    #[test]
    fn writes_location1_entries_on_single_lines() {
        let mut history = Vec::new();
        add_location1(&mut history, 2_000_000_000, "Kin Prefecture".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) location1: Kin Prefecture\n");
    }

    #[test]
    fn writes_story_entries_on_single_lines() {
        let mut history = Vec::new();
        add_story(&mut history, 2_000_000_000, "From the darkness".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) story: From the darkness\n");
    }

    #[test]
    fn writes_combat1_entries_on_single_lines() {
        let mut history = Vec::new();
        add_combat1(&mut history, 2_000_000_000, "RYUZO".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) combat1: RYUZO\n");
    }

    #[test]
    fn writes_grace_entries_on_single_lines() {
        let mut history = Vec::new();
        add_grace(&mut history, 2_000_000_000, "Road to the Manor".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) grace: Road to the Manor\n");
    }

    #[test]
    fn writes_place_entries_on_single_lines() {
        let mut history = Vec::new();
        add_place(&mut history, 2_000_000_000, "Caria Manor".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(output, b"0:02 (:0) place: Caria Manor\n");
    }

    #[test]
    fn writes_and_deduplicates_region_entries() {
        let mut history = Vec::new();
        add_region(&mut history, 2_000_000_000, "Three Sisters".to_string());
        add_region(&mut history, 120_000_000_000, "Three Sisters".to_string());
        add_region(&mut history, 122_000_000_001, "Three Sisters".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(
            output,
            b"0:02 (:0) region: Three Sisters\n2:02 (:0) region: Three Sisters\n"
        );
    }

    #[test]
    fn writes_and_deduplicates_lost_grace_discovered_events() {
        let mut history = Vec::new();
        add_lost_grace_discovered(&mut history, 2_000_000_000);
        add_lost_grace_discovered(&mut history, 120_000_000_000);
        add_lost_grace_discovered(&mut history, 122_000_000_001);
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(
            output,
            b"0:02 (:0) lost_grace_discovered\n2:02 (:0) lost_grace_discovered\n"
        );
    }

    #[test]
    fn writes_and_deduplicates_boss_names() {
        let mut history = Vec::new();
        add_boss(
            &mut history,
            2_000_000_000,
            "Royal Knight Loretta".to_string(),
        );
        add_boss(
            &mut history,
            120_000_000_000,
            "Royal Knight Loretta".to_string(),
        );
        add_boss(
            &mut history,
            122_000_000_001,
            "Royal Knight Loretta".to_string(),
        );
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(
            output,
            b"0:02 (:0) boss: Royal Knight Loretta\n2:02 (:0) boss: Royal Knight Loretta\n"
        );
    }

    #[test]
    fn deduplicates_matching_place_entries_within_120_seconds() {
        let mut history = Vec::new();
        add_place(&mut history, 1_000_000_000, "Caria Manor".to_string());
        add_place(&mut history, 120_000_000_000, "Caria Manor".to_string());
        add_place(&mut history, 121_000_000_001, "Caria Manor".to_string());

        let places: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::Place(place) => place.name.as_str(),
                _ => panic!("unexpected history entry"),
            })
            .collect();
        assert_eq!(places, vec!["Caria Manor", "Caria Manor"]);
    }

    #[test]
    fn writes_place2_and_place3_labels_on_single_lines() {
        let mut history = Vec::new();
        add_place2(&mut history, 2_000_000_000, "Limgrave".to_string());
        add_place3(&mut history, 3_000_000_000, "Church of Elleh".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(
            output,
            b"0:02 (:0) place2: Limgrave\n0:03 (:0) place3: Church of Elleh\n"
        );
    }

    #[test]
    fn deduplicates_place_kinds_only_within_the_same_kind() {
        let mut history = Vec::new();
        add_place(&mut history, 1_000_000_000, "Limgrave".to_string());
        add_place2(&mut history, 2_000_000_000, "Limgrave".to_string());
        add_place3(&mut history, 3_000_000_000, "Limgrave".to_string());
        add_place2(&mut history, 4_000_000_000, "Limgrave".to_string());
        add_place3(&mut history, 5_000_000_000, "Limgrave".to_string());
        add_place(&mut history, 6_000_000_000, "Limgrave".to_string());
        let mut output = Vec::new();

        write_history_single_line(&history, &mut output);

        assert_eq!(
            output,
            b"0:01 (:0) place: Limgrave\n0:02 (:0) place2: Limgrave\n0:03 (:0) place3: Limgrave\n"
        );
    }

    #[test]
    fn normalizes_place_name_ocr_edge_noise() {
        assert_eq!(
            super::normalize_place_name("A2 Ranni's Rise,   West"),
            "A2 Ranni's Rise, West"
        );
        assert_eq!(super::normalize_place_name("Ranni's Rise"), "Ranni's Rise");
        assert_eq!(super::normalize_place_name("A"), "A");
    }

    #[test]
    fn rejects_fragmentary_region_ocr_but_keeps_region_names() {
        assert_eq!(super::normalize_region_name("ee"), "");
        assert_eq!(super::normalize_region_name("Foe"), "");
        assert_eq!(
            super::normalize_region_name("ad Erhfee Sisters"),
            "ad Erhfee Sisters"
        );
        assert_eq!(
            super::normalize_region_name("Three Sisters"),
            "Three Sisters"
        );
        assert_eq!(super::normalize_region_name("Caelid"), "Caelid");
        assert_eq!(
            super::normalize_region_name("Liurnia of Lakes"),
            "Liurnia of Lakes"
        );
    }

    #[test]
    fn rejects_short_lowercase_boss_ocr_fragments() {
        assert_eq!(super::normalize_boss_name("inet"), "");
        assert_eq!(
            super::normalize_boss_name("Royal Knight Loretta"),
            "Royal Knight Loretta"
        );
        assert_eq!(
            super::normalize_boss_name("royal knight loretta"),
            "royal knight loretta"
        );
        assert_eq!(
            super::normalize_boss_name("Mohg, Lord of Blood"),
            "Mohg, Lord of Blood"
        );
    }

    #[test]
    fn deduplicates_matching_grace_entries_within_120_seconds() {
        let mut history = Vec::new();
        add_grace(&mut history, 1_000_000_000, "Road to the Manor".to_string());
        add_grace(
            &mut history,
            120_000_000_000,
            "Road to the Manor".to_string(),
        );
        add_grace(
            &mut history,
            121_000_000_001,
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
    fn deduplicates_matching_textacts_within_120_seconds() {
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
            125_000_000_000,
            "A12".to_string(),
            "Opening".to_string(),
        );
        add_textact(
            &mut history,
            125_000_000_001,
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
                HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => {
                    panic!("unexpected Place entry")
                }
                HistoryEntry::LostGraceDiscovered(_) => {
                    panic!("unexpected LostGraceDiscovered entry")
                }
                HistoryEntry::Region(_) => panic!("unexpected Region entry"),
                HistoryEntry::Boss(_) => panic!("unexpected Boss entry"),
                HistoryEntry::NewItem(_) => panic!("unexpected NewItem entry"),
            })
            .collect();
        assert_eq!(
            timestamps,
            vec![5_000_000_000, 10_000_000_000, 125_000_000_001]
        );
    }

    #[test]
    fn deduplicates_matching_story_entries_within_120_seconds() {
        let mut history = Vec::new();
        add_story(&mut history, 1_000_000_000, "From the darkness".to_string());
        add_story(
            &mut history,
            120_000_000_000,
            "From the darkness".to_string(),
        );
        add_story(
            &mut history,
            121_000_000_001,
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
                HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => {
                    panic!("unexpected Place entry")
                }
                HistoryEntry::LostGraceDiscovered(_) => {
                    panic!("unexpected LostGraceDiscovered entry")
                }
                HistoryEntry::Region(_) => panic!("unexpected Region entry"),
                HistoryEntry::Boss(_) => panic!("unexpected Boss entry"),
                HistoryEntry::NewItem(_) => panic!("unexpected NewItem entry"),
            })
            .collect();
        assert_eq!(stories, vec!["From the darkness", "From the darkness"]);
    }

    #[test]
    fn deduplicates_matching_location2_entries_within_120_seconds() {
        let mut history = Vec::new();
        add_location2(
            &mut history,
            1_000_000_000,
            "Loyal Friend's Grave".to_string(),
        );
        add_location2(
            &mut history,
            120_000_000_000,
            "Loyal Friend's Grave".to_string(),
        );
        add_location2(
            &mut history,
            121_000_000_001,
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
                HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => {
                    panic!("unexpected Place entry")
                }
                HistoryEntry::LostGraceDiscovered(_) => {
                    panic!("unexpected LostGraceDiscovered entry")
                }
                HistoryEntry::Region(_) => panic!("unexpected Region entry"),
                HistoryEntry::Boss(_) => panic!("unexpected Boss entry"),
                HistoryEntry::NewItem(_) => panic!("unexpected NewItem entry"),
            })
            .collect();
        assert_eq!(
            locations,
            vec!["Loyal Friend's Grave", "Loyal Friend's Grave"]
        );
    }

    #[test]
    fn deduplicates_matching_location1_entries_within_120_seconds() {
        let mut history = Vec::new();
        add_location1(&mut history, 1_000_000_000, "Kin Prefecture".to_string());
        add_location1(&mut history, 120_000_000_000, "Kin Prefecture".to_string());
        add_location1(&mut history, 121_000_000_001, "Kin Prefecture".to_string());

        let locations: Vec<&str> = history
            .iter()
            .map(|entry| match entry {
                HistoryEntry::TextAct(_) => panic!("unexpected TextAct entry"),
                HistoryEntry::Location1(location) => location.location1.as_str(),
                HistoryEntry::Location2(_) => panic!("unexpected Location2 entry"),
                HistoryEntry::Story(_) => panic!("unexpected Story entry"),
                HistoryEntry::Combat1(_) => panic!("unexpected Combat1 entry"),
                HistoryEntry::Grace(_) => panic!("unexpected Grace entry"),
                HistoryEntry::Place(_) | HistoryEntry::Place2(_) | HistoryEntry::Place3(_) => {
                    panic!("unexpected Place entry")
                }
                HistoryEntry::LostGraceDiscovered(_) => {
                    panic!("unexpected LostGraceDiscovered entry")
                }
                HistoryEntry::Region(_) => panic!("unexpected Region entry"),
                HistoryEntry::Boss(_) => panic!("unexpected Boss entry"),
                HistoryEntry::NewItem(_) => panic!("unexpected NewItem entry"),
            })
            .collect();
        assert_eq!(locations, vec!["Kin Prefecture", "Kin Prefecture"]);
    }

    #[test]
    fn merge_history_uses_the_configured_dedupe_window() {
        let mut history = vec![HistoryEntry::Place(Place {
            timestamp_ns: 1_000_000_000,
            name: "Caria Manor".to_string(),
        })];
        let frame_history = vec![HistoryEntry::Place(Place {
            timestamp_ns: 12_000_000_000,
            name: "Caria Manor".to_string(),
        })];

        merge_history(&mut history, frame_history, 10_000_000_000);

        assert_eq!(history.len(), 2);
    }

    #[test]
    fn newitem_history_preserves_utf8_and_uses_the_dedupe_window() {
        let name = "\u{00c9}p\u{00e9}e, d'acier [5]-2";
        let entry = |timestamp_ns| HistoryEntry::NewItem(super::NewItem {
            timestamp_ns,
            name: name.to_string(),
        });
        let mut history = vec![entry(0)];
        merge_history(&mut history, vec![entry(5_000_000_000)], 10_000_000_000);
        merge_history(&mut history, vec![entry(11_000_000_000)], 10_000_000_000);
        assert_eq!(history.len(), 2);
        let mut output = Vec::new();
        write_history_single_line(&history, &mut output);
        assert_eq!(
            String::from_utf8(output).unwrap(),
            format!("0:00 (:0) newitem: {name}\n0:11 (:0) newitem: {name}\n")
        );
    }

    #[test]
    #[ignore = "requires local samples/lizard.png, FFmpeg, and Tesseract"]
    fn newitem_from_lizard_png() {
        let output = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/lizard.png"))
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}", String::from_utf8_lossy(&output.stderr)
        );
        let (width, height) = (3_840, 2_160);
        let frame = output.stdout;
        assert_eq!(frame.len(), width * height * 4);
        let detection = super::find_newitem_detection(&frame, width, height)
            .expect("three separator lines");
        for use_library in [true, false] {
            let cache = crate::ocr_support::OcrCache::default();
            let mut session = crate::ocr_support::OcrSession::new(use_library, &cache);
            assert_eq!(
                session.recognize(&frame, width, height, &detection.label, 1).as_deref(),
                Some("NEW")
            );
            assert_eq!(
                session.recognize_item_name(&frame, width, height, &detection.name, 1).as_deref(),
                Some("Lizard Greatsword")
            );
            let mut state = super::ProcessingState {
                game: super::Game::EldenRing,
                current_frame: 0,
                frame_width: width,
                frame_height: height,
                frame_size: frame.len(),
                frame_rate_num: 30,
                frame_rate_den: 1,
                check_story_line_thickness: false,
                verbose: false,
            };
            let mut history = Vec::new();
            super::process_frame(
                &mut state,
                &frame,
                &mut Vec::new(),
                true,
                &mut session,
                &mut crate::ghost::DetectorScratch::default(),
                &super::DetectorPriority::default(),
                &mut history,
            ).unwrap();
            assert_eq!(
                history,
                vec![HistoryEntry::NewItem(super::NewItem {
                    timestamp_ns: 0,
                    name: "Lizard Greatsword".to_string(),
                })]
            );

            let mut without_new = frame.clone();
            for row in detection.label.top..detection.label.top + detection.label.height {
                for column in detection.label.left..detection.label.left + detection.label.width {
                    let pixel = (row * width + column) * 4;
                    without_new[pixel..pixel + 4].copy_from_slice(&[20, 20, 20, 255]);
                }
            }
            assert!(super::find_newitem_detection(&without_new, width, height).is_some());
            let mut history = Vec::new();
            assert!(!super::process_newitem_detector(
                &state, &without_new, &mut Vec::new(), &mut None,
                &mut session, &mut history, 0,
            ).unwrap());
            assert!(history.is_empty());
        }
    }

    fn decode_sample_png(name: &str) -> (Vec<u8>, usize, usize) {
        let output = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples").join(name))
            .args([
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let (width, height) = (3_840, 2_160);
        assert_eq!(output.stdout.len(), width * height * 4);
        (output.stdout, width, height)
    }

    fn assert_single_sample_entry(
        frame: &[u8],
        width: usize,
        height: usize,
        process: impl Fn(
            &super::ProcessingState,
            &[u8],
            &mut crate::ocr_support::OcrSession<'_>,
            &mut super::History,
        ) -> std::io::Result<bool>,
        expected: &HistoryEntry,
    ) {
        for use_library in [true, false] {
            let cache = crate::ocr_support::OcrCache::default();
            let mut session = crate::ocr_support::OcrSession::new(use_library, &cache);
            let state = super::ProcessingState {
                game: super::Game::EldenRing,
                current_frame: 0,
                frame_width: width,
                frame_height: height,
                frame_size: frame.len(),
                frame_rate_num: 30,
                frame_rate_den: 1,
                check_story_line_thickness: false,
                verbose: false,
            };
            let mut history = Vec::new();
            assert!(process(&state, frame, &mut session, &mut history).unwrap());
            assert_eq!(history.len(), 1);
            assert_eq!(&history[0], expected);
        }
    }

    #[test]
    #[ignore = "requires local samples/limgrave.png, FFmpeg, and Tesseract"]
    fn place2_from_limgrave_png() {
        let (frame, width, height) = decode_sample_png("limgrave.png");
        let detection = super::find_place2_detection(&frame, width, height).expect("place2 banner");
        assert!(detection.line.top.abs_diff(1_156) <= 2);
        assert_single_sample_entry(
            &frame,
            width,
            height,
            |state, frame, session, history| {
                super::process_place2_detector(
                    state,
                    frame,
                    &mut Vec::new(),
                    &mut None,
                    session,
                    history,
                    0,
                )
            },
            &HistoryEntry::Place2(super::Place2 {
                timestamp_ns: 0,
                name: "Limgrave".to_string(),
            }),
        );
    }

    #[test]
    #[ignore = "requires local samples/elleh_g.png, FFmpeg, and Tesseract"]
    fn grace_from_elleh_png() {
        let (frame, width, height) = decode_sample_png("elleh_g.png");
        let detection = super::find_grace_detection(&frame, width, height).expect("grace box");
        assert!(detection.symbol.left.abs_diff(101) <= 3);
        assert!(detection.symbol.top.abs_diff(253) <= 3);
        assert_single_sample_entry(
            &frame,
            width,
            height,
            |state, frame, session, history| {
                super::process_grace_detector(
                    state,
                    frame,
                    &mut Vec::new(),
                    &mut None,
                    session,
                    history,
                    0,
                )
            },
            &HistoryEntry::Grace(super::Grace {
                timestamp_ns: 0,
                name: "Church of Elleh".to_string(),
            }),
        );
    }

    #[test]
    #[ignore = "requires local samples/elleh_p.png, FFmpeg, and Tesseract"]
    fn place3_from_elleh_png() {
        let (frame, width, height) = decode_sample_png("elleh_p.png");
        assert!(super::find_place2_detection(&frame, width, height).is_none());
        let detection = super::find_place3_detection(&frame, width, height).expect("place3 banner");
        assert!(detection.line.top.abs_diff(776) <= 3);
        assert_single_sample_entry(
            &frame,
            width,
            height,
            |state, frame, session, history| {
                super::process_place3_detector(
                    state,
                    frame,
                    &mut Vec::new(),
                    &mut None,
                    session,
                    history,
                    0,
                )
            },
            &HistoryEntry::Place3(super::Place3 {
                timestamp_ns: 0,
                name: "Church of Elleh".to_string(),
            }),
        );
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

