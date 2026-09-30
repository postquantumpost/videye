use std::fs::File;
use std::io::{self, Write};
use std::process::{Command, Stdio};

#[derive(Debug, PartialEq, Eq)]
struct BoundingBox {
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

struct Component {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    area: usize,
    pixels: Option<Vec<(usize, usize)>>,
}

#[derive(Debug, PartialEq, Eq)]
enum Location1Side {
    Left,
    Right,
}

struct Location1Symbol {
    bounds: BoundingBox,
    side: Location1Side,
}

#[derive(Clone, Copy)]
struct StoryLine {
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    support: usize,
}

#[derive(Clone, Copy)]
struct CombatLine {
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    support: usize,
}

const MAX_GROUP_ASPECT_RATIO: f32 = 1.2;
const MAX_COMPONENT_ASPECT_RATIO: f32 = 1.5;

pub(crate) struct ProcessingState {
    pub(crate) current_frame: u64,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) frame_size: usize,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
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

pub(crate) fn process_frame(
    state: &mut ProcessingState,
    frame: &[u8],
    output_file: &mut impl Write,
    annotate_frame: bool,
    history: &mut History,
) -> io::Result<Option<Vec<u8>>> {
    state.current_frame += 1;
    let elapsed_ns = frame_timestamp_ns(state);
    let elapsed_seconds = elapsed_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let nanoseconds = elapsed_ns % 1_000_000_000;
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
            );
            let location1 = location1.trim();
            if !location1.is_empty() {
                add_location1(history, elapsed_ns, location1.to_string());
            }
        }
    }

    if let Some(crop) = find_story_crop(frame, state.frame_width, state.frame_height) {
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

fn text_crop_between_symbols(
    left: &BoundingBox,
    right: &BoundingBox,
    frame_width: usize,
    frame_height: usize,
) -> Option<BoundingBox> {
    let (left_center_y, right_center_y) = aligned_symbol_centers(left, right)?;

    let gap_left = left.left.checked_add(left.width)?;
    if gap_left >= right.left || right.left > frame_width {
        return None;
    }

    let crop_height = left.height.saturating_add(right.height).min(frame_height);
    if crop_height == 0 {
        return None;
    }
    let center_y = (left_center_y + right_center_y) / 2;
    let crop_top = center_y
        .saturating_sub(crop_height / 2)
        .min(frame_height - crop_height);

    Some(BoundingBox {
        left: gap_left,
        top: crop_top,
        width: right.left - gap_left,
        height: crop_height,
    })
}

fn title_crop_below_symbols(
    left: &BoundingBox,
    right: &BoundingBox,
    frame_width: usize,
    frame_height: usize,
) -> Option<BoundingBox> {
    if frame_width == 0 || frame_height == 0 {
        return None;
    }
    let (left_center_y, right_center_y) = aligned_symbol_centers(left, right)?;
    let symbol_height = left.height.checked_add(right.height)? / 2;
    if symbol_height == 0 {
        return None;
    }

    let crop_height = symbol_height.checked_mul(2)?.min(frame_height);
    let center_y = left_center_y.checked_add(right_center_y)? / 2;
    let crop_offset = symbol_height.checked_add(symbol_height / 2)?;
    let crop_top = center_y
        .saturating_add(crop_offset)
        .min(frame_height - crop_height);

    Some(BoundingBox {
        left: 0,
        top: crop_top,
        width: frame_width,
        height: crop_height,
    })
}

fn aligned_symbol_centers(left: &BoundingBox, right: &BoundingBox) -> Option<(usize, usize)> {
    if !similar_symbol_sizes(left, right) {
        return None;
    }

    let left_center_y = left.top.checked_add(left.height / 2)?;
    let right_center_y = right.top.checked_add(right.height / 2)?;
    let vertical_tolerance = (left.height.max(right.height) / 2).max(2);
    (left_center_y.abs_diff(right_center_y) <= vertical_tolerance)
        .then_some((left_center_y, right_center_y))
}

fn similar_symbol_sizes(left: &BoundingBox, right: &BoundingBox) -> bool {
    let width_min = left.width.min(right.width);
    let height_min = left.height.min(right.height);
    width_min > 0
        && height_min > 0
        && left.width.max(right.width) as f32 / width_min as f32 <= 1.25
        && left.height.max(right.height) as f32 / height_min as f32 <= 1.25
}

fn find_diamond_symbols(state: &ProcessingState, frame: &[u8]) -> Vec<BoundingBox> {
    find_diamond_symbols_in_band(state, frame, state.frame_height / 2, is_dark_pixel)
}

fn find_location_symbols(state: &ProcessingState, frame: &[u8]) -> Vec<BoundingBox> {
    let center_y = state.frame_height.saturating_mul(24) / 100;
    find_diamond_symbols_in_band(state, frame, center_y, is_bright_pixel)
}

fn find_location1_symbols(state: &ProcessingState, frame: &[u8]) -> Vec<Location1Symbol> {
    let width = state.frame_width;
    let height = state.frame_height;
    let Some(expected_len) = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return Vec::new();
    };
    if width == 0 || height == 0 || frame.len() < expected_len {
        return Vec::new();
    }

    let center_y = height.saturating_mul(24) / 100;
    let max_group_dimension = ((height as f32 * 0.06).ceil() as usize).max(12);
    let band_radius = max_group_dimension;
    let first_y = center_y.saturating_sub(band_radius);
    let last_y = center_y.saturating_add(band_radius).min(height - 1);
    let band_height = last_y - first_y + 1;
    let Some(band_size) = width.checked_mul(band_height) else {
        return Vec::new();
    };
    let mut visited = vec![false; band_size];
    let max_component_dimension = ((height as f32 * 0.025).ceil() as usize).max(5);
    let mut components = Vec::new();

    for y in first_y..=last_y {
        for x in 0..width {
            let band_index = (y - first_y) * width + x;
            if visited[band_index] || !is_location1_pixel(frame, width, x, y) {
                continue;
            }

            let component = collect_component(
                frame,
                width,
                first_y,
                last_y,
                x,
                y,
                max_component_dimension.saturating_mul(max_component_dimension),
                is_location1_pixel,
                &mut visited,
            );
            let component_width = component.max_x - component.min_x + 1;
            let component_height = component.max_y - component.min_y + 1;
            if component.area >= 3
                && component_width >= 2
                && component_height >= 2
                && component_width <= max_component_dimension
                && component_height <= max_component_dimension
                && is_roughly_diamond_shaped(&component)
            {
                components.push(component);
            }
        }
    }

    components.sort_by_key(|component| component.min_x);
    let center_tolerance = ((height as f32 * 0.012).ceil() as usize).max(2);
    let mut symbols = Vec::new();
    for first in 0..components.len() {
        for second in first + 1..components.len() {
            if components[second].min_x - components[first].min_x > max_group_dimension {
                break;
            }
            for third in second + 1..components.len() {
                if components[third].min_x - components[first].min_x > max_group_dimension {
                    break;
                }
                let group = [&components[first], &components[second], &components[third]];
                if let Some(symbol) =
                    location1_symbol_bounds(group, center_y, center_tolerance, max_group_dimension)
                {
                    if !symbols.iter().any(|existing: &Location1Symbol| {
                        existing.bounds == symbol.bounds && existing.side == symbol.side
                    }) {
                        symbols.push(symbol);
                    }
                }
            }
        }
    }

    symbols.sort_by_key(|symbol| symbol.bounds.left);
    symbols
}

fn find_story_crop(frame: &[u8], width: usize, height: usize) -> Option<BoundingBox> {
    const TOP_LINE_Y_RATIO: f32 = 1604.0 / 2160.0;
    const BOTTOM_LINE_Y_RATIO: f32 = 1716.0 / 2160.0;
    const LINE_Y_TOLERANCE_RATIO: f32 = 0.02;

    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let search_top = height.saturating_mul(65) / 100;
    let search_bottom = height.saturating_mul(90) / 100;
    let search_left = width.saturating_mul(20) / 100;
    let search_right = width.saturating_mul(80) / 100;
    let min_line_width = (width.saturating_mul(14) / 100).max(8);
    let allowed_gap = (width.saturating_mul(25) / 1_000).max(3);
    let row_tolerance = (height / 200).max(1);
    let mut lines = Vec::new();
    let mut current_line: Option<StoryLine> = None;

    for y in search_top..search_bottom.min(height) {
        let Some((left, right, support)) = strongest_story_run(
            frame,
            width,
            y,
            search_left,
            search_right,
            allowed_gap,
            min_line_width,
        ) else {
            if let Some(line) = current_line.take() {
                lines.push(line);
            }
            continue;
        };

        match &mut current_line {
            Some(line) if y <= line.bottom.saturating_add(row_tolerance) => {
                line.bottom = y;
                line.left = line.left.min(left);
                line.right = line.right.max(right);
                line.support = line.support.max(support);
            }
            Some(_) => {
                lines.push(current_line.replace(StoryLine {
                    top: y,
                    bottom: y,
                    left,
                    right,
                    support,
                })?);
            }
            None => {
                current_line = Some(StoryLine {
                    top: y,
                    bottom: y,
                    left,
                    right,
                    support,
                });
            }
        }
    }
    if let Some(line) = current_line {
        lines.push(line);
    }

    let min_separation = (height.saturating_mul(25) / 1_000).max(4);
    let max_separation = (height.saturating_mul(12) / 100).max(min_separation + 1);
    let line_y_tolerance = height as f32 * LINE_Y_TOLERANCE_RATIO;
    let expected_top_line_y = height as f32 * TOP_LINE_Y_RATIO;
    let expected_bottom_line_y = height as f32 * BOTTOM_LINE_Y_RATIO;
    let mut best_pair: Option<(StoryLine, StoryLine, usize)> = None;
    for first in 0..lines.len() {
        for second in first + 1..lines.len() {
            let top_line = lines[first];
            let bottom_line = lines[second];
            let top_line_center = (top_line.top + top_line.bottom) as f32 / 2.0;
            let bottom_line_center = (bottom_line.top + bottom_line.bottom) as f32 / 2.0;
            if (top_line_center - expected_top_line_y).abs() > line_y_tolerance
                || (bottom_line_center - expected_bottom_line_y).abs() > line_y_tolerance
            {
                continue;
            }
            let separation = bottom_line.top.saturating_sub(top_line.bottom);
            if separation < min_separation || separation > max_separation {
                continue;
            }

            let overlap = top_line
                .right
                .min(bottom_line.right)
                .saturating_sub(top_line.left.max(bottom_line.left));
            let shorter_width =
                (top_line.right - top_line.left).min(bottom_line.right - bottom_line.left);
            if shorter_width == 0 || overlap * 2 < shorter_width {
                continue;
            }

            let score = top_line.support.saturating_add(bottom_line.support);
            if best_pair.is_none_or(|(_, _, best_score)| score > best_score) {
                best_pair = Some((top_line, bottom_line, score));
            }
        }
    }

    let (top_line, bottom_line, _) = best_pair?;
    let left = top_line.left.min(bottom_line.left);
    let right = top_line.right.max(bottom_line.right);
    let horizontal_margin = ((right - left) / 20).max(width / 100);
    let crop_left = left.saturating_sub(horizontal_margin);
    let crop_right = right.saturating_add(horizontal_margin).min(width);
    let crop_top = top_line.bottom.saturating_add(1);
    let crop_bottom = bottom_line.top;
    (crop_right > crop_left && crop_bottom > crop_top).then_some(BoundingBox {
        left: crop_left,
        top: crop_top,
        width: crop_right - crop_left,
        height: crop_bottom - crop_top,
    })
}

fn find_combat1_crop(frame: &[u8], width: usize, height: usize) -> Option<BoundingBox> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let search_top = height.saturating_mul(2) / 100;
    let search_bottom = height.saturating_mul(12) / 100;
    let search_left = width.saturating_mul(15) / 100;
    let search_right = width.saturating_mul(85) / 100;
    let min_line_width = (width.saturating_mul(20) / 100).max(8);
    let row_tolerance = (height / 500).max(1);
    let gray_lines = find_combat_lines(
        frame,
        width,
        height,
        search_top,
        search_bottom,
        search_left,
        search_right,
        min_line_width,
        row_tolerance,
        false,
    );
    let red_lines = find_combat_lines(
        frame,
        width,
        height,
        search_top,
        search_bottom,
        search_left,
        search_right,
        min_line_width,
        row_tolerance,
        true,
    );

    let min_separation = (height.saturating_mul(4) / 1_000).max(2);
    let max_separation = (height.saturating_mul(5) / 100).max(min_separation + 1);
    let mut best_pair: Option<(CombatLine, CombatLine, usize)> = None;
    for gray_line in &gray_lines {
        for red_line in &red_lines {
            let separation = red_line.top.saturating_sub(gray_line.bottom);
            if red_line.top <= gray_line.bottom
                || separation < min_separation
                || separation > max_separation
            {
                continue;
            }

            let overlap = gray_line
                .right
                .min(red_line.right)
                .saturating_sub(gray_line.left.max(red_line.left))
                .saturating_add(1);
            let shorter_width =
                (gray_line.right - gray_line.left + 1).min(red_line.right - red_line.left + 1);
            if shorter_width == 0 || overlap.saturating_mul(2) < shorter_width {
                continue;
            }

            let score = gray_line
                .support
                .saturating_add(red_line.support)
                .saturating_add(overlap);
            if best_pair.is_none_or(|(_, _, best_score)| score > best_score) {
                best_pair = Some((*gray_line, *red_line, score));
            }
        }
    }

    let (gray_line, red_line, _) = best_pair?;
    let left = gray_line.left.min(red_line.left);
    let right = gray_line.right.max(red_line.right).saturating_add(1);
    let horizontal_margin = (width / 25).max(2);
    let crop_left = left.saturating_sub(horizontal_margin);
    let crop_right = right.saturating_add(horizontal_margin).min(width);
    let crop_top = red_line
        .bottom
        .saturating_add(1)
        .saturating_add((height / 200).max(2));
    let crop_height = (height / 20).max(12).min(height.saturating_sub(crop_top));
    (crop_right > crop_left && crop_height > 0).then_some(BoundingBox {
        left: crop_left,
        top: crop_top,
        width: crop_right - crop_left,
        height: crop_height,
    })
}

#[allow(clippy::too_many_arguments)]
fn find_combat_lines(
    frame: &[u8],
    width: usize,
    height: usize,
    search_top: usize,
    search_bottom: usize,
    search_left: usize,
    search_right: usize,
    min_line_width: usize,
    row_tolerance: usize,
    red: bool,
) -> Vec<CombatLine> {
    let mut lines = Vec::new();
    let mut current_line: Option<CombatLine> = None;

    for y in search_top..search_bottom.min(height) {
        let run = strongest_combat_run(
            frame,
            width,
            y,
            search_left,
            search_right,
            min_line_width,
            red,
        );
        let Some((left, right, support)) = run else {
            if let Some(line) = current_line.take() {
                lines.push(line);
            }
            continue;
        };

        match &mut current_line {
            Some(line) if y <= line.bottom.saturating_add(row_tolerance) => {
                line.bottom = y;
                line.left = line.left.min(left);
                line.right = line.right.max(right);
                line.support = line.support.max(support);
            }
            Some(_) => {
                lines.push(
                    current_line
                        .replace(CombatLine {
                            top: y,
                            bottom: y,
                            left,
                            right,
                            support,
                        })
                        .expect("current combat line is present"),
                );
            }
            None => {
                current_line = Some(CombatLine {
                    top: y,
                    bottom: y,
                    left,
                    right,
                    support,
                });
            }
        }
    }
    if let Some(line) = current_line {
        lines.push(line);
    }
    lines
}

fn strongest_combat_run(
    frame: &[u8],
    width: usize,
    y: usize,
    search_left: usize,
    search_right: usize,
    min_line_width: usize,
    red: bool,
) -> Option<(usize, usize, usize)> {
    let mut run_start = None;
    let mut last_support = 0;
    let mut support = 0;
    let mut best = None;

    for x in search_left..search_right.min(width) {
        if is_combat_line_pixel(frame, width, x, y, red) {
            if run_start.is_none() {
                run_start = Some(x);
            }
            last_support = x;
            support += 1;
        } else if run_start.is_some() && x.saturating_sub(last_support) > 2 {
            update_combat_run(
                &mut best,
                run_start.take()?,
                last_support,
                support,
                min_line_width,
            );
            support = 0;
        }
    }
    if let Some(run_start) = run_start {
        update_combat_run(&mut best, run_start, last_support, support, min_line_width);
    }
    best
}

fn update_combat_run(
    best: &mut Option<(usize, usize, usize)>,
    left: usize,
    right: usize,
    support: usize,
    min_line_width: usize,
) {
    let span = right.saturating_sub(left) + 1;
    if span >= min_line_width && support.saturating_mul(2) >= span {
        if best.is_none_or(|(_, _, best_support)| support > best_support) {
            *best = Some((left, right, support));
        }
    }
}

fn is_combat_line_pixel(frame: &[u8], width: usize, x: usize, y: usize, red: bool) -> bool {
    let pixel = (y * width + x) * 4;
    let red_value = u16::from(frame[pixel]);
    let green = u16::from(frame[pixel + 1]);
    let blue = u16::from(frame[pixel + 2]);
    if red {
        red_value >= 150
            && red_value > green.saturating_mul(3) / 2
            && red_value > blue.saturating_mul(3) / 2
            && green < 130
            && blue < 130
    } else {
        let minimum = red_value.min(green).min(blue);
        let maximum = red_value.max(green).max(blue);
        minimum >= 70 && maximum <= 220 && maximum - minimum <= 35
    }
}

fn strongest_story_run(
    frame: &[u8],
    width: usize,
    y: usize,
    search_left: usize,
    search_right: usize,
    allowed_gap: usize,
    min_line_width: usize,
) -> Option<(usize, usize, usize)> {
    let mut run_start = None;
    let mut last_support = 0;
    let mut support = 0;
    let mut best = None;

    for x in search_left..search_right.min(width) {
        let pixel = (y * width + x) * 4;
        let red = u32::from(frame[pixel]);
        let green = u32::from(frame[pixel + 1]);
        let blue = u32::from(frame[pixel + 2]);
        let luma = (299 * red + 587 * green + 114 * blue) / 1_000;
        if luma >= 96 {
            if run_start.is_none() {
                run_start = Some(x);
            }
            last_support = x;
            support += 1;
        } else if run_start.is_some() && x.saturating_sub(last_support) > allowed_gap {
            update_story_run(
                &mut best,
                run_start.take()?,
                last_support,
                support,
                min_line_width,
            );
            support = 0;
        }
    }

    if let Some(run_start) = run_start {
        update_story_run(&mut best, run_start, last_support, support, min_line_width);
    }
    best
}

fn update_story_run(
    best: &mut Option<(usize, usize, usize)>,
    left: usize,
    right: usize,
    support: usize,
    min_line_width: usize,
) {
    let span = right.saturating_sub(left) + 1;
    if span >= min_line_width && support.saturating_mul(2) >= span {
        if best.is_none_or(|(_, _, best_support)| support > best_support) {
            *best = Some((left, right, support));
        }
    }
}

fn is_location1_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    let pixel = (y * width + x) * 4;
    let red = frame[pixel];
    let green = frame[pixel + 1];
    let blue = frame[pixel + 2];
    red >= 100 && green >= 80 && red >= green && green >= blue && red.saturating_sub(blue) >= 12
}

fn location1_symbol_bounds(
    group: [&Component; 3],
    center_y: usize,
    center_tolerance: usize,
    max_group_dimension: usize,
) -> Option<Location1Symbol> {
    if !group
        .iter()
        .all(|component| is_roughly_diamond_shaped(component))
        || !have_similar_sizes(&group)
    {
        return None;
    }

    let mut centers: Vec<(usize, usize)> = group
        .iter()
        .map(|component| {
            (
                (component.min_x + component.max_x) / 2,
                (component.min_y + component.max_y) / 2,
            )
        })
        .collect();
    centers.sort_by_key(|center| center.1);
    let (top_x, top_y) = centers[0];
    let (middle_x, middle_y) = centers[1];
    let (bottom_x, bottom_y) = centers[2];
    let top_gap = middle_y.checked_sub(top_y)?;
    let bottom_gap = bottom_y.checked_sub(middle_y)?;
    if top_gap == 0 || bottom_gap == 0 || top_gap.abs_diff(bottom_gap) > center_tolerance {
        return None;
    }

    let aligned_x = (top_x + bottom_x) / 2;
    let max_component_width = group
        .iter()
        .map(|component| component.max_x - component.min_x + 1)
        .max()?;
    if top_x.abs_diff(bottom_x) > max_component_width
        || middle_y.abs_diff((top_y + bottom_y) / 2) > center_tolerance
        || middle_x.abs_diff(aligned_x) <= max_component_width / 2
    {
        return None;
    }

    let min_x = group.iter().map(|component| component.min_x).min()?;
    let min_y = group.iter().map(|component| component.min_y).min()?;
    let max_x = group.iter().map(|component| component.max_x).max()?;
    let max_y = group.iter().map(|component| component.max_y).max()?;
    let bounds_width = max_x - min_x + 1;
    let bounds_height = max_y - min_y + 1;
    if bounds_width >= bounds_height
        || bounds_height > max_group_dimension
        || bounds_width > max_group_dimension
        || bounds_height as f32 / bounds_width as f32 > 3.0
        || ((top_y + bottom_y) / 2).abs_diff(center_y) > center_tolerance
    {
        return None;
    }

    Some(Location1Symbol {
        bounds: BoundingBox {
            left: min_x,
            top: min_y,
            width: bounds_width,
            height: bounds_height,
        },
        side: if middle_x < aligned_x {
            Location1Side::Left
        } else {
            Location1Side::Right
        },
    })
}

fn have_similar_sizes(group: &[&Component]) -> bool {
    let min_width = group
        .iter()
        .map(|component| component.max_x - component.min_x + 1)
        .min()
        .unwrap_or(0);
    let max_width = group
        .iter()
        .map(|component| component.max_x - component.min_x + 1)
        .max()
        .unwrap_or(0);
    let min_height = group
        .iter()
        .map(|component| component.max_y - component.min_y + 1)
        .min()
        .unwrap_or(0);
    let max_height = group
        .iter()
        .map(|component| component.max_y - component.min_y + 1)
        .max()
        .unwrap_or(0);

    min_width > 0
        && min_height > 0
        && max_width as f32 / min_width as f32 <= 1.5
        && max_height as f32 / min_height as f32 <= 1.5
}

fn find_diamond_symbols_in_band(
    state: &ProcessingState,
    frame: &[u8],
    center_y: usize,
    is_target_pixel: fn(&[u8], usize, usize, usize) -> bool,
) -> Vec<BoundingBox> {
    let width = state.frame_width;
    let height = state.frame_height;
    let Some(expected_len) = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return Vec::new();
    };
    if width == 0 || height == 0 || frame.len() < expected_len {
        return Vec::new();
    }

    let max_group_dimension = ((height as f32 * 0.06).ceil() as usize).max(12);
    let band_radius = max_group_dimension;
    let first_y = center_y.saturating_sub(band_radius);
    let last_y = center_y.saturating_add(band_radius).min(height - 1);
    let band_height = last_y - first_y + 1;
    let Some(band_size) = width.checked_mul(band_height) else {
        return Vec::new();
    };
    let mut visited = vec![false; band_size];
    let max_component_dimension = ((height as f32 * 0.025).ceil() as usize).max(5);
    let mut components = Vec::new();

    for y in first_y..=last_y {
        for x in 0..width {
            let band_index = (y - first_y) * width + x;
            if visited[band_index] || !is_target_pixel(frame, width, x, y) {
                continue;
            }

            let component = collect_component(
                frame,
                width,
                first_y,
                last_y,
                x,
                y,
                max_component_dimension.saturating_mul(max_component_dimension),
                is_target_pixel,
                &mut visited,
            );
            let component_width = component.max_x - component.min_x + 1;
            let component_height = component.max_y - component.min_y + 1;
            if component.area >= 3
                && component_width >= 2
                && component_height >= 2
                && component_width <= max_component_dimension
                && component_height <= max_component_dimension
                && (component_width as f32 / component_height as f32) >= 0.5
                && (component_width as f32 / component_height as f32) <= 2.0
            {
                components.push(component);
            }
        }
    }

    components.sort_by_key(|component| component.min_x);
    let center_tolerance = ((height as f32 * 0.012).ceil() as usize).max(2);
    let mut symbols = Vec::new();

    for first in 0..components.len() {
        for second in first + 1..components.len() {
            if components[second].min_x - components[first].min_x > max_group_dimension {
                break;
            }
            for third in second + 1..components.len() {
                if components[third].min_x - components[first].min_x > max_group_dimension {
                    break;
                }
                for fourth in third + 1..components.len() {
                    if components[fourth].min_x - components[first].min_x > max_group_dimension {
                        break;
                    }
                    let group = [
                        &components[first],
                        &components[second],
                        &components[third],
                        &components[fourth],
                    ];
                    if let Some(symbol) =
                        symbol_bounds(&group, center_y, center_tolerance, max_group_dimension)
                    {
                        if !symbols.contains(&symbol) {
                            symbols.push(symbol);
                        }
                    }
                }
            }
        }
    }

    symbols.sort_by_key(|symbol| symbol.left);
    symbols
}

fn is_dark_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    let pixel = (y * width + x) * 4;
    let red = u32::from(frame[pixel]);
    let green = u32::from(frame[pixel + 1]);
    let blue = u32::from(frame[pixel + 2]);
    (299 * red + 587 * green + 114 * blue) / 1000 < 128
}

fn is_bright_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    let pixel = (y * width + x) * 4;
    let red = frame[pixel];
    let green = frame[pixel + 1];
    let blue = frame[pixel + 2];
    red >= 180
        && green >= 180
        && blue >= 180
        && red.max(green).max(blue) - red.min(green).min(blue) <= 45
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

fn collect_component(
    frame: &[u8],
    width: usize,
    first_y: usize,
    last_y: usize,
    start_x: usize,
    start_y: usize,
    max_stored_pixels: usize,
    is_target_pixel: fn(&[u8], usize, usize, usize) -> bool,
    visited: &mut [bool],
) -> Component {
    let mut pending = vec![(start_x, start_y)];
    let start_index = (start_y - first_y) * width + start_x;
    visited[start_index] = true;
    let mut component = Component {
        min_x: start_x,
        min_y: start_y,
        max_x: start_x,
        max_y: start_y,
        area: 0,
        pixels: Some(Vec::new()),
    };

    while let Some((x, y)) = pending.pop() {
        component.min_x = component.min_x.min(x);
        component.min_y = component.min_y.min(y);
        component.max_x = component.max_x.max(x);
        component.max_y = component.max_y.max(y);
        component.area += 1;
        if let Some(pixels) = &mut component.pixels {
            if pixels.len() < max_stored_pixels {
                pixels.push((x, y));
            } else {
                component.pixels = None;
            }
        }

        let neighbors = [
            x.checked_sub(1).map(|neighbor_x| (neighbor_x, y)),
            (x + 1 < width).then_some((x + 1, y)),
            y.checked_sub(1)
                .filter(|neighbor_y| *neighbor_y >= first_y)
                .map(|neighbor_y| (x, neighbor_y)),
            (y < last_y).then_some((x, y + 1)),
        ];
        for (neighbor_x, neighbor_y) in neighbors.into_iter().flatten() {
            let neighbor_index = (neighbor_y - first_y) * width + neighbor_x;
            if !visited[neighbor_index] && is_target_pixel(frame, width, neighbor_x, neighbor_y) {
                visited[neighbor_index] = true;
                pending.push((neighbor_x, neighbor_y));
            }
        }
    }

    component
}

fn symbol_bounds(
    group: &[&Component; 4],
    center_y: usize,
    center_tolerance: usize,
    max_group_dimension: usize,
) -> Option<BoundingBox> {
    let min_x = group.iter().map(|part| part.min_x).min()?;
    let min_y = group.iter().map(|part| part.min_y).min()?;
    let max_x = group.iter().map(|part| part.max_x).max()?;
    let max_y = group.iter().map(|part| part.max_y).max()?;
    let bounds_width = max_x - min_x + 1;
    let bounds_height = max_y - min_y + 1;
    if bounds_width > max_group_dimension
        || bounds_height > max_group_dimension
        || !is_aspect_ratio_within(bounds_width, bounds_height, MAX_GROUP_ASPECT_RATIO)
    {
        return None;
    }
    if !has_three_similar_diamonds(group) {
        return None;
    }

    let centers: Vec<(f32, f32)> = group
        .iter()
        .map(|part| {
            (
                (part.min_x + part.max_x) as f32 / 2.0,
                (part.min_y + part.max_y) as f32 / 2.0,
            )
        })
        .collect();
    let center_x = centers.iter().map(|center| center.0).sum::<f32>() / 4.0;
    let group_center_y = centers.iter().map(|center| center.1).sum::<f32>() / 4.0;
    if (group_center_y - center_y as f32).abs() > center_tolerance as f32 {
        return None;
    }

    let mut radii_and_angles: Vec<(f32, f32)> = centers
        .iter()
        .map(|(x, y)| {
            let dx = x - center_x;
            let dy = y - group_center_y;
            (dx.hypot(dy), dy.atan2(dx))
        })
        .collect();
    radii_and_angles.sort_by(|left, right| left.1.total_cmp(&right.1));
    let min_radius = radii_and_angles
        .iter()
        .map(|point| point.0)
        .fold(f32::INFINITY, f32::min);
    let max_radius = radii_and_angles
        .iter()
        .map(|point| point.0)
        .fold(0.0, f32::max);
    if min_radius < 1.0 || max_radius > min_radius * 2.5 {
        return None;
    }

    for index in 0..radii_and_angles.len() {
        let current = radii_and_angles[index].1;
        let next = radii_and_angles[(index + 1) % radii_and_angles.len()].1
            + if index + 1 == radii_and_angles.len() {
                std::f32::consts::TAU
            } else {
                0.0
            };
        let gap = next - current;
        if !(std::f32::consts::PI / 6.0..=5.0 * std::f32::consts::PI / 6.0).contains(&gap) {
            return None;
        }
    }

    Some(BoundingBox {
        left: min_x,
        top: min_y,
        width: bounds_width,
        height: bounds_height,
    })
}

fn has_three_similar_diamonds(group: &[&Component; 4]) -> bool {
    let diamonds: Vec<&Component> = group
        .iter()
        .copied()
        .filter(|component| is_roughly_diamond_shaped(component))
        .collect();
    if diamonds.len() < 3 {
        return false;
    }

    let min_width = diamonds
        .iter()
        .map(|component| component.max_x - component.min_x + 1)
        .min()
        .unwrap_or(0);
    let max_width = diamonds
        .iter()
        .map(|component| component.max_x - component.min_x + 1)
        .max()
        .unwrap_or(0);
    let min_height = diamonds
        .iter()
        .map(|component| component.max_y - component.min_y + 1)
        .min()
        .unwrap_or(0);
    let max_height = diamonds
        .iter()
        .map(|component| component.max_y - component.min_y + 1)
        .max()
        .unwrap_or(0);

    min_width > 0
        && min_height > 0
        && max_width as f32 / min_width as f32 <= 1.5
        && max_height as f32 / min_height as f32 <= 1.5
}

fn is_roughly_diamond_shaped(component: &Component) -> bool {
    let width = component.max_x - component.min_x + 1;
    let height = component.max_y - component.min_y + 1;
    if width < 3 || height < 3 || !is_aspect_ratio_within(width, height, MAX_COMPONENT_ASPECT_RATIO)
    {
        return false;
    }
    let Some(pixels) = &component.pixels else {
        return false;
    };
    if pixels.is_empty() {
        return false;
    }

    let center_x = (component.min_x + component.max_x) as f32 / 2.0;
    let center_y = (component.min_y + component.max_y) as f32 / 2.0;
    let radius_x = (width - 1) as f32 / 2.0;
    let radius_y = (height - 1) as f32 / 2.0;
    let diamond_pixels = pixels
        .iter()
        .filter(|(x, y)| {
            ((*x as f32 - center_x).abs() / radius_x) + ((*y as f32 - center_y).abs() / radius_y)
                <= 1.25
        })
        .count();

    diamond_pixels as f32 / pixels.len() as f32 >= 0.8
}

fn is_aspect_ratio_within(width: usize, height: usize, max_ratio: f32) -> bool {
    let shorter_side = width.min(height);
    shorter_side > 0 && width.max(height) as f32 / shorter_side as f32 <= max_ratio
}

fn recognize_text(
    frame: &[u8],
    frame_width: usize,
    frame_height: usize,
    crop: &BoundingBox,
    frame_number: u64,
) -> String {
    ocr_result_or_empty(
        recognize_text_inner(frame, frame_width, frame_height, crop),
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
) -> io::Result<String> {
    let rgb_crop = crop_rgba_to_rgb(frame, frame_width, frame_height, crop)?;
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
        add_combat1, add_location1, add_location2, add_story, add_textact, find_combat1_crop,
        find_diamond_symbols, find_location1_symbols, find_location_symbols, find_story_crop,
        write_history_single_line, BoundingBox, Component, HistoryEntry, Location1Side,
        ProcessingState,
    };

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
    fn finds_four_diamond_symbols_on_centerline() {
        let width = 480;
        let height = 300;
        let mut frame = vec![255; width * height * 4];
        draw_symbol(&mut frame, width, 120, height / 2);
        draw_symbol(&mut frame, width, 360, height / 2);
        let state = ProcessingState {
            current_frame: 0,
            frame_width: width,
            frame_height: height,
            frame_size: frame.len(),
            frame_rate_num: 1,
            frame_rate_den: 1,
        };

        assert_eq!(
            find_diamond_symbols(&state, &frame),
            vec![
                BoundingBox {
                    left: 114,
                    top: 144,
                    width: 13,
                    height: 13,
                },
                BoundingBox {
                    left: 354,
                    top: 144,
                    width: 13,
                    height: 13,
                },
            ]
        );
    }

    #[test]
    fn finds_white_symbols_in_upper_location_band() {
        let width = 480;
        let height = 300;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        draw_white_symbol(&mut frame, width, 120, height * 24 / 100);
        draw_white_symbol(&mut frame, width, 360, height * 24 / 100);
        let state = ProcessingState {
            current_frame: 0,
            frame_width: width,
            frame_height: height,
            frame_size: frame.len(),
            frame_rate_num: 1,
            frame_rate_den: 1,
        };

        assert_eq!(
            find_location_symbols(&state, &frame),
            vec![
                BoundingBox {
                    left: 114,
                    top: 66,
                    width: 13,
                    height: 13,
                },
                BoundingBox {
                    left: 354,
                    top: 66,
                    width: 13,
                    height: 13,
                },
            ]
        );
    }

    #[test]
    fn finds_mirrored_tall_gold_three_diamond_location_symbols() {
        let width = 480;
        let height = 300;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        let center_y = height * 24 / 100;
        draw_location1_symbol(&mut frame, width, 100, center_y, false);
        draw_location1_symbol(&mut frame, width, 380, center_y, true);
        let state = ProcessingState {
            current_frame: 0,
            frame_width: width,
            frame_height: height,
            frame_size: frame.len(),
            frame_rate_num: 1,
            frame_rate_den: 1,
        };

        let symbols = find_location1_symbols(&state, &frame);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0].side, Location1Side::Left);
        assert_eq!(symbols[1].side, Location1Side::Right);
        assert!(symbols
            .iter()
            .all(|symbol| symbol.bounds.height > symbol.bounds.width));
    }

    #[test]
    fn finds_story_crop_between_bright_segmented_horizontal_rules() {
        let width = 480;
        let height = 300;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        for y in [223, 224, 238, 239] {
            for x in 120..360 {
                let gap = x % 43 < 5;
                let intensity = if x % 31 < 13 {
                    [150, 130, 90, 255]
                } else {
                    [110, 95, 65, 255]
                };
                let pixel = (y * width + x) * 4;
                if !gap {
                    frame[pixel..pixel + 4].copy_from_slice(&intensity);
                }
            }
        }
        let crop = find_story_crop(&frame, width, height).unwrap();

        assert_eq!(crop.top, 225);
        assert_eq!(crop.height, 13);
        assert!(crop.left < 120);
        assert!(crop.left + crop.width > 360);
    }

    #[test]
    fn finds_combat1_crop_below_overlapping_gray_and_red_rules() {
        let width = 200;
        let height = 200;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        for (y, color) in [(12, [150, 150, 150, 255]), (22, [230, 40, 55, 255])] {
            for x in 30..170 {
                let pixel = (y * width + x) * 4;
                frame[pixel..pixel + 4].copy_from_slice(&color);
            }
        }

        assert_eq!(
            find_combat1_crop(&frame, width, height),
            Some(BoundingBox {
                left: 22,
                top: 25,
                width: 156,
                height: 12,
            })
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
    fn rejects_frames_shorter_than_rgba_buffer_size() {
        let width = 2;
        let height = 2;
        let frame = vec![255; width * height * 3];
        let state = ProcessingState {
            current_frame: 0,
            frame_width: width,
            frame_height: height,
            frame_size: width * height * 4,
            frame_rate_num: 1,
            frame_rate_den: 1,
        };

        assert!(super::find_diamond_symbols(&state, &frame).is_empty());
    }

    #[test]
    fn requires_three_similar_diamond_components() {
        let first = diamond_component(46, 46, 2);
        let second = diamond_component(54, 46, 2);
        let third = diamond_component(46, 54, 2);
        let fourth = square_component(54, 54, 2);
        assert!(super::has_three_similar_diamonds(&[
            &first, &second, &third, &fourth
        ]));

        let fourth = square_component(46, 54, 2);
        let third = square_component(54, 54, 2);
        assert!(!super::has_three_similar_diamonds(&[
            &first, &second, &third, &fourth
        ]));
    }

    #[test]
    fn rejects_three_diamonds_with_mismatched_sizes() {
        let first = diamond_component(46, 46, 2);
        let second = diamond_component(54, 46, 2);
        let third = diamond_component(46, 54, 4);
        let fourth = square_component(54, 54, 2);

        assert!(!super::has_three_similar_diamonds(&[
            &first, &second, &third, &fourth
        ]));
    }

    #[test]
    fn requires_symbol_bounds_to_be_roughly_square() {
        assert!(super::is_aspect_ratio_within(12, 10, 1.2));
        assert!(super::is_aspect_ratio_within(10, 12, 1.2));
        assert!(!super::is_aspect_ratio_within(13, 10, 1.2));
        assert!(!super::is_aspect_ratio_within(10, 13, 1.2));
        assert!(!super::is_aspect_ratio_within(0, 10, 1.2));
    }

    #[test]
    fn checks_aspect_ratio_of_all_four_grouped_components() {
        let top_left = diamond_component(95, 95, 1);
        let top_right = diamond_component(105, 95, 1);
        let bottom_left = diamond_component(95, 105, 1);
        let bottom_right = diamond_component(105, 105, 1);
        assert_eq!(
            super::symbol_bounds(
                &[&top_left, &top_right, &bottom_left, &bottom_right],
                100,
                2,
                20,
            ),
            Some(BoundingBox {
                left: 94,
                top: 94,
                width: 13,
                height: 13,
            })
        );

        let top_left = diamond_component(94, 96, 1);
        let top_right = diamond_component(106, 96, 1);
        let bottom_left = diamond_component(94, 104, 1);
        let bottom_right = diamond_component(106, 104, 1);
        assert!(super::symbol_bounds(
            &[&top_left, &top_right, &bottom_left, &bottom_right],
            100,
            2,
            20,
        )
        .is_none());

        let top_left = diamond_component(93, 97, 1);
        let top_right = diamond_component(107, 97, 1);
        let bottom_left = diamond_component(93, 103, 1);
        let bottom_right = diamond_component(107, 103, 1);
        assert!(super::symbol_bounds(
            &[&top_left, &top_right, &bottom_left, &bottom_right],
            100,
            2,
            20,
        )
        .is_none());
    }

    #[test]
    fn crops_full_gap_with_text_height_scaled_from_symbols() {
        let left = BoundingBox {
            left: 100,
            top: 50,
            width: 20,
            height: 20,
        };
        let right = BoundingBox {
            left: 200,
            top: 52,
            width: 20,
            height: 20,
        };

        assert_eq!(
            super::text_crop_between_symbols(&left, &right, 300, 120),
            Some(BoundingBox {
                left: 120,
                top: 41,
                width: 80,
                height: 40,
            })
        );
    }

    #[test]
    fn title_crop_uses_full_frame_width_below_symbol_pair() {
        let left = BoundingBox {
            left: 100,
            top: 50,
            width: 20,
            height: 20,
        };
        let right = BoundingBox {
            left: 200,
            top: 52,
            width: 20,
            height: 20,
        };

        assert_eq!(
            super::title_crop_below_symbols(&left, &right, 300, 160),
            Some(BoundingBox {
                left: 0,
                top: 91,
                width: 300,
                height: 40,
            })
        );
    }

    #[test]
    fn skips_symbol_pairs_with_different_size_or_vertical_position() {
        let left = BoundingBox {
            left: 100,
            top: 50,
            width: 20,
            height: 20,
        };
        let different_size = BoundingBox {
            left: 200,
            top: 50,
            width: 30,
            height: 20,
        };
        let different_height = BoundingBox {
            left: 200,
            top: 80,
            width: 20,
            height: 20,
        };

        assert!(super::text_crop_between_symbols(&left, &different_size, 300, 120).is_none());
        assert!(super::text_crop_between_symbols(&left, &different_height, 300, 120).is_none());
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

    fn diamond_component(center_x: usize, center_y: usize, radius: usize) -> Component {
        let mut pixels = Vec::new();
        for y in center_y - radius..=center_y + radius {
            for x in center_x - radius..=center_x + radius {
                if x.abs_diff(center_x) + y.abs_diff(center_y) <= radius {
                    pixels.push((x, y));
                }
            }
        }
        Component {
            min_x: center_x - radius,
            min_y: center_y - radius,
            max_x: center_x + radius,
            max_y: center_y + radius,
            area: pixels.len(),
            pixels: Some(pixels),
        }
    }

    fn square_component(center_x: usize, center_y: usize, radius: usize) -> Component {
        let pixels: Vec<(usize, usize)> = (center_y - radius..=center_y + radius)
            .flat_map(|y| (center_x - radius..=center_x + radius).map(move |x| (x, y)))
            .collect();
        Component {
            min_x: center_x - radius,
            min_y: center_y - radius,
            max_x: center_x + radius,
            max_y: center_y + radius,
            area: pixels.len(),
            pixels: Some(pixels),
        }
    }

    fn draw_symbol(frame: &mut [u8], width: usize, center_x: usize, center_y: usize) {
        draw_symbol_with_color(frame, width, center_x, center_y, [0, 0, 0, 255]);
    }

    fn draw_white_symbol(frame: &mut [u8], width: usize, center_x: usize, center_y: usize) {
        draw_symbol_with_color(frame, width, center_x, center_y, [255, 255, 255, 255]);
    }

    fn draw_location1_symbol(
        frame: &mut [u8],
        width: usize,
        center_x: usize,
        center_y: usize,
        side_right: bool,
    ) {
        let side_x = if side_right {
            center_x + 6
        } else {
            center_x - 6
        };
        for (diamond_x, diamond_y) in [
            (center_x, center_y - 6),
            (center_x, center_y + 6),
            (side_x, center_y),
        ] {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx.abs() + dy.abs() <= 1 {
                        let x = (diamond_x as isize + dx) as usize;
                        let y = (diamond_y as isize + dy) as usize;
                        let pixel = (y * width + x) * 4;
                        frame[pixel..pixel + 4].copy_from_slice(&[190, 186, 165, 255]);
                    }
                }
            }
        }
    }

    fn draw_symbol_with_color(
        frame: &mut [u8],
        width: usize,
        center_x: usize,
        center_y: usize,
        color: [u8; 4],
    ) {
        for (offset_x, offset_y) in [(-4, -4), (4, -4), (-4, 4), (4, 4)] {
            let diamond_x = (center_x as isize + offset_x) as usize;
            let diamond_y = (center_y as isize + offset_y) as usize;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    if dx.abs() + dy.abs() <= 2 {
                        let x = (diamond_x as isize + dx) as usize;
                        let y = (diamond_y as isize + dy) as usize;
                        let pixel = (y * width + x) * 4;
                        frame[pixel..pixel + 4].copy_from_slice(&color);
                    }
                }
            }
        }
    }
}
