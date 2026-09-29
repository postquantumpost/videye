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
}

pub(crate) struct ProcessingState {
    pub(crate) current_frame: u64,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
    pub(crate) frame_size: usize,
    pub(crate) frame_rate_num: u32,
    pub(crate) frame_rate_den: u32,
}

pub(crate) fn process_frame(
    state: &mut ProcessingState,
    frame: &[u8],
    output_file: &mut File,
) -> io::Result<()> {
    state.current_frame += 1;
    let elapsed_ns = u128::from(state.current_frame - 1)
        * u128::from(state.frame_rate_den)
        * 1_000_000_000
        / u128::from(state.frame_rate_num);
    let elapsed_seconds = elapsed_ns / 1_000_000_000;
    let hours = elapsed_seconds / 3_600;
    let minutes = (elapsed_seconds / 60) % 60;
    let seconds = elapsed_seconds % 60;
    let nanoseconds = elapsed_ns % 1_000_000_000;
    writeln!(
        output_file,
        "Frame {} at {:02}:{:02}:{:02}.{:09}: {}x{} ({} RGB bytes)",
        state.current_frame,
        hours,
        minutes,
        seconds,
        nanoseconds,
        state.frame_width,
        state.frame_height,
        frame.len()
    )?;
    writeln!(output_file, "{hours:02}:{minutes:02}:{seconds:02}")?;

    let symbols = find_diamond_symbols(state, frame);
    for symbol in symbols {
        writeln!(
            output_file,
            "Diamond symbol: x={} y={} width={} height={}",
            symbol.left, symbol.top, symbol.width, symbol.height
        )?;
    }
    Ok(())
}

fn find_diamond_symbols(state: &ProcessingState, frame: &[u8]) -> Vec<BoundingBox> {
    let width = state.frame_width;
    let height = state.frame_height;
    let Some(expected_len) = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
    else {
        return Vec::new();
    };
    if width == 0 || height == 0 || frame.len() < expected_len {
        return Vec::new();
    }

    let center_y = height / 2;
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
            if visited[band_index] || !is_dark_pixel(frame, width, x, y) {
                continue;
            }

            let component = collect_component(frame, width, first_y, last_y, x, y, &mut visited);
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
    let pixel = (y * width + x) * 3;
    let red = u32::from(frame[pixel]);
    let green = u32::from(frame[pixel + 1]);
    let blue = u32::from(frame[pixel + 2]);
    (299 * red + 587 * green + 114 * blue) / 1000 < 128
}

fn collect_component(
    frame: &[u8],
    width: usize,
    first_y: usize,
    last_y: usize,
    start_x: usize,
    start_y: usize,
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
    };

    while let Some((x, y)) = pending.pop() {
        component.min_x = component.min_x.min(x);
        component.min_y = component.min_y.min(y);
        component.max_x = component.max_x.max(x);
        component.max_y = component.max_y.max(y);
        component.area += 1;

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
            if !visited[neighbor_index] && is_dark_pixel(frame, width, neighbor_x, neighbor_y) {
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
    if bounds_width > max_group_dimension || bounds_height > max_group_dimension {
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

fn recognize_text(state: &ProcessingState, frame: &[u8]) -> io::Result<String> {
    let mut tesseract = Command::new("tesseract")
        .args(["stdin", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = tesseract.stdin.take().ok_or_else(|| {
        io::Error::new(io::ErrorKind::BrokenPipe, "failed to open Tesseract stdin")
    })?;
    write!(stdin, "P6\n{} {}\n255\n", state.frame_width, state.frame_height)?;
    stdin.write_all(frame)?;
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

#[cfg(test)]
mod tests {
    use super::{find_diamond_symbols, BoundingBox, ProcessingState};

    #[test]
    fn finds_four_diamond_symbols_on_centerline() {
        let width = 480;
        let height = 300;
        let mut frame = vec![255; width * height * 3];
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

    fn draw_symbol(frame: &mut [u8], width: usize, center_x: usize, center_y: usize) {
        for (offset_x, offset_y) in [(-4, -4), (4, -4), (-4, 4), (4, 4)] {
            let diamond_x = (center_x as isize + offset_x) as usize;
            let diamond_y = (center_y as isize + offset_y) as usize;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    if dx.abs() + dy.abs() <= 2 {
                        let x = (diamond_x as isize + dx) as usize;
                        let y = (diamond_y as isize + dy) as usize;
                        let pixel = (y * width + x) * 3;
                        frame[pixel..pixel + 3].fill(0);
                    }
                }
            }
        }
    }
}