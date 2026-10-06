use crate::frame_processor::BoundingBox;

pub(crate) struct GraceDetection {
    pub(crate) symbol: BoundingBox,
    pub(crate) label: BoundingBox,
}

pub(crate) struct PlaceDetection {
    pub(crate) line: BoundingBox,
    pub(crate) text: BoundingBox,
}

pub(crate) struct BossDetection {
    pub(crate) line: BoundingBox,
    pub(crate) text: BoundingBox,
}

pub(crate) struct LostGraceDiscoveredDetection {
    pub(crate) first_l: BoundingBox,
    pub(crate) final_d: BoundingBox,
    pub(crate) text: BoundingBox,
}

pub(crate) struct Region2Detection {
    pub(crate) banner: BoundingBox,
    pub(crate) text: BoundingBox,
}

struct OrangeComponent {
    bounds: BoundingBox,
    pixels: Vec<(usize, usize)>,
}

pub(crate) fn find_grace_detection(
    frame: &[u8],
    width: usize,
    height: usize,
) -> Option<GraceDetection> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let expected_size = (height.saturating_mul(55) / 1_000).max(12);
    let max_dimension = width.min(height);
    let min_width = (expected_size.saturating_mul(80) / 100).max(8);
    let max_width = (expected_size.saturating_mul(120) / 100).min(max_dimension);
    let min_height = (expected_size.saturating_mul(85) / 100).max(8);
    let max_height = (expected_size.saturating_mul(115) / 100).min(max_dimension);
    if min_width > max_width || min_height > max_height {
        return None;
    }

    let expected_left = width.saturating_mul(25) / 1_000;
    let expected_top = height.saturating_mul(12) / 100;
    let x_radius = (width.saturating_mul(12) / 1_000).max(2);
    let y_radius = (height.saturating_mul(15) / 1_000).max(2);
    let position_step = (expected_size / 12).max(1);
    let dimension_step = (expected_size / 20).max(1);
    let x_positions = candidate_positions(
        expected_left,
        x_radius,
        width.saturating_sub(min_width),
        position_step,
    );
    let y_positions = candidate_positions(
        expected_top,
        y_radius,
        height.saturating_sub(min_height),
        position_step,
    );
    let widths = candidate_positions(
        expected_size,
        expected_size.saturating_mul(20) / 100,
        max_dimension,
        dimension_step,
    )
    .into_iter()
    .filter(|candidate| (*candidate >= min_width) && (*candidate <= max_width))
    .collect::<Vec<_>>();
    let heights = candidate_positions(
        expected_size,
        expected_size.saturating_mul(15) / 100,
        max_dimension,
        dimension_step,
    )
    .into_iter()
    .filter(|candidate| (*candidate >= min_height) && (*candidate <= max_height))
    .collect::<Vec<_>>();

    let mut best: Option<(f32, BoundingBox)> = None;
    for &box_width in &widths {
        for &left in &x_positions {
            if left.saturating_add(box_width) > width {
                continue;
            }
            for &box_height in &heights {
                for &top in &y_positions {
                    if top.saturating_add(box_height) > height {
                        continue;
                    }
                    let Some(border_score) =
                        border_score(frame, width, height, left, top, box_width, box_height)
                    else {
                        continue;
                    };
                    let position_penalty = left.abs_diff(expected_left) as f32
                        / x_radius.max(1) as f32
                        + top.abs_diff(expected_top) as f32 / y_radius.max(1) as f32;
                    let size_penalty = (box_width.abs_diff(expected_size)
                        + box_height.abs_diff(expected_size))
                        as f32
                        / (expected_size.max(1) * 2) as f32;
                    let score = border_score - position_penalty * 0.08 - size_penalty * 0.12;
                    if best
                        .as_ref()
                        .is_none_or(|(best_score, _)| score > *best_score)
                    {
                        best = Some((
                            score,
                            BoundingBox {
                                left,
                                top,
                                width: box_width,
                                height: box_height,
                            },
                        ));
                    }
                }
            }
        }
    }

    let (_, symbol) = best?;
    let label_left = symbol
        .left
        .saturating_add(symbol.width)
        .saturating_add((width.saturating_mul(8) / 1_000).max(2));
    let label_top = symbol
        .top
        .saturating_add(symbol.height.saturating_mul(15) / 100);
    if label_left >= width || label_top >= height {
        return None;
    }
    let label_height = (symbol.height.saturating_mul(75) / 100)
        .max(5)
        .min(height - label_top);
    let label_width = (width.saturating_mul(26) / 100).min(width - label_left);
    (label_width >= 20 && label_height >= 5).then_some(GraceDetection {
        symbol,
        label: BoundingBox {
            left: label_left,
            top: label_top,
            width: label_width,
            height: label_height,
        },
    })
}

pub(crate) fn find_place_detection(
    frame: &[u8],
    width: usize,
    height: usize,
) -> Option<PlaceDetection> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let expected_y = height.saturating_mul(423) / 1_000;
    let y_radius = (height.saturating_mul(18) / 1_000).max(2);
    let expected_length = (width.saturating_mul(272) / 1_000).max(24);
    let min_length = expected_length.saturating_mul(75) / 100;
    let max_length = expected_length.saturating_mul(125) / 100;
    let search_left = width.saturating_mul(20) / 100;
    let search_right = width.saturating_mul(80) / 100;
    let mut best: Option<(f32, BoundingBox)> = None;

    for y in
        expected_y.saturating_sub(y_radius)..=expected_y.saturating_add(y_radius).min(height - 1)
    {
        let Some((left, right, support)) = place_line_run(
            frame,
            width,
            height,
            y,
            search_left,
            search_right,
            (expected_length / 100).max(2),
        ) else {
            continue;
        };
        let line_width = right - left + 1;
        let center = left + line_width / 2;
        if line_width < min_length
            || line_width > max_length
            || center.abs_diff(width / 2) > width.saturating_mul(6) / 100
            || support * 2 < line_width
        {
            continue;
        }

        let coverage = support as f32 / line_width as f32;
        let width_penalty = line_width.abs_diff(expected_length) as f32 / expected_length as f32;
        let y_penalty = y.abs_diff(expected_y) as f32 / y_radius.max(1) as f32;
        let score = coverage - width_penalty * 0.35 - y_penalty * 0.08;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((
                score,
                BoundingBox {
                    left,
                    top: y,
                    width: line_width,
                    height: 1,
                },
            ));
        }
    }

    let (_, line) = best?;
    let text_height = (height.saturating_mul(55) / 1_000).max(18).min(line.top);
    let gap = (height.saturating_mul(3) / 1_000).max(1);
    let text_top = line.top.saturating_sub(text_height.saturating_add(gap));
    (line.width >= 20 && text_top < line.top).then_some(PlaceDetection {
        text: BoundingBox {
            left: line.left,
            top: text_top,
            width: line.width,
            height: line.top - gap - text_top,
        },
        line,
    })
}

pub(crate) fn find_boss_detection(
    frame: &[u8],
    width: usize,
    height: usize,
) -> Option<BossDetection> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let expected_left = width.saturating_mul(242) / 1_000;
    let expected_y = height.saturating_mul(780) / 1_000;
    let y_radius = (height.saturating_mul(40) / 1_000).max(2);
    let min_width = width.saturating_add(1).saturating_div(2).max(24);
    let min_red_thickness = (height.saturating_mul(5) / 1_000).max(2);
    let search_left = width.saturating_mul(20) / 100;
    let search_right = width.saturating_mul(80) / 100;
    let allowed_gap = (width / 300).max(2);
    let mut best: Option<(f32, BoundingBox)> = None;

    for y in
        expected_y.saturating_sub(y_radius)..=expected_y.saturating_add(y_radius).min(height - 1)
    {
        let Some((left, right, support)) =
            boss_line_run(frame, width, y, search_left, search_right, allowed_gap)
        else {
            continue;
        };
        let line_width = right - left + 1;
        if line_width < min_width
            || support.saturating_mul(100) < line_width.saturating_mul(55)
            || left.abs_diff(expected_left) > width.saturating_mul(35) / 1_000
        {
            continue;
        }
        let red_thickness = boss_red_thickness(frame, width, height, left, right, y);
        if red_thickness < min_red_thickness {
            continue;
        }
        let position_penalty =
            left.abs_diff(expected_left) as f32 / (width.saturating_mul(35) / 1_000).max(1) as f32;
        let y_penalty = y.abs_diff(expected_y) as f32 / y_radius.max(1) as f32;
        let score = support as f32 - position_penalty * 20.0 - y_penalty * 2.0;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((
                score,
                BoundingBox {
                    left,
                    top: y,
                    width: line_width,
                    height: 1,
                },
            ));
        }
    }

    let (_, line) = best?;
    let text_height = (height.saturating_mul(45) / 1_000).max(16).min(line.top);
    let gap = (height.saturating_mul(4) / 1_000).max(2);
    let text_top = line.top.saturating_sub(text_height.saturating_add(gap));
    let text_width = (line.width.saturating_mul(40) / 100).min(width - line.left);
    (text_width >= 24 && text_top < line.top).then_some(BossDetection {
        text: BoundingBox {
            left: line.left,
            top: text_top,
            width: text_width,
            height: line.top - gap - text_top,
        },
        line,
    })
}

pub(crate) fn find_lost_grace_discovered_detection(
    frame: &[u8],
    width: usize,
    height: usize,
) -> Option<LostGraceDiscoveredDetection> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let expected_top = height.saturating_mul(481) / 1_000;
    let expected_glyph_height = (height.saturating_mul(52) / 1_000).max(8);
    let y_radius = (height.saturating_mul(18) / 1_000).max(2);
    let first_l = find_banner_glyph(
        frame,
        width,
        height,
        width.saturating_mul(220) / 1_000,
        width.saturating_mul(15) / 1_000,
        expected_top,
        y_radius,
        expected_glyph_height,
        true,
    )?;
    let final_d = find_banner_glyph(
        frame,
        width,
        height,
        width.saturating_mul(751) / 1_000,
        width.saturating_mul(15) / 1_000,
        expected_top,
        y_radius,
        expected_glyph_height,
        false,
    )?;
    if first_l.left >= final_d.left
        || first_l.top.abs_diff(final_d.top) > (height.saturating_mul(12) / 1_000).max(2)
    {
        return None;
    }

    let text_left = first_l
        .left
        .saturating_sub((width.saturating_mul(6) / 1_000).max(2));
    let text_right = final_d
        .left
        .saturating_add(final_d.width)
        .saturating_add((width.saturating_mul(6) / 1_000).max(2))
        .min(width);
    let text_top = first_l
        .top
        .saturating_sub((height.saturating_mul(12) / 1_000).max(2));
    let text_bottom = first_l
        .top
        .saturating_add(first_l.height.max(final_d.height))
        .saturating_add((height.saturating_mul(12) / 1_000).max(2))
        .min(height);
    (text_right > text_left && text_bottom > text_top).then_some(LostGraceDiscoveredDetection {
        first_l,
        final_d,
        text: BoundingBox {
            left: text_left,
            top: text_top,
            width: text_right - text_left,
            height: text_bottom - text_top,
        },
    })
}

pub(crate) fn find_region2_detection(
    frame: &[u8],
    width: usize,
    height: usize,
) -> Option<Region2Detection> {
    let expected_len = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || frame.len() < expected_len {
        return None;
    }

    let band_width = width.saturating_mul(192) / 1_000;
    if band_width < 24 {
        return None;
    }
    let band_left = (width - band_width) / 2;
    let band_right = band_left + band_width;
    let sample_rows = (height.saturating_mul(2) / 1_000).max(1);
    let top = find_region2_transition(
        frame,
        width,
        height,
        band_left,
        band_right,
        height.saturating_mul(320) / 1_000,
        height.saturating_mul(400) / 1_000,
        sample_rows,
        true,
    )?;
    let bottom = find_region2_transition(
        frame,
        width,
        height,
        band_left,
        band_right,
        height.saturating_mul(400) / 1_000,
        height.saturating_mul(480) / 1_000,
        sample_rows,
        false,
    )?;
    let band_height = bottom.saturating_sub(top);
    let minimum_height = (height.saturating_mul(40) / 1_000).max(2);
    let maximum_height = height.saturating_mul(90) / 1_000;
    if band_height < minimum_height || band_height > maximum_height {
        return None;
    }

    Some(Region2Detection {
        banner: BoundingBox {
            left: band_left,
            top,
            width: band_width,
            height: band_height,
        },
        text: BoundingBox {
            left: band_left,
            top,
            width: band_width,
            height: band_height,
        },
    })
}

fn find_region2_transition(
    frame: &[u8],
    width: usize,
    height: usize,
    band_left: usize,
    band_right: usize,
    search_start: usize,
    search_end: usize,
    sample_rows: usize,
    bright_to_dark: bool,
) -> Option<usize> {
    let search_start = search_start.max(sample_rows);
    let search_end = search_end.min(height.saturating_sub(sample_rows));
    if search_start > search_end {
        return None;
    }

    let mut best: Option<(u32, usize)> = None;
    for y in search_start..=search_end {
        let before = (y - sample_rows..y)
            .map(|sample_y| region2_row_luma(frame, width, band_left, band_right, sample_y))
            .sum::<u32>()
            / sample_rows as u32;
        let after = (y..y + sample_rows)
            .map(|sample_y| region2_row_luma(frame, width, band_left, band_right, sample_y))
            .sum::<u32>()
            / sample_rows as u32;
        let contrast = if bright_to_dark {
            before.saturating_sub(after)
        } else {
            after.saturating_sub(before)
        };
        if best.is_none_or(|(best_contrast, _)| contrast > best_contrast) {
            best = Some((contrast, y));
        }
    }

    best.filter(|(contrast, _)| *contrast >= 21).map(|(_, y)| y)
}

fn region2_row_luma(
    frame: &[u8],
    width: usize,
    band_left: usize,
    band_right: usize,
    y: usize,
) -> u32 {
    let total_luma = (band_left..band_right)
        .map(|x| {
            let pixel = (y * width + x) * 4;
            (299 * u32::from(frame[pixel])
                + 587 * u32::from(frame[pixel + 1])
                + 114 * u32::from(frame[pixel + 2]))
                / 1_000
        })
        .sum::<u32>();
    total_luma / (band_right - band_left) as u32
}

#[allow(clippy::too_many_arguments)]
fn find_banner_glyph(
    frame: &[u8],
    width: usize,
    height: usize,
    expected_left: usize,
    x_radius: usize,
    expected_top: usize,
    y_radius: usize,
    expected_height: usize,
    is_l: bool,
) -> Option<BoundingBox> {
    let expected_width = if is_l {
        expected_height.saturating_mul(68) / 100
    } else {
        expected_height.saturating_mul(96) / 100
    };
    let maximum_glyph_width = expected_height.saturating_mul(125) / 100;
    let left = expected_left.saturating_sub(x_radius);
    let top = expected_top.saturating_sub(y_radius);
    let right = expected_left
        .saturating_add(x_radius)
        .saturating_add(maximum_glyph_width)
        .min(width);
    let bottom = expected_top
        .saturating_add(expected_height)
        .saturating_add(y_radius)
        .min(height);
    if right <= left || bottom <= top {
        return None;
    }

    let components = orange_components(frame, width, left, top, right, bottom);
    let min_height = expected_height.saturating_mul(75) / 100;
    let max_height = expected_height.saturating_mul(130) / 100;
    let mut best: Option<(f32, BoundingBox)> = None;
    for component in components {
        let shape_matches = banner_glyph_shape(&component, is_l);
        let bounds = component.bounds;
        if bounds.height < min_height
            || bounds.height > max_height
            || bounds.left.abs_diff(expected_left) > x_radius
            || bounds.top.abs_diff(expected_top) > y_radius
            || !shape_matches
        {
            continue;
        }

        let min_width = if is_l {
            bounds.height.saturating_mul(45) / 100
        } else {
            bounds.height.saturating_mul(70) / 100
        };
        let max_width = if is_l {
            bounds.height.saturating_mul(90) / 100
        } else {
            bounds.height.saturating_mul(125) / 100
        };
        if bounds.width < min_width || bounds.width > max_width {
            continue;
        }
        let area = bounds.width.saturating_mul(bounds.height).max(1);
        if component.pixels.len().saturating_mul(100) < area.saturating_mul(14) {
            continue;
        }

        let position_penalty = bounds.left.abs_diff(expected_left) as f32 / x_radius.max(1) as f32
            + bounds.top.abs_diff(expected_top) as f32 / y_radius.max(1) as f32;
        let width_penalty =
            bounds.width.abs_diff(expected_width) as f32 / expected_width.max(1) as f32;
        let score = component.pixels.len() as f32 / area as f32
            - position_penalty * 0.12
            - width_penalty * 0.08;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((score, bounds));
        }
    }
    best.map(|(_, bounds)| bounds)
}

fn orange_components(
    frame: &[u8],
    frame_width: usize,
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
) -> Vec<OrangeComponent> {
    let region_width = right - left;
    let region_height = bottom - top;
    let mut visited = vec![false; region_width * region_height];
    let mut pending = Vec::new();
    let mut components = Vec::new();

    for y in top..bottom {
        for x in left..right {
            let region_index = (y - top) * region_width + (x - left);
            if visited[region_index] || !is_banner_orange_pixel(frame, frame_width, x, y) {
                continue;
            }

            pending.clear();
            pending.push((x, y));
            visited[region_index] = true;
            let mut pixels = Vec::new();
            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;
            while let Some((pixel_x, pixel_y)) = pending.pop() {
                pixels.push((pixel_x, pixel_y));
                min_x = min_x.min(pixel_x);
                max_x = max_x.max(pixel_x);
                min_y = min_y.min(pixel_y);
                max_y = max_y.max(pixel_y);
                for neighbor_y in
                    pixel_y.saturating_sub(1).max(top)..=pixel_y.saturating_add(1).min(bottom - 1)
                {
                    for neighbor_x in pixel_x.saturating_sub(1).max(left)
                        ..=pixel_x.saturating_add(1).min(right - 1)
                    {
                        let neighbor_index =
                            (neighbor_y - top) * region_width + (neighbor_x - left);
                        if !visited[neighbor_index]
                            && is_banner_orange_pixel(frame, frame_width, neighbor_x, neighbor_y)
                        {
                            visited[neighbor_index] = true;
                            pending.push((neighbor_x, neighbor_y));
                        }
                    }
                }
            }
            if pixels.len() >= 12 {
                components.push(OrangeComponent {
                    bounds: BoundingBox {
                        left: min_x,
                        top: min_y,
                        width: max_x - min_x + 1,
                        height: max_y - min_y + 1,
                    },
                    pixels,
                });
            }
        }
    }
    components
}

fn is_banner_orange_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    let pixel = (y * width + x) * 4;
    let red = u16::from(frame[pixel]);
    let green = u16::from(frame[pixel + 1]);
    let blue = u16::from(frame[pixel + 2]);
    red >= 135 && red * 100 >= green * 135 && green >= 45 && green * 100 >= blue * 115 && blue < 130
}

fn banner_glyph_shape(component: &OrangeComponent, is_l: bool) -> bool {
    let bounds = &component.bounds;
    let mut left_rows = vec![false; bounds.height];
    let mut right_rows = vec![false; bounds.height];
    let mut bottom_columns = vec![false; bounds.width];
    let mut top_columns = vec![false; bounds.width];
    for &(x, y) in &component.pixels {
        let x_offset = x - bounds.left;
        let y_offset = y - bounds.top;
        if x_offset * 100 <= bounds.width * 28 {
            left_rows[y_offset] = true;
        }
        if x_offset * 100 >= bounds.width * 72 {
            right_rows[y_offset] = true;
        }
        if y_offset * 100 >= bounds.height * 82 {
            bottom_columns[x_offset] = true;
        }
        if y_offset * 100 <= bounds.height * 18 {
            top_columns[x_offset] = true;
        }
    }
    let left_support = left_rows.into_iter().filter(|present| *present).count();
    if is_l {
        let bottom_support = bottom_columns
            .into_iter()
            .filter(|present| *present)
            .count();
        left_support * 100 >= bounds.height * 55 && bottom_support * 100 >= bounds.width * 55
    } else {
        let right_support = right_rows.into_iter().filter(|present| *present).count();
        let top_support = top_columns.into_iter().filter(|present| *present).count();
        let bottom_support = bottom_columns
            .into_iter()
            .filter(|present| *present)
            .count();
        left_support * 100 >= bounds.height * 55
            && right_support * 100 >= bounds.height * 45
            && top_support * 100 >= bounds.width * 35
            && bottom_support * 100 >= bounds.width * 35
    }
}

fn place_line_run(
    frame: &[u8],
    width: usize,
    height: usize,
    y: usize,
    search_left: usize,
    search_right: usize,
    allowed_gap: usize,
) -> Option<(usize, usize, usize)> {
    let mut best = None;
    let mut start = None;
    let mut last_support = 0;
    let mut support = 0;

    for x in search_left..search_right.min(width) {
        if is_place_line_pixel(frame, width, height, x, y) {
            if start.is_none() {
                start = Some(x);
            }
            last_support = x;
            support += 1;
        } else if start.is_some() && x.saturating_sub(last_support) > allowed_gap {
            update_place_run(&mut best, start.take()?, last_support, support);
            support = 0;
        }
    }
    if let Some(start) = start {
        update_place_run(&mut best, start, last_support, support);
    }
    best
}

fn boss_line_run(
    frame: &[u8],
    width: usize,
    y: usize,
    search_left: usize,
    search_right: usize,
    allowed_gap: usize,
) -> Option<(usize, usize, usize)> {
    let mut best = None;
    let mut start = None;
    let mut last_support = 0;
    let mut support = 0;

    for x in search_left..search_right.min(width) {
        if is_boss_line_pixel(frame, width, x, y) {
            if start.is_none() {
                start = Some(x);
            }
            last_support = x;
            support += 1;
        } else if start.is_some() && x.saturating_sub(last_support) > allowed_gap {
            update_place_run(&mut best, start.take()?, last_support, support);
            support = 0;
        }
    }
    if let Some(start) = start {
        update_place_run(&mut best, start, last_support, support);
    }
    best
}

fn is_boss_line_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    is_boss_red_pixel(frame, width, x, y, 55)
}

fn is_boss_fill_pixel(frame: &[u8], width: usize, x: usize, y: usize) -> bool {
    is_boss_red_pixel(frame, width, x, y, 35)
}

fn is_boss_red_pixel(frame: &[u8], width: usize, x: usize, y: usize, minimum_red: u16) -> bool {
    let pixel = (y * width + x) * 4;
    let red = u16::from(frame[pixel]);
    let green = u16::from(frame[pixel + 1]);
    let blue = u16::from(frame[pixel + 2]);
    red >= minimum_red && red * 100 >= green * 200 && red * 100 >= blue * 180
}

fn boss_red_thickness(
    frame: &[u8],
    width: usize,
    height: usize,
    left: usize,
    right: usize,
    top: usize,
) -> usize {
    let line_width = right - left + 1;
    let minimum_support = line_width.saturating_mul(55).saturating_add(99) / 100;
    let mut thickness = 0;
    for y in top..height {
        let support = (left..=right)
            .filter(|x| is_boss_fill_pixel(frame, width, *x, y))
            .count();
        if support < minimum_support {
            break;
        }
        thickness += 1;
    }
    thickness
}

fn update_place_run(
    best: &mut Option<(usize, usize, usize)>,
    start: usize,
    end: usize,
    support: usize,
) {
    if best.is_none_or(|(_, _, best_support)| support > best_support) {
        *best = Some((start, end, support));
    }
}

fn is_place_line_pixel(frame: &[u8], width: usize, height: usize, x: usize, y: usize) -> bool {
    if y == 0 || y + 1 >= height {
        return false;
    }
    let pixel = (y * width + x) * 4;
    let red = frame[pixel];
    let green = frame[pixel + 1];
    let blue = frame[pixel + 2];
    let minimum = red.min(green).min(blue);
    let maximum = red.max(green).max(blue);
    let luma = (299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue)) / 1_000;
    let above = ((y - 1) * width + x) * 4;
    let below = ((y + 1) * width + x) * 4;
    let above_luma = (299 * u32::from(frame[above])
        + 587 * u32::from(frame[above + 1])
        + 114 * u32::from(frame[above + 2]))
        / 1_000;
    let below_luma = (299 * u32::from(frame[below])
        + 587 * u32::from(frame[below + 1])
        + 114 * u32::from(frame[below + 2]))
        / 1_000;
    (45..=230).contains(&luma)
        && maximum - minimum <= 55
        && luma.abs_diff(above_luma).max(luma.abs_diff(below_luma)) >= 6
}

fn candidate_positions(center: usize, radius: usize, limit: usize, step: usize) -> Vec<usize> {
    let start = center.saturating_sub(radius).min(limit);
    let end = center.saturating_add(radius).min(limit);
    let mut positions = (start..=end).step_by(step).collect::<Vec<_>>();
    positions.push(center.clamp(start, end));
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn border_score(
    frame: &[u8],
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    box_width: usize,
    box_height: usize,
) -> Option<f32> {
    let mut supported = [0usize; 4];
    let sample_count = (box_width.min(box_height) / 2).max(8);
    let border_width = (box_width.min(box_height) / 32).max(1).min(2);
    for sample in 0..sample_count {
        let x_offset = 2 + sample * (box_width - 4) / sample_count;
        let y_offset = 2 + sample * (box_height - 4) / sample_count;
        for offset in 0..border_width {
            let candidates = [
                (
                    (left + x_offset, top + offset),
                    (left + x_offset, top.saturating_sub(1)),
                ),
                (
                    (left + x_offset, top + box_height - 1 - offset),
                    (left + x_offset, top + box_height),
                ),
                (
                    (left + offset, top + y_offset),
                    (left.saturating_sub(1), top + y_offset),
                ),
                (
                    (left + box_width - 1 - offset, top + y_offset),
                    (left + box_width, top + y_offset),
                ),
            ];
            for (edge, ((x, y), (neighbor_x, neighbor_y))) in candidates.into_iter().enumerate() {
                if is_border_pixel(frame, width, height, x, y, neighbor_x, neighbor_y) {
                    supported[edge] += 1;
                }
            }
        }
    }

    let edge_samples = sample_count * border_width;
    let ratios = supported.map(|support| support as f32 / edge_samples as f32);
    if ratios.iter().any(|ratio| *ratio < 0.45) {
        return None;
    }
    Some(ratios.iter().sum::<f32>() / ratios.len() as f32)
}

fn is_border_pixel(
    frame: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    neighbor_x: usize,
    neighbor_y: usize,
) -> bool {
    if neighbor_x >= width || neighbor_y >= height {
        return false;
    }
    let pixel = (y * width + x) * 4;
    let red = frame[pixel];
    let green = frame[pixel + 1];
    let blue = frame[pixel + 2];
    let minimum = red.min(green).min(blue);
    let maximum = red.max(green).max(blue);
    let luma = (299 * u32::from(red) + 587 * u32::from(green) + 114 * u32::from(blue)) / 1_000;
    let neighbor = (neighbor_y * width + neighbor_x) * 4;
    let neighbor_luma = (299 * u32::from(frame[neighbor])
        + 587 * u32::from(frame[neighbor + 1])
        + 114 * u32::from(frame[neighbor + 2]))
        / 1_000;
    minimum >= 45 && maximum <= 230 && maximum - minimum <= 70 && luma.abs_diff(neighbor_luma) >= 12
}

#[cfg(test)]
mod tests {
    use super::{
        find_boss_detection, find_grace_detection, find_lost_grace_discovered_detection,
        find_place_detection, find_region2_detection,
    };

    #[test]
    fn finds_the_scaled_grace_box_and_label_crop() {
        for (width, height) in [(480, 300), (1_365, 768)] {
            let left = width * 25 / 1_000;
            let top = height * 12 / 100;
            let size = height * 55 / 1_000;
            let box_width = size * 92 / 100;
            let mut frame = vec![0; width * height * 4];
            for pixel in frame.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            for offset in 0..2 {
                for x_offset in 0..box_width {
                    set_gray_pixel(&mut frame, width, left + x_offset, top + offset);
                    set_gray_pixel(&mut frame, width, left + x_offset, top + size - 1 - offset);
                }
                for y_offset in 0..size {
                    set_gray_pixel(&mut frame, width, left + offset, top + y_offset);
                    set_gray_pixel(
                        &mut frame,
                        width,
                        left + box_width - 1 - offset,
                        top + y_offset,
                    );
                }
            }

            let detection = find_grace_detection(&frame, width, height).unwrap();
            assert_eq!(detection.symbol.left, left);
            assert_eq!(detection.symbol.top, top);
            assert_eq!(detection.symbol.width, box_width);
            assert_eq!(detection.symbol.height, size);
            assert!(detection.label.left > detection.symbol.left + detection.symbol.width);
            assert!(detection.label.width > 0 && detection.label.height > 0);
        }
    }

    #[test]
    fn rejects_frames_without_a_grace_box() {
        let width = 480;
        let height = 300;
        let frame = vec![0; width * height * 4];

        assert!(find_grace_detection(&frame, width, height).is_none());
        assert!(find_grace_detection(&frame[..10], width, height).is_none());

        let gray_frame = vec![100; width * height * 4];
        assert!(find_grace_detection(&gray_frame, width, height).is_none());
    }

    #[test]
    fn finds_the_scaled_place_underline_and_text_crop() {
        for (width, height) in [(1_365, 768), (1_920, 1_080)] {
            let expected_y = height * 423 / 1_000;
            let expected_length = width * 272 / 1_000;
            let left = (width - expected_length) / 2;
            let mut frame = vec![0; width * height * 4];
            for pixel in frame.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[32, 32, 32, 255]);
            }
            for x in left..left + expected_length {
                set_pixel(&mut frame, width, x, expected_y, [150, 150, 150, 255]);
            }

            let detection = find_place_detection(&frame, width, height).unwrap();
            assert_eq!(detection.line.top, expected_y);
            assert_eq!(detection.line.left, left);
            assert_eq!(detection.line.width, expected_length);
            assert!(detection.text.top < detection.line.top);
            assert!(detection.text.left <= detection.line.left);
        }
    }

    #[test]
    fn rejects_frames_without_a_place_underline() {
        let width = 480;
        let height = 300;
        let frame = vec![32; width * height * 4];

        assert!(find_place_detection(&frame, width, height).is_none());
        assert!(find_place_detection(&frame[..10], width, height).is_none());
    }

    #[test]
    fn finds_the_scaled_boss_health_line_and_name_crop() {
        for (width, height) in [(1_365usize, 768usize), (1_920, 1_080)] {
            let line_width = width.saturating_add(1).saturating_div(2);
            let red_thickness = (height * 7 / 1_000).max(2);
            let (frame, left, top) = boss_bar_frame(width, height, line_width, red_thickness, true);

            let detection = find_boss_detection(&frame, width, height).unwrap();
            assert_eq!(detection.line.left, left);
            assert_eq!(detection.line.top, top);
            assert_eq!(detection.line.width, line_width);
            assert_eq!(detection.text.left, left);
            assert!(detection.text.top < detection.line.top);
            assert!(detection.text.top + detection.text.height < detection.line.top);
            assert!(detection.text.width > 0 && detection.text.height > 0);
        }
    }

    #[test]
    fn finds_a_full_boss_health_line_at_joris_vertical_position() {
        let width = 1_920;
        let height = 1_080;
        let line_width = width * 52 / 100;
        let red_thickness = (height * 7 / 1_000).max(2);
        let top = height * 755 / 1_000;
        let (frame, left, _) =
            boss_bar_frame_at(width, height, line_width, red_thickness, false, top);

        let detection = find_boss_detection(&frame, width, height).unwrap();

        assert_eq!(detection.line.left, left);
        assert!((top..top + red_thickness).contains(&detection.line.top));
        assert_eq!(detection.line.width, line_width);
    }

    #[test]
    fn rejects_boss_health_lines_shorter_than_half_the_frame_width() {
        let width = 1_365;
        let height = 768;
        let line_width = width * 49 / 100;
        let red_thickness = (height * 7 / 1_000).max(2);
        let (frame, _, _) = boss_bar_frame(width, height, line_width, red_thickness, true);

        assert!(find_boss_detection(&frame, width, height).is_none());
    }

    #[test]
    fn requires_minimum_red_thickness_but_not_a_bright_outline() {
        let width = 1_365;
        let height = 768;
        let line_width = width * 52 / 100;
        let min_red_thickness = (height * 5 / 1_000).max(2);
        let (thin_frame, _, _) =
            boss_bar_frame(width, height, line_width, min_red_thickness - 1, true);
        let (no_bright_line_frame, _, _) =
            boss_bar_frame(width, height, line_width, min_red_thickness + 1, false);

        assert!(find_boss_detection(&thin_frame, width, height).is_none());
        assert!(find_boss_detection(&no_bright_line_frame, width, height).is_some());
    }

    #[test]
    fn finds_boss_line_with_dim_outline_below_dark_red_edge() {
        let width = 1_920;
        let height = 1_080;
        let line_width = width / 2;
        let red_thickness = (height * 5 / 1_000).max(2);
        let (mut frame, left, top) =
            boss_bar_frame(width, height, line_width, red_thickness, false);
        let red_bottom = top + red_thickness - 1;
        let outline_y = top + red_thickness;
        for x in left..left + line_width {
            set_pixel(&mut frame, width, x, red_bottom, [41, 0, 0, 255]);
            set_pixel(&mut frame, width, x, outline_y, [96, 50, 50, 255]);
        }

        assert!(find_boss_detection(&frame, width, height).is_some());
    }

    #[test]
    fn rejects_frames_without_a_boss_health_line() {
        let width = 1_920;
        let height = 1_080;
        let frame = vec![32; width * height * 4];

        assert!(find_boss_detection(&frame, width, height).is_none());
        assert!(find_boss_detection(&frame[..10], width, height).is_none());
    }

    #[test]
    fn finds_the_scaled_region2_band_and_text_crop() {
        for (width, height) in [(1_365, 768), (1_920, 1_080)] {
            let band_width = width * 192 / 1_000;
            let band_left = (width - band_width) / 2;
            let top = height * 360 / 1_000;
            let bottom = height * 424 / 1_000;
            let mut frame = vec![0; width * height * 4];
            for pixel in frame.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[150, 150, 150, 255]);
            }
            for y in top..bottom {
                for x in band_left..band_left + band_width {
                    set_pixel(&mut frame, width, x, y, [125, 125, 125, 255]);
                }
            }
            for y in top + (bottom - top) / 3..top + (bottom - top) * 2 / 3 {
                for x in band_left + band_width / 4..band_left + band_width * 3 / 4 {
                    set_pixel(&mut frame, width, x, y, [225, 225, 225, 255]);
                }
            }

            let detection = find_region2_detection(&frame, width, height).unwrap();
            assert_eq!(detection.banner.left, band_left);
            assert_eq!(detection.banner.width, band_width);
            assert_eq!(detection.banner.top, top);
            assert_eq!(detection.banner.height, bottom - top);
            assert_eq!(detection.text.left, band_left);
            assert_eq!(detection.text.top, top);
            assert_eq!(detection.text.width, band_width);
            assert_eq!(detection.text.height, bottom - top);
        }
    }

    #[test]
    fn rejects_region2_frames_without_bright_dark_bright_transitions() {
        let width = 1_920;
        let height = 1_080;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[100, 100, 100, 255]);
        }

        assert!(find_region2_detection(&frame, width, height).is_none());
        assert!(find_region2_detection(&frame[..10], width, height).is_none());
    }

    #[test]
    fn finds_lost_grace_banner_from_l_and_d_at_multiple_resolutions() {
        for (width, height) in [(1_365, 768), (1_920, 1_080)] {
            let mut frame = vec![24; width * height * 4];
            for pixel in frame.chunks_exact_mut(4) {
                pixel[3] = 255;
            }
            let glyph_height = height * 52 / 1_000;
            let l_left = width * 220 / 1_000;
            let d_left = width * 751 / 1_000;
            let glyph_top = height * 481 / 1_000;
            draw_l(&mut frame, width, l_left, glyph_top, glyph_height);
            draw_d(&mut frame, width, d_left, glyph_top, glyph_height);

            let detection = find_lost_grace_discovered_detection(&frame, width, height).unwrap();
            assert!(detection.first_l.left.abs_diff(l_left) <= 2);
            assert!(detection.final_d.left.abs_diff(d_left) <= 2);
            assert!(detection.text.left < detection.first_l.left);
            assert!(detection.text.left + detection.text.width > detection.final_d.left);
        }
    }

    #[test]
    fn rejects_lost_grace_banner_when_end_anchor_is_missing() {
        let width = 1_920;
        let height = 1_080;
        let mut frame = vec![24; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        draw_l(
            &mut frame,
            width,
            width * 220 / 1_000,
            height * 481 / 1_000,
            height * 52 / 1_000,
        );

        assert!(find_lost_grace_discovered_detection(&frame, width, height).is_none());
    }

    fn set_gray_pixel(frame: &mut [u8], width: usize, x: usize, y: usize) {
        set_pixel(frame, width, x, y, [150, 150, 150, 255]);
    }

    fn set_pixel(frame: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 4]) {
        let pixel = (y * width + x) * 4;
        frame[pixel..pixel + 4].copy_from_slice(&color);
    }

    fn boss_bar_frame(
        width: usize,
        height: usize,
        line_width: usize,
        red_thickness: usize,
        include_bright_line: bool,
    ) -> (Vec<u8>, usize, usize) {
        let top = height * 806 / 1_000;
        boss_bar_frame_at(
            width,
            height,
            line_width,
            red_thickness,
            include_bright_line,
            top,
        )
    }

    fn boss_bar_frame_at(
        width: usize,
        height: usize,
        line_width: usize,
        red_thickness: usize,
        include_bright_line: bool,
        top: usize,
    ) -> (Vec<u8>, usize, usize) {
        let left = width * 242 / 1_000;
        let mut frame = vec![0; width * height * 4];
        for pixel in frame.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        for y in top..top + red_thickness {
            for x in left..left + line_width {
                set_pixel(&mut frame, width, x, y, [120, 8, 12, 255]);
            }
        }
        if include_bright_line {
            for x in left..left + line_width {
                set_pixel(
                    &mut frame,
                    width,
                    x,
                    top + red_thickness,
                    [200, 200, 200, 255],
                );
            }
        }
        (frame, left, top)
    }

    fn draw_l(frame: &mut [u8], width: usize, left: usize, top: usize, height: usize) {
        let glyph_width = height * 68 / 100;
        for y in top..top + height {
            for x in left..left + (height * 13 / 100).max(2) {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
        }
        for y in top + height * 82 / 100..top + height {
            for x in left..left + glyph_width {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
        }
    }

    fn draw_d(frame: &mut [u8], width: usize, left: usize, top: usize, height: usize) {
        let glyph_width = height * 96 / 100;
        let stroke = (height * 12 / 100).max(2);
        for y in top..top + height {
            for x in left..left + stroke {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
            let curve_x = left + glyph_width - stroke - (y.abs_diff(top + height / 2) / 5);
            for x in curve_x..left + glyph_width {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
        }
        for y in top..top + stroke {
            for x in left..left + glyph_width - stroke / 2 {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
        }
        for y in top + height - stroke..top + height {
            for x in left..left + glyph_width - stroke / 2 {
                set_pixel(frame, width, x, y, [220, 125, 40, 255]);
            }
        }
    }
}
