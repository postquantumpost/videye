use crate::frame_processor::{
    merge_history, process_frame, write_history, write_history_single_line, History, OcrCache,
    ProcessingState,
};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Instant;

struct FrameProcessingResult {
    log: Vec<u8>,
    annotated_frame: Option<Vec<u8>>,
    history: History,
}

pub fn process_files(
    input: &str,
    output: &str,
    video_output: Option<&str>,
    parallel_count: usize,
    check_story_line_thickness: bool,
    use_tesseract_library: bool,
) -> Result<(), String> {
    let started_at = Instant::now();
    let message = match video_output {
        Some(video_output) => format!("Process {input} to {output} and {video_output}."),
        None => format!("Process {input} to {output}."),
    };
    let mut output_file = File::create(output)
        .map_err(|error| format!("failed to create output file {output}: {error}"))?;
    writeln!(output_file, "{message}")
        .map_err(|error| format!("failed to write output file {output}: {error}"))?;
    println!("{message}");

    let (width, height, frame_rate_num, frame_rate_den) = video_dimensions(input)?;
    let frame_size = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "video frame dimensions are too large".to_string())?;
    let input_uri = file_uri(input)?;
    let mut decoder = Command::new("gst-launch-1.0")
        .args(["-q", "uridecodebin"])
        .arg(format!("uri={input_uri}"))
        .args([
            "!",
            "queue",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=RGBA",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("failed to start GStreamer decoder: {error}"))?;
    let mut video_encoder = video_output
        .map(|video_output| {
            start_video_encoder(video_output, width, height, frame_rate_num, frame_rate_den)
        })
        .transpose()?;
    let mut video_frames = video_encoder
        .as_mut()
        .map(|encoder| {
            encoder
                .stdin
                .take()
                .ok_or_else(|| "failed to write encoded video frames".to_string())
        })
        .transpose()?;
    let stdout = decoder
        .stdout
        .take()
        .ok_or_else(|| "failed to read decoded video frames".to_string())?;
    let mut frames = BufReader::new(stdout);
    let mut state = ProcessingState {
        current_frame: 0,
        frame_width: width,
        frame_height: height,
        frame_size,
        frame_rate_num,
        frame_rate_den,
        check_story_line_thickness,
    };
    let mut history = History::new();
    let ocr_cache = OcrCache::default();
    loop {
        let mut frame_batch = Vec::with_capacity(parallel_count);
        for _ in 0..parallel_count {
            let mut frame = vec![0; state.frame_size];
            let mut bytes_read = 0;
            while bytes_read < state.frame_size {
                match frames.read(&mut frame[bytes_read..]) {
                    Ok(0) if bytes_read == 0 => break,
                    Ok(0) => return Err("decoder returned an incomplete video frame".to_string()),
                    Ok(count) => bytes_read += count,
                    Err(error) => {
                        return Err(format!("failed to read decoded video frame: {error}"))
                    }
                }
            }

            if bytes_read == 0 {
                break;
            }
            frame_batch.push(frame);
        }

        if frame_batch.is_empty() {
            break;
        }

        let results = process_frame_batch(
            frame_batch,
            &state,
            video_output.is_some(),
            use_tesseract_library,
            &ocr_cache,
        )?;
        for result in results {
            output_file
                .write_all(&result.log)
                .map_err(|error| format!("failed to write frame log: {error}"))?;
            if let (Some(video_frames), Some(annotated_frame)) =
                (video_frames.as_mut(), result.annotated_frame)
            {
                video_frames
                    .write_all(&annotated_frame)
                    .map_err(|error| format!("failed to write encoded video frame: {error}"))?;
            }
            merge_history(&mut history, result.history);
            state.current_frame += 1;
        }
    }

    let status = decoder
        .wait()
        .map_err(|error| format!("failed to wait for GStreamer decoder: {error}"))?;
    if !status.success() {
        return Err(format!("GStreamer decoder exited with status {status}"));
    }

    drop(video_frames);
    if let Some(mut video_encoder) = video_encoder {
        let status = video_encoder
            .wait()
            .map_err(|error| format!("failed to wait for GStreamer video encoder: {error}"))?;
        if !status.success() {
            return Err(format!(
                "GStreamer video encoder exited with status {status}"
            ));
        }
    }

    write_history(&history, &mut output_file)
        .map_err(|error| format!("failed to write text history to {output}: {error}"))?;
    write_history_single_line(&history, &mut output_file).map_err(|error| {
        format!("failed to write single-line text history to {output}: {error}")
    })?;

    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let frames_per_second = if elapsed_seconds > 0.0 {
        state.current_frame as f64 / elapsed_seconds
    } else {
        0.0
    };
    let summary = format!(
        "Processed {} frames in {:.2} seconds ({:.2} FPS).",
        state.current_frame, elapsed_seconds, frames_per_second
    );
    writeln!(output_file, "{summary}")
        .map_err(|error| format!("failed to write processing summary to {output}: {error}"))?;
    println!("{summary}");

    Ok(())
}

fn process_frame_batch(
    frames: Vec<Vec<u8>>,
    state: &ProcessingState,
    annotate_frames: bool,
    use_tesseract_library: bool,
    ocr_cache: &OcrCache,
) -> Result<Vec<FrameProcessingResult>, String> {
    let frame_width = state.frame_width;
    let frame_height = state.frame_height;
    let frame_size = state.frame_size;
    let frame_rate_num = state.frame_rate_num;
    let frame_rate_den = state.frame_rate_den;
    let check_story_line_thickness = state.check_story_line_thickness;
    let first_frame_number = state.current_frame;

    thread::scope(|scope| {
        let handles = frames
            .into_iter()
            .enumerate()
            .map(|(offset, frame)| {
                let current_frame = first_frame_number + offset as u64;
                scope.spawn(move || -> Result<FrameProcessingResult, String> {
                    let mut frame_state = ProcessingState {
                        current_frame,
                        frame_width,
                        frame_height,
                        frame_size,
                        frame_rate_num,
                        frame_rate_den,
                        check_story_line_thickness,
                    };
                    let mut log = Vec::new();
                    let mut history = History::new();
                    let annotated_frame = process_frame(
                        &mut frame_state,
                        &frame,
                        &mut log,
                        annotate_frames,
                        use_tesseract_library,
                        ocr_cache,
                        &mut history,
                    )
                    .map_err(|error| {
                        format!(
                            "failed to process video frame {}: {error}",
                            current_frame + 1
                        )
                    })?;
                    Ok(FrameProcessingResult {
                        log,
                        annotated_frame,
                        history,
                    })
                })
            })
            .collect::<Vec<_>>();

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| "frame processing thread panicked".to_string())??;
            results.push(result);
        }
        Ok(results)
    })
}

fn start_video_encoder(
    output: &str,
    width: usize,
    height: usize,
    frame_rate_num: u32,
    frame_rate_den: u32,
) -> Result<Child, String> {
    let extension = Path::new(output)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (encoder, encoder_option, parser, muxer) = match extension.as_str() {
        "mp4" | "m4v" | "mov" => ("x264enc", "tune=zerolatency", Some("h264parse"), "mp4mux"),
        "mkv" => (
            "x264enc",
            "tune=zerolatency",
            Some("h264parse"),
            "matroskamux",
        ),
        "webm" => ("vp8enc", "deadline=1", None, "webmmux"),
        _ => {
            return Err(format!(
                "unsupported video output extension for {output}; use .mp4, .m4v, .mov, .mkv, or .webm"
            ));
        }
    };

    let mut arguments = vec![
        "-q".to_string(),
        "fdsrc".to_string(),
        "fd=0".to_string(),
        "!".to_string(),
        "rawvideoparse".to_string(),
        "format=rgba".to_string(),
        format!("width={width}"),
        format!("height={height}"),
        format!("framerate={frame_rate_num}/{frame_rate_den}"),
        "!".to_string(),
        "videoconvert".to_string(),
        "!".to_string(),
        encoder.to_string(),
        encoder_option.to_string(),
        "!".to_string(),
    ];
    if let Some(parser) = parser {
        arguments.push(parser.to_string());
        arguments.push("!".to_string());
    }
    arguments.extend([
        muxer.to_string(),
        "!".to_string(),
        "filesink".to_string(),
        format!("location={output}"),
    ]);

    Command::new("gst-launch-1.0")
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("failed to start GStreamer video encoder: {error}"))
}

fn video_dimensions(input: &str) -> Result<(usize, usize, u32, u32), String> {
    let result = Command::new("gst-discoverer-1.0")
        .arg(input)
        .output()
        .map_err(|error| format!("failed to start GStreamer video discovery: {error}"))?;
    if !result.status.success() {
        return Err(format!("failed to inspect input video {input}"));
    }

    let details = String::from_utf8_lossy(&result.stdout);
    let mut width = None;
    let mut height = None;
    let mut frame_rate = None;
    for line in details.lines() {
        if width.is_none() {
            width = line
                .trim()
                .strip_prefix("Width:")
                .and_then(|value| value.trim().parse().ok());
        }
        if height.is_none() {
            height = line
                .trim()
                .strip_prefix("Height:")
                .and_then(|value| value.trim().parse().ok());
        }
        if frame_rate.is_none() {
            frame_rate = line
                .trim()
                .strip_prefix("Frame rate:")
                .and_then(|value| value.trim().split_once('/'))
                .and_then(|(numerator, denominator)| {
                    Some((numerator.parse().ok()?, denominator.parse().ok()?))
                });
        }
    }

    match (width, height, frame_rate) {
        (Some(width), Some(height), Some((rate_num, rate_den)))
            if width > 0 && height > 0 && rate_num > 0 && rate_den > 0 =>
        {
            Ok((width, height, rate_num, rate_den))
        }
        _ => Err(format!("could not determine video dimensions for {input}")),
    }
}

fn file_uri(input: &str) -> Result<String, String> {
    let path = std::fs::canonicalize(input)
        .map_err(|error| format!("failed to open input video {input}: {error}"))?;
    let mut uri = String::from("file://");
    for byte in path.to_string_lossy().bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.' | b'~') {
            uri.push(byte as char);
        } else {
            uri.push_str(&format!("%{byte:02X}"));
        }
    }
    Ok(uri)
}
