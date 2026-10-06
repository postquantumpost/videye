use crate::frame_processor::{
    merge_history, process_frame, write_history, write_history_single_line, DetectorPriority, Game,
    History, ProcessingState,
};
use crate::ghost::DetectorScratch;
use crate::ocr_support::{OcrCache, OcrSession};
use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

struct FrameProcessingResult {
    log: Vec<u8>,
    annotated_frame: Option<Vec<u8>>,
    history: History,
}

struct FrameJob {
    sequence: u64,
    current_frame: u64,
    frame: Vec<u8>,
}

#[derive(Default)]
struct FrameSelection {
    input_frames_seen: u64,
    retained_indices: VecDeque<u64>,
}

struct FrameDecoder {
    pipeline: gst::Pipeline,
    sink: gst_app::AppSink,
    selection: Arc<Mutex<FrameSelection>>,
    frame_size: usize,
}

impl FrameDecoder {
    fn new(input_uri: &str, skip: usize, frame_size: usize) -> Result<Self, String> {
        gst::init().map_err(|error| format!("failed to initialize GStreamer: {error}"))?;

        let pipeline = gst::Pipeline::new();
        let source = make_element("uridecodebin")?;
        source.set_property("uri", input_uri);
        let queue = make_element("queue")?;
        let drop_point = make_element("identity")?;
        let convert = make_element("videoconvert")?;
        let caps_filter = make_element("capsfilter")?;
        let sink = gst_app::AppSink::builder()
            .sync(false)
            .max_buffers(4)
            .build();
        let sink_element = sink.clone().upcast::<gst::Element>();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGBA")
            .build();
        caps_filter.set_property("caps", &caps);

        pipeline
            .add_many([
                &source,
                &queue,
                &drop_point,
                &convert,
                &caps_filter,
                &sink_element,
            ])
            .map_err(|error| format!("failed to build GStreamer decoder: {error}"))?;
        gst::Element::link_many([&queue, &drop_point, &convert, &caps_filter, &sink_element])
            .map_err(|error| format!("failed to link GStreamer decoder: {error}"))?;

        let queue_sink = queue
            .static_pad("sink")
            .ok_or_else(|| "GStreamer decoder queue has no sink pad".to_string())?;
        source.connect_pad_added(move |_source, pad| {
            let caps = pad.current_caps().unwrap_or_else(|| pad.query_caps(None));
            let is_video = caps
                .structure(0)
                .is_some_and(|structure| structure.name().starts_with("video/"));
            if is_video && !queue_sink.is_linked() {
                if let Err(error) = pad.link(&queue_sink) {
                    eprintln!("failed to link decoded video stream: {error}");
                }
            }
        });

        let selection = Arc::new(Mutex::new(FrameSelection::default()));
        let probe_selection = Arc::clone(&selection);
        let probe = drop_point
            .static_pad("src")
            .ok_or_else(|| "GStreamer frame drop point has no source pad".to_string())?
            .add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                let Some(_buffer) = info.buffer() else {
                    return gst::PadProbeReturn::Ok;
                };
                let mut selection = probe_selection
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let frame_index = selection.input_frames_seen;
                selection.input_frames_seen = selection.input_frames_seen.saturating_add(1);
                if frame_should_be_kept(frame_index, skip) {
                    selection.retained_indices.push_back(frame_index);
                    gst::PadProbeReturn::Ok
                } else {
                    gst::PadProbeReturn::Drop
                }
            })
            .ok_or_else(|| "failed to install GStreamer frame drop probe".to_string())?;
        let _ = probe;

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|error| format!("failed to start GStreamer decoder: {error}"))?;

        Ok(Self {
            pipeline,
            sink,
            selection,
            frame_size,
        })
    }

    fn next_frame(&mut self, mut frame: Vec<u8>) -> Result<Option<(u64, Vec<u8>)>, String> {
        let Some(sample) = self.sink.try_pull_sample(gst::ClockTime::NONE) else {
            if let Some(error) = self.pipeline_error() {
                return Err(error);
            }
            if self.sink.is_eos() {
                return Ok(None);
            }
            return Err("GStreamer decoder stopped before end of stream".to_string());
        };
        let buffer = sample
            .buffer()
            .ok_or_else(|| "GStreamer decoder returned a sample without a buffer".to_string())?;
        let mapped = buffer
            .map_readable()
            .map_err(|error| format!("failed to read decoded video frame: {error}"))?;
        if mapped.as_slice().len() != self.frame_size {
            return Err(format!(
                "GStreamer returned a video frame with {} bytes; expected {}",
                mapped.as_slice().len(),
                self.frame_size
            ));
        }
        frame.copy_from_slice(mapped.as_slice());
        let frame_index = self
            .selection
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .retained_indices
            .pop_front()
            .ok_or_else(|| "GStreamer returned a frame without a source index".to_string())?;
        Ok(Some((frame_index, frame)))
    }

    fn input_frames_seen(&self) -> u64 {
        self.selection
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .input_frames_seen
    }

    fn finish(&self) -> Result<(), String> {
        if let Some(error) = self.pipeline_error() {
            return Err(error);
        }
        self.pipeline
            .set_state(gst::State::Null)
            .map(|_| ())
            .map_err(|error| format!("failed to stop GStreamer decoder: {error}"))
    }

    fn pipeline_error(&self) -> Option<String> {
        let bus = self.pipeline.bus()?;
        let message = bus.pop_filtered(&[gst::MessageType::Error])?;
        match message.view() {
            gst::MessageView::Error(error) => Some(format!(
                "GStreamer decoder failed: {} ({:?})",
                error.error(),
                error.debug()
            )),
            _ => None,
        }
    }
}

impl Drop for FrameDecoder {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

fn make_element(factory: &str) -> Result<gst::Element, String> {
    gst::ElementFactory::make(factory)
        .build()
        .map_err(|error| format!("failed to create GStreamer {factory}: {error}"))
}

fn frame_stride(skip: usize) -> u64 {
    u64::try_from(skip).unwrap_or(u64::MAX).saturating_add(1)
}

pub fn process_files(
    input: &str,
    output: &str,
    video_output: Option<&str>,
    game: Game,
    parallel_count: usize,
    skip: usize,
    dedupe_seconds: u64,
    check_story_line_thickness: bool,
    use_tesseract_library: bool,
    verbose: bool,
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
    let mut decoder = FrameDecoder::new(&input_uri, skip, frame_size)?;
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
    let mut state = ProcessingState {
        game,
        current_frame: 0,
        frame_width: width,
        frame_height: height,
        frame_size,
        frame_rate_num,
        frame_rate_den,
        check_story_line_thickness,
        verbose,
    };
    let mut history = History::new();
    let ocr_cache = OcrCache::default();
    let detector_priority = DetectorPriority::default();
    thread::scope(|scope| {
        let (job_sender, job_receiver) = mpsc::sync_channel::<FrameJob>(parallel_count);
        let job_receiver = Arc::new(Mutex::new(job_receiver));
        let (result_sender, result_receiver) =
            mpsc::channel::<(u64, Vec<u8>, Result<FrameProcessingResult, String>)>();
        let mut worker_handles = Vec::with_capacity(parallel_count);
        for _ in 0..parallel_count {
            let job_receiver = Arc::clone(&job_receiver);
            let result_sender = result_sender.clone();
            let cache = &ocr_cache;
            let detector_priority = &detector_priority;
            let frame_width = state.frame_width;
            let frame_height = state.frame_height;
            let frame_size = state.frame_size;
            let frame_rate_num = state.frame_rate_num;
            let frame_rate_den = state.frame_rate_den;
            let check_story_line_thickness = state.check_story_line_thickness;
            let game = state.game;
            worker_handles.push(scope.spawn(move || {
                let mut ocr_session = OcrSession::new(use_tesseract_library, cache);
                let mut detector_scratch = DetectorScratch::default();
                loop {
                    let job = job_receiver
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .recv();
                    let Ok(job) = job else {
                        break;
                    };
                    let sequence = job.sequence;
                    let current_frame = job.current_frame;
                    let frame = job.frame;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let mut frame_state = ProcessingState {
                            game,
                            current_frame,
                            frame_width,
                            frame_height,
                            frame_size,
                            frame_rate_num,
                            frame_rate_den,
                            check_story_line_thickness,
                            verbose,
                        };
                        let mut log = Vec::new();
                        let mut history = History::new();
                        process_frame(
                            &mut frame_state,
                            &frame,
                            &mut log,
                            video_output.is_some(),
                            &mut ocr_session,
                            &mut detector_scratch,
                            detector_priority,
                            &mut history,
                        )
                        .map(|annotated_frame| FrameProcessingResult {
                            log,
                            annotated_frame,
                            history,
                        })
                        .map_err(|error| {
                            format!(
                                "failed to process video frame {}: {error}",
                                current_frame + 1
                            )
                        })
                    }))
                    .unwrap_or_else(|_| {
                        Err(format!(
                            "frame processing thread panicked on frame {}",
                            current_frame + 1
                        ))
                    });
                    let failed = result.is_err();
                    if result_sender.send((sequence, frame, result)).is_err() || failed {
                        break;
                    }
                }
            }));
        }
        drop(result_sender);

        let processing_result = (|| {
            let mut pending_results = BTreeMap::new();
            let mut available_frame_buffers = Vec::with_capacity(parallel_count);
            let mut next_sequence = 0u64;
            let mut end_of_stream = false;
            loop {
                while !end_of_stream
                    && next_sequence.saturating_sub(state.current_frame) < parallel_count as u64
                {
                    let frame = available_frame_buffers
                        .pop()
                        .unwrap_or_else(|| vec![0; state.frame_size]);
                    let Some((current_frame, frame)) = decoder.next_frame(frame)? else {
                        end_of_stream = true;
                        break;
                    };
                    job_sender
                        .send(FrameJob {
                            sequence: next_sequence,
                            current_frame,
                            frame,
                        })
                        .map_err(|_| "frame processing workers stopped unexpectedly".to_string())?;
                    next_sequence += 1;
                }

                if end_of_stream && next_sequence == state.current_frame {
                    break;
                }

                let (sequence, frame, result) = result_receiver
                    .recv()
                    .map_err(|_| "frame processing workers stopped unexpectedly".to_string())?;
                available_frame_buffers.push(frame);
                pending_results.insert(sequence, result);
                while let Some(result) = pending_results.remove(&state.current_frame) {
                    let result = result?;
                    output_file
                        .write_all(&result.log)
                        .map_err(|error| format!("failed to write frame log: {error}"))?;
                    if let (Some(video_frames), Some(annotated_frame)) =
                        (video_frames.as_mut(), result.annotated_frame)
                    {
                        video_frames.write_all(&annotated_frame).map_err(|error| {
                            format!("failed to write encoded video frame: {error}")
                        })?;
                    }
                    merge_history(
                        &mut history,
                        result.history,
                        u128::from(dedupe_seconds) * 1_000_000_000,
                    );
                    state.current_frame += 1;
                }
            }
            Ok(())
        })();
        drop(job_sender);
        for handle in worker_handles {
            if handle.join().is_err() {
                return Err("frame processing thread panicked".to_string());
            }
        }
        processing_result
    })?;

    let input_frames_seen = decoder.input_frames_seen();
    decoder.finish()?;

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

    if verbose {
        write_history(&history, &mut output_file)
            .map_err(|error| format!("failed to write text history to {output}: {error}"))?;
    }
    write_history_single_line(&history, &mut output_file).map_err(|error| {
        format!("failed to write single-line text history to {output}: {error}")
    })?;

    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let frames_per_second = if elapsed_seconds > 0.0 {
        input_frames_seen as f64 / elapsed_seconds
    } else {
        0.0
    };
    let summary = format!(
        "Processed {} of {} input frames in {:.2} seconds ({:.2} FPS).",
        state.current_frame, input_frames_seen, elapsed_seconds, frames_per_second
    );
    writeln!(output_file, "{summary}")
        .map_err(|error| format!("failed to write processing summary to {output}: {error}"))?;
    println!("{summary}");

    Ok(())
}

fn frame_should_be_kept(frame_index: u64, skip: usize) -> bool {
    frame_index % frame_stride(skip) == 0
}

#[cfg(test)]
mod tests {
    use super::frame_should_be_kept;

    #[test]
    fn keeps_every_n_plus_one_frame_from_the_source() {
        let retained = (0..8)
            .filter(|frame_index| frame_should_be_kept(*frame_index, 2))
            .collect::<Vec<_>>();

        assert_eq!(retained, [0, 3, 6]);
    }

    #[test]
    fn skip_one_keeps_alternating_source_frames() {
        let retained = (0..5)
            .filter(|frame_index| frame_should_be_kept(*frame_index, 1))
            .collect::<Vec<_>>();

        assert_eq!(retained, [0, 2, 4]);
    }

    #[test]
    fn skip_zero_keeps_every_source_frame() {
        assert!((0..8).all(|frame_index| frame_should_be_kept(frame_index, 0)));
    }

    #[test]
    fn keeps_first_frame_when_eof_precedes_the_next_stride() {
        let retained = (0..3)
            .filter(|frame_index| frame_should_be_kept(*frame_index, 4))
            .collect::<Vec<_>>();

        assert_eq!(retained, [0]);
    }
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
