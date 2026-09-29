use std::fs::File;
use std::io::{self, Write};
use std::process::{Command, Stdio};

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

    let text = recognize_text(state, frame)?;
    writeln!(output_file, "OCR:")?;
    writeln!(output_file, "{}", text.trim_end())
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