use std::env;
use std::process;

mod frame_processor;
mod processing;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        eprintln!("Usage: videye --in inputfile --out outputfile [--vout video_outputfile] [--parallel-count count] [--check-story-line-thickness true|false]");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut video_output = None;
    let mut parallel_count = 4;
    let mut check_story_line_thickness = true;

    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;

        match argument.as_str() {
            "--in" => input = Some(value),
            "--out" => output = Some(value),
            "--vout" => video_output = Some(value),
            "--parallel-count" => {
                parallel_count = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid parallel count: {value}"))?;
                if parallel_count == 0 {
                    return Err("parallel count must be greater than zero".to_string());
                }
            }
            "--check-story-line-thickness" => {
                check_story_line_thickness = match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(format!("invalid story line thickness check: {value}")),
                }
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    if let Some(input) = &input {
        println!("Input: {input}");
    }
    if let Some(output) = &output {
        println!("Output: {output}");
    }
    if let Some(video_output) = &video_output {
        println!("Video output: {video_output}");
    }

    if let (Some(input), Some(output)) = (&input, &output) {
        processing::process_files(
            input,
            output,
            video_output.as_deref(),
            parallel_count,
            check_story_line_thickness,
        )?;
    }

    Ok(())
}
