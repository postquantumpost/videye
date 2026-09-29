use std::env;
use std::process;

mod frame_processor;
mod processing;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        eprintln!("Usage: videye --in inputfile --out outputfile --vout video_outputfile");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut video_output = None;

    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;

        match argument.as_str() {
            "--in" => input = Some(value),
            "--out" => output = Some(value),
            "--vout" => video_output = Some(value),
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

    if let (Some(input), Some(output), Some(video_output)) = (&input, &output, &video_output) {
        processing::process_files(input, output, video_output)?;
    }

    Ok(())
}
