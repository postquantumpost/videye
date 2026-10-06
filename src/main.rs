use std::env;
use std::process;

use frame_processor::Game;

mod elden;
mod frame_processor;
mod ghost;
mod ocr_support;
mod processing;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        eprintln!("Usage: videye --in inputfile --out outputfile [--game gt|er] [--vout video_outputfile] [--parallel-count count] [--skip n] [--dedupe seconds] [--check-story-line-thickness true|false] [--use-tesseract-library true|false] [--verbose]");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut video_output = None;
    let mut game = Game::default();
    let mut parallel_count = 16;
    let mut skip = 0;
    let mut dedupe_seconds = 120u64;
    let mut check_story_line_thickness = true;
    let mut use_tesseract_library = true;
    let mut verbose = false;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--verbose" => {
                verbose = true;
            }
            _ => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;

                match argument.as_str() {
                    "--in" => input = Some(value),
                    "--out" => output = Some(value),
                    "--vout" => video_output = Some(value),
                    "--game" => {
                        game = match value.as_str() {
                            "gt" => Game::GhostofTsushima,
                            "er" => Game::EldenRing,
                            _ => return Err(format!("invalid game: {value}; use gt or er")),
                        }
                    }
                    "--parallel-count" => {
                        parallel_count = value
                            .parse::<usize>()
                            .map_err(|_| format!("invalid parallel count: {value}"))?;
                        if parallel_count == 0 {
                            return Err("parallel count must be greater than zero".to_string());
                        }
                    }
                    "--skip" => {
                        skip = value
                            .parse::<usize>()
                            .map_err(|_| format!("invalid skip count: {value}"))?;
                    }
                    "--dedupe" => {
                        dedupe_seconds = value
                            .parse::<u64>()
                            .map_err(|_| format!("invalid dedupe duration: {value}"))?;
                    }
                    "--check-story-line-thickness" => {
                        check_story_line_thickness = match value.as_str() {
                            "true" => true,
                            "false" => false,
                            _ => {
                                return Err(format!("invalid story line thickness check: {value}"))
                            }
                        }
                    }
                    "--use-tesseract-library" => {
                        use_tesseract_library = match value.as_str() {
                            "true" => true,
                            "false" => false,
                            _ => return Err(format!("invalid Tesseract library setting: {value}")),
                        }
                    }
                    _ => return Err(format!("unknown argument: {argument}")),
                }
            }
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
            game,
            parallel_count,
            skip,
            dedupe_seconds,
            check_story_line_thickness,
            use_tesseract_library,
            verbose,
        )?;
    }

    Ok(())
}
