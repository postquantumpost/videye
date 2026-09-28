use std::env;
use std::process;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        eprintln!("Usage: videye [--in inputfile] [--out outputfile]");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let mut input = None;
    let mut output = None;

    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;

        match argument.as_str() {
            "--in" => input = Some(value),
            "--out" => output = Some(value),
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    if let Some(input) = input {
        println!("Input: {input}");
    }
    if let Some(output) = output {
        println!("Output: {output}");
    }

    Ok(())
}
