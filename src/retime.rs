use crate::frame_processor::history_timestamp_is_real;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn retime_file(filename: &str) -> Result<(), String> {
    let path = fs::canonicalize(filename)
        .map_err(|error| format!("failed to resolve retime file {filename}: {error}"))?;
    let input = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read retime file {filename}: {error}"))?;
    let output =
        retime_text(&input).map_err(|error| format!("failed to retime {filename}: {error}"))?;
    let permissions = fs::metadata(&path)
        .map_err(|error| format!("failed to read permissions for {filename}: {error}"))?
        .permissions();
    let parent = path
        .parent()
        .ok_or_else(|| format!("retime file has no parent directory: {filename}"))?;
    let temporary_directory = create_temporary_directory(parent)
        .map_err(|error| format!("failed to create retime temporary directory: {error}"))?;

    let result = (|| -> io::Result<()> {
        let temporary_file = temporary_directory.join("retimed");
        let mut file = fs::File::create(&temporary_file)?;
        file.write_all(output.as_bytes())?;
        file.set_permissions(permissions)?;
        file.sync_all()?;
        drop(file);
        fs::rename(temporary_file, &path)
    })();
    let cleanup = fs::remove_dir_all(&temporary_directory);
    match (result, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), cleanup) => Err(format!(
            "failed to replace retime file {filename}: {error}{}",
            cleanup
                .err()
                .map(|error| format!("; failed to remove temporary directory: {error}"))
                .unwrap_or_default()
        )),
        (Ok(()), Err(error)) => Err(format!(
            "retimed {filename}, but failed to remove temporary directory: {error}"
        )),
    }
}

fn create_temporary_directory(parent: &Path) -> io::Result<PathBuf> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    loop {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".videye-retime-{}-{id}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

fn retime_text(input: &str) -> Result<String, String> {
    let mut output = String::with_capacity(input.len());
    let mut previous_real_timestamp_ns = None;
    for (index, line) in input.split_inclusive('\n').enumerate() {
        let timestamp_end = line.find(char::is_whitespace).unwrap_or(line.len());
        let timestamp = &line[..timestamp_end];
        if !timestamp.starts_with(|c: char| c.is_ascii_digit())
            || !timestamp.contains([':', '\u{0589}'])
        {
            output.push_str(line);
            continue;
        }
        let seconds = parse_timestamp(timestamp)
            .ok_or_else(|| format!("invalid timestamp on line {}: {timestamp}", index + 1))?;
        let is_real = history_timestamp_is_real(
            u128::from(seconds) * 1_000_000_000,
            &mut previous_real_timestamp_ns,
        );
        let separator = if is_real { ":" } else { "\u{0589}" };
        output.push_str(&timestamp.replace([':', '\u{0589}'], separator));
        output.push_str(&line[timestamp_end..]);
    }
    Ok(output)
}

fn parse_timestamp(timestamp: &str) -> Option<u64> {
    let fields: Vec<_> = timestamp.split([':', '\u{0589}']).collect();
    if !(2..=3).contains(&fields.len()) {
        return None;
    }
    let mut seconds = 0u64;
    for (index, field) in fields.iter().enumerate() {
        if field.is_empty() || !field.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let value = field.parse::<u64>().ok()?;
        if index > 0 && value >= 60 {
            return None;
        }
        seconds = seconds.checked_mul(60)?.checked_add(value)?;
    }
    Some(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recalculates_spacing_after_edits() {
        let input = "0:00 begin\n0:09 A\n0\u{0589}10 B\n0:19 C\n0\u{0589}20 D\n\
                     1\u{0589}01\u{0589}01 E\n1:01:05 F\n1\u{0589}01\u{0589}11 G\n";
        let expected = "0:00 begin\n0\u{0589}09 A\n0:10 B\n0\u{0589}19 C\n0:20 D\n\
                        1:01:01 E\n1\u{0589}01\u{0589}05 F\n1:01:11 G\n";
        assert_eq!(retime_text(input).unwrap(), expected);
        assert_eq!(retime_text(expected).unwrap(), expected);
    }

    #[test]
    fn preserves_text_frames_metadata_and_line_endings() {
        let input =
            "Process input to output.\r\n\r\n0\u{0589}02       Grace \u{2075}\u{2076} label\r\n\
                     0:11         Item: name \u{2070}\r\nProcessed 3 frames.\r\n0:12 Last";
        let expected = "Process input to output.\r\n\r\n0:02       Grace \u{2075}\u{2076} label\r\n\
                        0\u{0589}11         Item: name \u{2070}\r\nProcessed 3 frames.\r\n0:12 Last";
        assert_eq!(retime_text(input).unwrap(), expected);
        assert_eq!(retime_text("").unwrap(), "");
    }

    #[test]
    fn handles_equal_and_backwards_times_like_history_writer() {
        assert_eq!(
            retime_text("0:20 A\n0:20 B\n0:05 C\n0:30 D\n").unwrap(),
            "0:20 A\n0\u{0589}20 B\n0\u{0589}05 C\n0:30 D\n"
        );
    }

    #[test]
    fn rejects_invalid_and_overflowing_timestamps() {
        for timestamp in [
            "0:60",
            "1:60:00",
            "0:x",
            "0:",
            "1:2:3:4",
            "18446744073709551615:00",
            "99999999999999999999999:00",
        ] {
            assert!(
                retime_text(&format!("header\n{timestamp} Bad\n"))
                    .unwrap_err()
                    .contains("line 2"),
                "{timestamp}"
            );
        }
    }

    #[test]
    fn rewrites_file_and_leaves_invalid_file_untouched() {
        let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp");
        fs::create_dir_all(&parent).unwrap();
        let directory = create_temporary_directory(&parent).unwrap();
        let path = directory.join("history.txt");
        fs::write(&path, "0:00 begin\n0:05 Edited\n").unwrap();
        retime_file(path.to_str().unwrap()).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "0:00 begin\n0\u{0589}05 Edited\n"
        );
        let invalid = "0:00 begin\n0:99 Invalid\n";
        fs::write(&path, invalid).unwrap();
        assert!(retime_file(path.to_str().unwrap()).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), invalid);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }
}
