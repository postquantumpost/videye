# videye
A video processing program to extract annotations.

Run it with input and text output paths. `--game gt` selects Ghost of Tsushima (the default), while `--game er` selects Elden Ring. Add `--vout` to also write an annotated video. The in-process Tesseract library backend is used by default; pass `--use-tesseract-library false` to use the subprocess backend:

```sh
cargo run -- --in inputfile --out outputfile [--game gt|er] [--vout annotated.mp4] [--parallel-count 16] [--skip n] [--dedupe seconds] [--use-tesseract-library true|false]
```

Frame processing uses sixteen worker threads by default; set `--parallel-count` to change the number of concurrent workers. `--skip n` processes one frame then skips the next `n` frames; it defaults to `0`. `--dedupe seconds` sets the detection deduplication window and defaults to `120` seconds. Frame numbers and timestamps remain based on the original input, and skipped frames are included in the reported FPS. The video output supports MP4, Matroska (`.mkv`), and WebM (`.webm`) files.

Elden Ring mode also has a `place2` detector for region-name banners such as `samples/limgrave.png`, where the text sits on a translucent dark band whose darkness depends on the scene. At sixteen columns it looks for the band's bottom edge (dark band, gray underline, brighter scene), then the top edge (brighter scene above the dark band), then near-white text strokes across the middle of the lower-case letters. Frames where the text is still fading in or out are skipped. The crop sent to OCR extends below the underline so descenders like the `g` are kept; scenery below the band that is not connected to the letters is removed. Matches are reported as `place2: <name>`. A `place3` detector uses the same approach for the smaller location banners that sit higher in the frame, such as `samples/elleh_p.png`; its matches are reported as `place3: <name>`. `place`, `place2`, and `place3` are separate history types, and duplicates are only suppressed within the same type. With the local sample available, `cargo test place2 -- --include-ignored` also checks `samples/limgrave.png` using both OCR backends.

Elden Ring mode emits `grace: <name>` when the site-of-grace menu is open. The icon box at the top left has soft edges, so besides a coarse grid search the detector also tries the exact positions where the box edges appear in brightness profiles. The OCR region ends at the first gap wider than the label height after the name, so bright scenery to the right is not read as text.

Elden Ring mode emits `newitem: <item name>` after detecting the three gray separators in the lower-center item popup and confirming `NEW` with OCR. Item names have a separate UTF-8 character filter supporting alphanumerics, accented characters, commas, dashes, apostrophes, and square brackets. Existing match character filters are unchanged. With the local sample available, run `cargo test newitem -- --include-ignored` to also check `samples/lizard.png` using both OCR backends without processing the sample video.

The default in-process backend requires the Tesseract and Leptonica development libraries, Clang/libclang, GStreamer development libraries, and English language data. On Ubuntu or Debian, install them with `sudo apt install libtesseract-dev libleptonica-dev clang libclang-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev tesseract-ocr-eng`. A native Tesseract crash in library mode can terminate Videye; use the subprocess backend to contain such crashes to the OCR child process.

With `--skip`, GStreamer drops skipped decoded frames before RGBA conversion, reducing conversion work and frame-buffer traffic while preserving source frame numbers and timestamps. Inter-frame codecs may still need to decode skipped frames to reach later frames.
