# videye
A video processing program to extract annotations.

Run it with input and text output paths. Add `--vout` to also write an annotated video. The in-process Tesseract library backend is used by default; pass `--use-tesseract-library false` to use the subprocess backend:

```sh
cargo run -- --in inputfile --out outputfile [--vout annotated.mp4] [--parallel-count 16] [--skip n] [--use-tesseract-library true|false]
```

Frame processing uses sixteen worker threads by default; set `--parallel-count` to change the number of concurrent workers. `--skip n` processes one frame then skips the next `n` frames; it defaults to `0`. Frame numbers and timestamps remain based on the original input, and skipped frames are included in the reported FPS. The video output supports MP4, Matroska (`.mkv`), and WebM (`.webm`) files.

The default in-process backend requires the Tesseract and Leptonica development libraries, Clang/libclang, and English language data. On Ubuntu or Debian, install them with `sudo apt install libtesseract-dev libleptonica-dev clang libclang-dev tesseract-ocr-eng`. A native Tesseract crash in library mode can terminate Videye; use the subprocess backend to contain such crashes to the OCR child process.
