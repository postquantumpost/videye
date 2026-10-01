# videye
A video processing program to extract annotations.

Run it with input and text output paths. Add `--vout` to also write an annotated video. Tesseract runs as a subprocess by default; `--use-tesseract-library true` selects the in-process backend:

```sh
cargo run -- --in inputfile --out outputfile [--vout annotated.mp4] [--parallel-count 30] [--use-tesseract-library true|false]
```

Frame processing uses thirty worker threads by default; set `--parallel-count` to change the number of concurrent workers. The video output supports MP4, Matroska (`.mkv`), and WebM (`.webm`) files.

The in-process backend requires the Tesseract and Leptonica development libraries, Clang/libclang, and English language data. On Ubuntu or Debian, install them with `sudo apt install libtesseract-dev libleptonica-dev clang libclang-dev tesseract-ocr-eng`. A native Tesseract crash in library mode can terminate Videye; subprocess mode contains such crashes to the OCR child process.
