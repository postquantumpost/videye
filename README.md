# videye
A video processing program to extract annotations.

Run it with input and text output paths. Add `--vout` to also write an annotated video:

```sh
cargo run -- --in inputfile --out outputfile [--vout annotated.mp4] [--parallel-count 30]
```

Frame processing uses thirty worker threads by default; set `--parallel-count` to change the number of concurrent workers. The video output supports MP4, Matroska (`.mkv`), and WebM (`.webm`) files.
