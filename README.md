# videye
A video processing program to extract annotations.

Run it with input, text output, and annotated video output paths:

```sh
cargo run -- --in inputfile --out outputfile --vout annotated.mp4
```

Processing runs only when all three options are provided. The video output supports MP4, Matroska (`.mkv`), and WebM (`.webm`) files.
