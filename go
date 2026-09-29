#!/usr/bin/env bash

set -x
#cargo run -- --in ../samples/samp3.mp4 --out notes.txt --vout annotated.mp4
cargo run -- --in ../samples/samp.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/loc1.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/story1.mp4 --out notes.txt --vout annotated.mp4
cat notes.txt
