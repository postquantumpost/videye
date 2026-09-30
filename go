#!/usr/bin/env bash

set -x
#cargo run -- --in ../samples/samp3.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/samp.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/loc1.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/story1.mp4 --out notes.txt --vout annotated.mp4
cargo run -- --in ../samples/ghost0.mp4 --out ghost0_notes.txt --vout ghost0_annotated.mp4
cargo run -- --in ../samples/ghost1.mp4 --out ghost1_notes.txt --vout ghost1_annotated.mp4
cat notes.txt
