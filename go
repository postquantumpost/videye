#!/usr/bin/env bash

set -x
#cargo run -- --in ../samples/samp3.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/samp.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/loc1.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/story1.mp4 --out notes.txt --vout annotated.mp4
#cargo run -- --in ../samples/ghost0.mp4 --out ghost0_notes.txt --vout ghost0_annotated.mp4
#cargo run -- --in ../samples/ghost1.mp4 --out ghost1_notes.txt --vout ghost1_annotated.mp4
#cargo run -- --in ../samples/gstoryv1.mp4 --out notes.txt --parallel-count 20
#cargo run -- --in ../samples/ghost0.mp4 --out notes.txt --parallel-count 30
#cargo run -- --in ../samples/story1.mp4 --out notes.txt --parallel-count 30
#f=/media/jph/slow_bad_3/2026/09/2026090218291302_ghost_80.mp4
f=../samples/samp.mp4
#cargo run -- --in $f --out notes.txt --parallel-count 16 --use-tesseract-library false
#cargo run -- --in $f --out notes.txt --parallel-count 32
#cargo run -- --in $f --out notes.txt --parallel-count 16 --skip 59
#cat notes.txt
cargo run -- --in $f --out notes.txt --parallel-count 16
