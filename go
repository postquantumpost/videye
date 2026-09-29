#!/usr/bin/env bash

set -x
cargo run -- --in ../samples/samp3.mp4 --out notes.txt
cat notes.txt
