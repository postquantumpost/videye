#!/usr/bin/env bash

set -x
cargo run -- --in ../samples/ghost0.mp4 --out notes.txt
cat notes.txt
