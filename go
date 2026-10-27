#!/usr/bin/env bash

set -x

cargo install --path .

#b=~/.cargo/bin/videye

#f=../samples/rest_at_grace.mp4
#f=../samples/place1.mp4
#f=../samples/boss0.mp4
#f=../samples/elden.mp4
#f=../samples/jori.mp4
#f=../samples/jogaku.mp4
#$b --in $f --out notes.txt --parallel-count 16 --game er
#$b --in $f --out notes.txt --parallel-count 16 --game gt

#f=./samples/boss2.mp4
#f=./samples/boss3.mp4
#f=./samples/rest_at_grace.mp4
#f=../samples/place_name.mp4
#f=../samples/limgrave.mp4
f=../samples/elleh.mp4
#f=./samples/elden.mp4
./vebatcher $f
