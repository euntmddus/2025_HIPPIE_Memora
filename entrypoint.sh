#!/bin/bash
INPUT=$1
OUTPUT=$2

echo "Running HippMapp3r..."
hippmapper seg_hipp -t1 $INPUT -o $OUTPUT
echo "Done."
