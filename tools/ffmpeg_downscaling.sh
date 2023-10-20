#!/bin/bash

input_folder="/home/adam/Projects/Garbage_Classifier_dataset_squares"
output_folder="/home/adam/Projects/Garbage_Classifier_dataset_resized"
width=512
height=512

for input_file in "$input_folder"/*; do
    if [ -f "$input_file" ]; then
        echo "Processing $input_file"
        filename=$(basename "$input_file")
        output_file="$output_folder/${filename%.*}.png"
        ffmpeg -i "$input_file" -vf "scale=$width:$height" "$output_file"
    fi
done
