
+#!/usr/bin/env bash

SRC_FOLDER=/home/marcus/violence-detection/data
OUT_FOLDER=/media/marcus/violence-detection/frame
NUM_WORKER=11

echo "Start to extract frames and optical flow from videos in folder: ${SRC_FOLDER}"
python /home/marcus/violence-detection/extract_frame.py ${SRC_FOLDER} ${OUT_FOLDER} --num_worker ${NUM_WORKER}
