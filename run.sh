#!/bin/sh

# list of folders to reset
TARGET_DIRS="frame resize_frame crop model_input"

for DIR in $TARGET_DIRS; do
    echo "Processing: $DIR"

    if [ -d "$DIR" ]; then
        echo "  Removing existing folder: $DIR"
        rm -rf "$DIR"
    fi

    echo "  Creating folder: $DIR"
    mkdir -p "$DIR"
done

echo "all folders are ready."

./ss928_objectTrack
