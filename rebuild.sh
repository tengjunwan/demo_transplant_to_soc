#!/bin/sh
if [ -d "build" ]; then
    echo "building folder already exists. deleting it ..."
    rm -rf build
fi
echo "delete ./output"
rm -rf output

# create a new build folder
echo "create build folder"
mkdir build

cd build

echo "running cmake and make ..."
cmake ..
make

echo "build completed"
