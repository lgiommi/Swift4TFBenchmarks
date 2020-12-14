#!/bin/sh

while getopts ":params:" opt; do
  case $opt in
    params) 2="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Argument params is %s\n" "$2"

cd ./SwiftML
rootPath=`pwd`
swift run swift-ml LeNet -p ${rootPath}/../../../params.json
cd -
python3 ./Keras/LeNet.py --params $2
python3 ./Pytorch/LeNet_v2.py --params $2
python3 ./create_plots.py --params $2