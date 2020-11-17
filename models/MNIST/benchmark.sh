#!/bin/sh

while getopts ":epochs:lr:batch_size:out:" opt; do
  case $opt in
    epochs) 2="$OPTARG"
    ;;
    lr) 4="$OPTARG"
    ;;
    batch_size) 6="$OPTARG"
    ;;
    out) 8="$OPTARG"
    ;;
    plots) {10}="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Argument epochs is %s\n" "$2"
printf "Argument lr is %s\n" "$4"
printf "Argument batch_size is %s\n" "$6"
printf "Argument out is %s\n" "$8"
printf "Argument plots is %s\n" "${10}"

rootPath=`pwd`
cd ./SwiftML
swift run swift-ml LeNet -p ${rootPath}/../params.json
cd -
mv $HOME/Library/Application\ Support/results.json . 
python3 ./Keras/LeNet.py --epochs $2 --lr $4 --batch_size $6 --out $8
python3 ./Pytorch/LeNet_v2.py --epochs $2 --lr $4 --batch_size $6 --out $8
python3 ./create_plots.py --input $8 --output ${10}
