#!/bin/sh
cd ./SwiftML
swift run swift-ml LeNet --epochs $1 --learning-rate $2 --batch-size $3 --out $4
cd -
mv $HOME/Library/Application\ Support/results.json . 
python3 ./Keras/LeNet.py --epochs $1 --lr $2 --batch_size $3 --out $4
python3 ./Pytorch/LeNet_v2.py --epochs $1 --lr $2 --batch_size $3 --out $4
