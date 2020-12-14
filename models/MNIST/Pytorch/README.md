Here an example of usage of the Lenet_v2.py example:
```
python LeNet.py --help
usage: PROG [-h] [--params PARAMS]

optional arguments:
  -h, --help       show this help message and exit
  --params PARAMS  name of the params file

Here an example on how to run the script:
python3 LeNet_v2.py --params $PWD/../params.json
```
An example of the params file is the following:
```
{
    "epochs": 3, 
    "batch_size": 128, 
    "lr":0.1, 
    "out":"output.json",
    "plots":"plots.pdf"
}
```
