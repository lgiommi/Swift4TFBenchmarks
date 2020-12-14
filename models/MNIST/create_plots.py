import argparse
import textwrap
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='PROG', formatter_class=argparse.RawDescriptionHelpFormatter,\
    epilog=textwrap.dedent('''\
         Here an example on how to run the script:
         python3 create_plots.py --input_file results.json --out plots.pdf
         '''))
parser.add_argument("--params", action="store", dest="params", default='', \
            help="name of the params file with absolute path")

opts = parser.parse_args()
params=opts.params
with open(params) as json_file:
    data = json.load(json_file)

INPUT = str(data["out"])
OUTPUT = str(data["plots"])

with open(INPUT) as json_file:
    data = json.load(json_file)

epochs=epochs=len(data['Swift']['loss'])
x_axis=[item for item in range(1,epochs+1)]
plt.figure(figsize=(12, 5))
plt.subplots_adjust(top=1.6)
plt.subplot(221)

plt.plot(x_axis,data['Swift']['loss'], label='Swift')
plt.plot(x_axis,data['Keras']['loss'], label='Keras')
plt.plot(x_axis,data['Pytorch']['loss'], label='Pytorch')
plt.xticks(x_axis,x_axis)
plt.ylabel('loss', weight='bold')
plt.xlabel('# of epochs', weight='bold')
plt.legend()
plt.subplot(222)

plt.plot(x_axis,data['Swift']['accuracy'], label='Swift')
plt.plot(x_axis,data['Keras']['accuracy'], label='Keras')
plt.plot(x_axis,data['Pytorch']['accuracy'], label='Pytorch')
plt.xticks(x_axis,x_axis)
plt.ylabel('Accuracy', weight='bold')
plt.xlabel('# of epochs', weight='bold')
plt.legend()
plt.subplot(223)

plt.bar(['Swift','Keras','Pytorch'],[data['Swift']['trainTime'],data['Keras']['trainTime'],data['Pytorch']['trainTime']])
plt.ylabel('Time for training (s)', weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT)