# example of a cnn for image classification
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import time
import json
import sys

import argparse
import textwrap

parser = argparse.ArgumentParser(prog='PROG', formatter_class=argparse.RawDescriptionHelpFormatter,\
    epilog=textwrap.dedent('''\
         Here an example on how to run the script:
         python3 LeNet.py --params $PWD/../params.json
         '''))
parser.add_argument("--params", action="store", dest="params", default='', \
            help="name of the params file")

opts = parser.parse_args()
if not opts.params:
    print('No params file is provided')
    sys.exit(1)
params=opts.params

with open(params) as json_file:
    data = json.load(json_file)

LEARNING_RATE = float(data['lr'])
BATCH_SIZE = int(data['batch_size'])
N_EPOCHS = int(data['epochs'])
OUT = str(data['out'])
PLOTS = str(data["plots"])

print(f"learning rate: {LEARNING_RATE}\tbatch size: {BATCH_SIZE} \tepochs: {N_EPOCHS}\toutput file: {OUT}\tplots file: {PLOTS}")

# load dataset
(x_train, y_train), (x_test, y_test) = load_data()
# reshape data to have a single channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# determine the shape of the input images
in_shape = x_train.shape[1:]
# determine the number of classes
n_classes = len(unique(y_train))
print(in_shape, n_classes)
# normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# define model
model = Sequential()
model.add(Conv2D(6, (5,5), activation='relu', padding='same', input_shape=in_shape))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (5,5), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
# define loss and optimizer
opt = SGD(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
time0=time.time()
history=model.fit(x_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test,y_test), verbose=1)
trainTime=time.time()-time0
results=history.history
results['trainTime']=trainTime
print(f"Training time: {results['trainTime']}")

with open(OUT) as json_file:
    data = json.load(json_file)
data["Keras"]=results
with open(OUT, 'w') as outfile:
    json.dump(data, outfile)

# evaluate the model
#loss, acc = model.evaluate(x_test, y_test, verbose=0)
#print('Accuracy: %.3f' % acc)
# make a prediction
#image = x_train[0]
#yhat = model.predict(asarray([image]))
#print('Predicted: class=%d' % argmax(yhat))
