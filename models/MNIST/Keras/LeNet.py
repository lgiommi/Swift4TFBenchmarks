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

import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument("--epochs", action="store", dest="epochs", default=None, \
            help="number of epochs used to train the model")
parser.add_argument("--lr", action="store", dest="lr", default=None, \
            help="learning rate")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=None, \
            help="batch size")

opts = parser.parse_args()
print(f"learning rate: {opts.lr}\tbatch size: {opts.batch_size}\tepochs: {opts.epochs}")

LEARNING_RATE = float(opts.lr)
BATCH_SIZE = int(opts.batch_size)
N_EPOCHS = int(opts.epochs)

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
print(f"Time for fitting: {time.time()-time0}")

results={}
results['keras']={}
for key in list(history.history.keys()):
    results['keras'][key]=history.history[key]
print (results['keras'])

with open('../results.txt', 'w') as outfile:
    json.dump(results, outfile)

# evaluate the model
#loss, acc = model.evaluate(x_test, y_test, verbose=0)
#print('Accuracy: %.3f' % acc)
# make a prediction
#image = x_train[0]
#yhat = model.predict(asarray([image]))
#print('Predicted: class=%d' % argmax(yhat))
