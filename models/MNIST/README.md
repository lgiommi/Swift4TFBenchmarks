## Models in different languages
Different version of the same ML model can be written using different languages: Keras, Swift and Pytorch.
In particular we chose the LeNet network as example.

### LeNet structure
In the following the structure of the model written in Keras is shown:
```
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', padding='same', input_shape=in_shape))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=SGD(learning_rate=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
###Benchmarks
The benchmark.sh bash script allows to run consecutively swift, keras and pytorch models producing a json file
as output with information about the values of the metrics obtained during the training process and the time spent
for different operations like the training phase. An example of usage is:
```
source benchmark.sh 2 0.1 128 results.json
