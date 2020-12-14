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
### Benchmarks
The benchmark.sh bash takes as argument a json file with parameters for the ML models and it allows to run consecutively
swift, keras and pytorch models producing a json file as output. This file contains information about the score metrics 
obtained during the training process and the time spent for the training phase. In the end a pdf file with plots is produced.
An example of usage is:
```
source benchmark.sh -p params.json
```
An example of the params.json file is:
```
{
    "epochs": 3, 
    "batch_size": 128, 
    "lr":0.1, 
    "out":"output.json",
    "plots":"plots.pdf"
}
