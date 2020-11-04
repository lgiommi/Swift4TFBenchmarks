## Models in different languages
In this folder there are different version of the same ML model written in different languages: Keras, Swift and Pytorch.
In particular we chose the LeNet network as example.

### LeNet structure
The structure of the model is the following (here we show the Keras one):
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
