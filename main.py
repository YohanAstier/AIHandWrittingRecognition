from keras.datasets import mnist
from keras import models
from keras import utils
from keras import layers

class Network():
    pass
    

if __name__ == "__main__":
    #Collect the training dataset
    train, test = mnist.load_data()
    utils.normalize(train[0], axis=1) 
    utils.normalize(test[0], axis=1) 

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    history = model.fit(train[0], train[1], validation_data=test, verbose=1)
