# video link - https://youtu.be/bte8Er0QhDg

#%%

"""The import os statement in Python allows you to access various operating system-dependent functionalities and methods through the os module. It provides a way to interact with the operating system, perform file and directory operations, retrieve environment variables, and more."""
import os 
""" used for computer vision - to laod and processs images """
import cv2
""" important as we are going to work on numpy arrays """
import numpy as np
"""used to visualize the digits """
import matplotlib.pyplot as plt
""" used for the machine learning part """
import tensorflow as tf

#%%
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)
model.save("handwritten.model")
#%%

model = tf.keras.models.load_model("handwritten.model")

loss,accuracy = model.evaluate(x_test, y_test)

print("Loss is : ",loss,"\n","Accuracy is : ",accuracy)



#%%
#hello
model = tf.keras.models.load_model("handwritten.model")

image_number = 1
while os.path.isfile(f"digits/0{image_number}.png"):
    try:
        img = cv2.imread(f"digits/0{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Prediction is : {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error loading")
    finally:
        image_number += 1

#%%

