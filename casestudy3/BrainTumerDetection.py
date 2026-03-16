# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

# parameters
img_size = 224
batch_size = 32
epochs = 10


train_path = "Brain-Tumor-Classification-DataSet-master/Training"
test_path = "Brain-Tumor-Classification-DataSet-master/Testing"

# data preprocessing
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

print(train_data.class_indices)

# build cnn model
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation="relu"))

model.add(layers.Dense(4,activation="softmax"))

model.summary()




# compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# train model
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data
)

# evaluate model
loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)

# plot accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train","Validation"])
plt.show()


# predict tumor type
img_path = "test_image.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = img_array/255.0
img_array = np.expand_dims(img_array,axis=0)

prediction = model.predict(img_array)

classes = ["Glioma Tumor","Meningioma Tumor","No Tumor","Pituitary Tumor"]

result = classes[np.argmax(prediction)]

print("Prediction:", result)

if result == "No Tumor":
    print("Brain is Normal")
else:
    print("Tumor Detected:", result)
