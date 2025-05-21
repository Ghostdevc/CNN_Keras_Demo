'''
problem statement: CIFAR10 dataset, classification of ten different class

plan: CNN
'''
# import libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D # Feature Extraction

from tensorflow.keras.layers import Flatten, Dense, Dropout #classification

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report



# load cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
fig, axes = plt.subplots(1, 5, figsize=(25, 20))
for i in range(5):
    axes[i].imshow(x_train[i])
    axes[i].set_title(class_labels[y_train[i][0]])
    axes[i].axis('off')



# normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# data augmentation
datagen = ImageDataGenerator(
    rotation_range=30, # rotation up to 30 degree
    width_shift_range=0.2, # shift up to 20%
    height_shift_range=0.2, # shift up to 20%
    shear_range=0.2, # shear up to 20%
    zoom_range=0.2, # zoom up to 20%
    horizontal_flip=True
)

datagen.fit(x_train)


x_train.shape


# sequential model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()



# model compile
model.compile(optimizer= RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# model training
history = model.fit(datagen.flow(x_train, y_train, batch_size=512), epochs=2, validation_data=(x_test, y_test))


# test predictions
y_pred = model.predict(x_test) # if y_pred = [0, 1] = 0.8 => 80%
y_pred_class = np.argmax(y_pred, axis=1) # get predicted classes
y_true = np.argmax(y_test, axis=1)

# classification report
report = classification_report(y_true, y_pred_class, target_names=class_labels)
print(report)

plt.figure()
# loss graphs
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()


# accuracy
plt.subplot(1, 2, 2) #
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()


