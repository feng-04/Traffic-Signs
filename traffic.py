import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # load image data
    images, labels = load_data(sys.argv[1])



    # Split data into training and testing set
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    
    model = get_model()
    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = []
    labels = []
    # get all folders inside the folder
    folders = os.listdir(data_dir)
    for folder in folders:
        files = os.listdir(os.path.join(data_dir, folder))
        for file in files:
            # get all files inside each one of the folder
            filepath = os.path.join(data_dir, folder, file)

            ima = cv2.imread(filepath)
            if ima is not None:
                (height, width) = ima.shape[:2]
                if (height, width) != (IMG_HEIGHT, IMG_WIDTH):
                    ima = cv2.resize(ima, (IMG_HEIGHT, IMG_WIDTH))
                images.append(ima)
                labels.append(int(folder))
            else:
                continue

                # resize image
    return(images, labels)








def get_model():
    model = tf.keras.models.Sequential([
        # convolution, 32 filters, each is a 3x3 kernal
        # the value of the kernal will be learnt through backpropagation and chain rule
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        ),

        # pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



if __name__ == "__main__":
    main()