!sudo update-alternatives --config python3
!sudo apt install python3-pip

import os
import tensorflow as tf
import numpy as np
import random
import math

class Simulator():
    def __init__(self, type) -> None:
        if type == 'D':
            # deuteranope
            self.color_matrix = tf.convert_to_tensor([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
        elif type == 'P':
            # protanope
            self.color_matrix = tf.convert_to_tensor([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
        elif type == 'T':
            # tritanope
            self.color_matrix = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
        else:
            raise("ERROR: invalid type passed into Simulator class (only accepts 'D', 'P', or 'T')")

        self.rgb2lms = tf.convert_to_tensor([[17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]])

    def simulate_image(self, image):
        # passes an image through the color-blindness simulator
        
        inverted_rgb2lms = tf.linalg.inv(self.rgb2lms)

        product1 = tf.matmul(inverted_rgb2lms, self.color_matrix)
        product2 = tf.matmul(product1, self.rgb2lms)

        original_image_shape = image.shape
        
        simulated_image = tf.transpose(tf.matmul(product2, tf.reshape(tf.transpose(image, perm=[2, 0, 1]), (image.shape[2], image.shape[0] * image.shape[1]))), perm=[1, 0])

        return tf.reshape(simulated_image, original_image_shape)
      
      
  
import os
import tensorflow as tf
import numpy as np
import random
import math

class Corrector(tf.keras.Model):
    def __init__(self, batch_size, type) -> None:
        super(Corrector, self).__init__()

        self.learning_rate = 0.001
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.batch_size = batch_size

        self.linear_corrector = tf.convert_to_tensor([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])

        self.corrector = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same'),
            tf.keras.layers.Conv2D(16, 3, padding='same'),
            tf.keras.layers.Conv2D(3, 3, padding='same'),
        ])

        self.simulator = Simulator(type)

        pass

    def linear_correct(self, image):
        original_size = image.shape

        corrected = tf.matmul(self.linear_corrector, tf.reshape(tf.transpose(image, perm=[2, 0, 1]), (original_size[2], original_size[0] * original_size[1])))

        return tf.reshape(tf.transpose(corrected, perm=[1, 0]), original_size)

    def call(self, inputs):
        # TODO: Write forward-pass logic
        
        # linear corrector layer
        simulator_difference = inputs - tf.map_fn(self.simulator.simulate_image, inputs)
        linear_output = inputs + tf.map_fn(self.linear_correct, simulator_difference)

        # convolutional layers
        corrector_output = self.corrector(linear_output)

        return corrector_output
    
    import os
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, MaxPooling2D, Dropout, Conv2D, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.applications import EfficientNetB0

class Referee(tf.keras.Model):
    def __init__(self, batch_size) -> None:
        super(Referee, self).__init__()

        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.001
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = batch_size
        self.conv_size = 128
        self.hidden_size = 400

        # TODO: Initialize all trainable parameters

        self.referee = Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(self.conv_size, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(self.hidden_size, activation='relu'),
            Dense(self.hidden_size, activation='relu'),
            Dense(100, activation='softmax')
        ])

    def call(self, inputs):
        # TODO: Write forward-pass logic
        # outputs a 1 by 1 by 20
        referee_output = self.referee(inputs)
        return referee_output
    
    def loss(self, probs, labels): 
        # TODO: Write loss function

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        return tf.reduce_mean(loss)
    import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('tkAgg')

def compute_accuracy(logits, labels):
    correct_predictions = tf.equal(tf.argmax(tf.squeeze(logits), 1), tf.squeeze(labels))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def load_data():
    (d12_data, d12_labels), (d3_data, d3_labels) = tf.keras.datasets.cifar100.load_data()

    d1 = (d12_data[:25000] / 255, d12_labels[:25000], 100)
    d2 = (d12_data[25000:] / 255, d12_labels[25000:], 100)
    d3 = (d3_data / 255, d3_labels, 100)

    return d1, d2, d3

def run_d1(referee, d1):
    # TODO: train referee
    for epoch in range(1):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in range(1):#0, len(d1[0]), referee.batch_size):
            print("Batch " + str(batch_counter))
            batch_counter += 1
            labels = d1[1][batch : batch + referee.batch_size]
            with tf.GradientTape() as tape:
                logits = referee.call(d1[0][batch : batch + referee.batch_size])
                loss = referee.loss(logits, tf.squeeze(labels))
                print("loss: " + str(loss)) 
            accuracy = compute_accuracy(logits, labels)
            print("ACCURACY: " + str(accuracy))
            gradients = tape.gradient(loss, referee.trainable_variables)
            referee.Adam.apply_gradients(zip(gradients, referee.trainable_variables))


def run_d2(corrector, referee, d2):
    # TODO: train corrector and test referee
    for epoch in range(2):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in range(0, len(d2[0]), referee.batch_size):
            print("Batch " + str(batch_counter))
            batch_counter += 1
            labels = d2[1][batch : batch + referee.batch_size]
            with tf.GradientTape() as tape:
                corrected_images = corrector(d2[0][batch : batch + referee.batch_size])
                simulated_corrected_images = tf.map_fn(corrector.simulator.simulate_image, corrected_images)
                logits = referee.call(simulated_corrected_images)
                loss = referee.loss(logits, labels)
                print("loss: " + str(loss)) 
                accuracy = compute_accuracy(logits, labels)
                print("ACCURACY: " + str(accuracy))
            gradients = tape.gradient(loss, corrector.trainable_variables)
            corrector.Adam.apply_gradients(zip(gradients, corrector.trainable_variables))
            

def run_d3(corrector, referee, d3):
    # TODO: test our corrected images vs uncorrected images

    total_acc_corrected = 0
    total_acc_uncorrected = 0
    batch_counter = 0

    for batch in range(0, len(d3[0]), referee.batch_size):
        print("Batch " + str(batch_counter))
        batch_counter += 1
        corrected_images = corrector(d3[0][batch : batch + referee.batch_size])
        uncorrected_images = d3[0][batch : batch + referee.batch_size]
        labels = d3[1][batch : batch + referee.batch_size]
        corrected_pred = referee.call(corrected_images)
        uncorrected_pred = referee.call(uncorrected_images)
        acc_corrected = compute_accuracy(corrected_pred, labels)
        acc_uncorrected = compute_accuracy(uncorrected_pred, labels)
        print("Accuracy with correction: " + str(acc_corrected))
        print("Accuracy without correction: " + str(acc_uncorrected))
        total_acc_corrected += acc_corrected
        total_acc_uncorrected += acc_uncorrected

    return total_acc_corrected / batch_counter, total_acc_uncorrected / batch_counter

def main():

    # 100 as batch_size for now, change later
    batch_size = 500

    corrector_deuteranope = Corrector(batch_size, 'D')
    corrector_protanope = Corrector(batch_size, 'P')
    corrector_tritanope = Corrector(batch_size, 'T')

    referee = Referee(batch_size)

    d1, d2, d3 = load_data()

    # testing that data loaded correctly

    # TODO: Train and test Corrector and Referee models 

    print("STARTING D1")

    # save the model so we don't have to train it again
    run_d1(referee, d1)
    referee.save_weights('../models/referee.tf')
    
    # load weights from saved model
    referee.load_weights('../models/referee.tf')
    #run for one batch to initialize params
    run_d1(referee, d1)


    print("STARTING D2")
    run_d2(corrector_deuteranope, referee, d2)
    corrector_deuteranope.save_weights('../models/corrector_deuteranope.tf')
    run_d2(corrector_protanope, referee, d2)
    corrector_protanope.save_weights('../models/corrector_protanope.tf')
    
    corrector_protanope.load_weights('../models/corrector_protanope.tf')

    run_d2(corrector_tritanope, referee, d2)
    corrector_tritanope.save_weights('../models/corrector_tritanope.tf')
    
    print("STARTING D3")
    accuracy_deuteranope_corrected, accuracy_deuteranope_uncorrected = run_d3(corrector_deuteranope, referee, d3)
    accuracy_protanope_corrected, accuracy_protanope_uncorrected = run_d3(corrector_protanope, referee, d3)
    accuracy_tritanope_corrected, accuracy_tritanope_uncorrected = run_d3(corrector_tritanope, referee, d3)

    print("ACCURACY DEUTERANOPE CORRECTED: " + str(accuracy_deuteranope_corrected))
    print("ACCURACY DEUTERANOPE UNCORRECTED: " + str(accuracy_deuteranope_uncorrected))
    print("ACCURACY PROTANOPE CORRECTED: " + str(accuracy_protanope_corrected))
    print("ACCURACY PROTANOPE UNCORRECTED: " + str(accuracy_protanope_uncorrected))
    print("ACCURACY TRITANOPE CORRECTED: " + str(accuracy_tritanope_corrected))
    print("ACCURACY TRITANOPE UNCORRECTED: " + str(accuracy_tritanope_uncorrected))

    # TODO: we can now use the trained corrector models to visualize some results here
    nc = 7
    nr = 10

    fig = plt.figure()

    image = d3[0][0]
    image_idx = 0

    for i in range(70):
        ax = fig.add_subplot(nr, nc, i+1)

        if (i % 7 == 0):
            image = d3[0][i]
            image_idx = i
            ax.imshow(image, cmap="Greys")
        elif (i % 7 == 1):
            ax.imshow(corrector_deuteranope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 2):
            ax.imshow(corrector_protanope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 3):
            ax.imshow(corrector_tritanope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 4):
            ax.imshow(corrector_deuteranope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])
        elif (i % 7 == 5):
            ax.imshow(corrector_protanope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])
        else:
            ax.imshow(corrector_tritanope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])

    plt.show()

    return

if __name__ == "__main__":
    main()
