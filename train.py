import tensorflow as tf
import cProfile
import matplotlib.pyplot as plt
from tensorflow import keras
from src.model import get_model
from src.dataset import read_data


initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

def run(model):

    best_f1 = 0
    df = read_data('./data')


    muTuple = model

    train_loader = tf.data.Dataset.from_tensor_slices((df[0], df[1]))
    validation_loader = tf.data.Dataset.from_tensor_slices((df[2], df[3]))

    batch_size = 2

    train_dataset = (
        train_loader.shuffle(len(df[0]))
            .batch(batch_size)
            .prefetch(2)
        )

    validation_dataset = (
            validation_loader.shuffle(len(df[2]))
            .batch(batch_size)
            .prefetch(2)
        )


    data = train_dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"]
    )

        
    model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            batch_size=20
        )

if __name__ == '__main__':
    
    model = get_model()
    run(model)