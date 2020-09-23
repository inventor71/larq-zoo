import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import larq_zoo as lqz
from urllib.request import urlopen
from PIL import Image

def preprocess(data):
    img = lqz.preprocess_input(data["image"])
    label = tf.one_hot(data["label"], 1000)
    return img, label 

data_dir = "/tf/data/tensorflow_datasets/"

dataset = (
    tfds.load("imagenet2012:5.0.0", split=tfds.Split.VALIDATION, data_dir=data_dir)
    .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(128)
    .prefetch(1)
)

model = lqz.literature.RealToBinaryNet()
model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
)

model.evaluate(dataset)
