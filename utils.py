import numpy as np
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import random


def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def decode_keys(data):
    decoded = []
    for i in data.keys():
        decoded.append(i.decode("utf-8"))
    return dict(zip(decoded, data.values()))


def prepare_dict(data):
    from copy import copy
    data_copy = copy(data)
    processed_imgs = []
    for i in data_copy["data"]:  # bruh
        processed_imgs.append(np.reshape(i, (3, 32, 32)).transpose(1, 2, 0))
    data_copy["data"] = processed_imgs
    data_copy["label"] = data_copy["fine_labels"]
    del data_copy["filenames"]
    del data_copy["batch_label"]
    del data_copy["coarse_labels"]
    del data_copy["fine_labels"]
    return data_copy


def visualize_data(train_images, train_labels, class_names):

    random_idx = random.sample(list(range(len(train_labels))), 9)
    plt.figure(figsize=(9, 9))
    for i in range(9):
        subp = int("33" + str(i + 1))
        plt.subplot(subp)
        class_ = train_labels[random_idx[i]]
        title = "Class: " + str(class_) + ", Label:" + class_names[class_]
        plt.title(title)
        plt.axis("off")
        plt.grid(False)
        plt.imshow(train_images[random_idx[i]])


def run_experiment(model, train_ds, test_ds, config=None):
    if config is None:
        config = {
            "learning rate": 0.001,
            "weight decay": 0.0001,
            "batch size": 256,
            "epochs": 100
        }
    optimizer = tfa.optimizers.AdamW(
        learning_rate=config["learning rate"], weight_decay=config["weight decay"]
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_ds,
        batch_size=config["batch size"],
        epochs=config["num epochs"],
        validation_data=test_ds,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history
