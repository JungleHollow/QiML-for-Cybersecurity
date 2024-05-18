import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
from pennylane.optimize import NesterovMomentumOptimizer

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical

import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

import pandas as pd

import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64")


def basic_neural_network():
    data = load_iris()

    X = data["data"]
    Y = data["target"]
    Y = to_categorical(Y, num_classes=len(np.unique(Y)))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_val_nn = scaler.transform(X_val)

    n_qubits = 4
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=dev.wires)
        qml.BasicEntanglerLayers(weights, wires=dev.wires)
        return [qml.expval(qml.PauliZ(wires=i)) for i in dev.wires]

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}

    # tmp_layer = layers.Dense(4, activation="relu")
    # tmp_layer_output = tmp_layer(X_train_nn)
    # print(tmp_layer_output)

    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    # qlayer_output = qlayer(tmp_layer_output)
    # print(qlayer_output)

    tf.get_logger().setLevel("ERROR")

    model = keras.Sequential([
        layers.Dense(4, activation="relu"),
        qlayer,
        layers.Dense(4, activation="relu"),
        qlayer,
        layers.Dense(4, activation="relu"),
        qlayer,
        layers.Dense(4, activation="relu"),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train_nn, Y_train, validation_data=(X_val_nn, Y_val), epochs=100, batch_size=5)


def cicddos_neural_network():
    filepath = f"./Datasets/CICDDoS2019/01-12(Training)/DrDoS_NTP.csv"
    cicddos_data = pd.read_csv(filepath, usecols=range(8, 85))

    cicddos_labels = pd.read_csv(filepath, usecols=[87])
    cicddos_labels = np.where(cicddos_labels == "BENIGN", 0, 1)
    cicddos_labels = to_categorical(cicddos_labels, num_classes=len(np.unique(cicddos_labels)))

    X_train, X_val, Y_train, Y_val = train_test_split(cicddos_data, cicddos_labels, test_size=0.2, random_state=0)

    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_val = (X_val - X_val.min()) / (X_val.max() - X_val.min())

    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train)
    X_val_nn = scaler.transform(X_val)

    n_qubits = 20
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=dev.wires)
        qml.BasicEntanglerLayers(weights, wires=dev.wires)
        return [qml.expval(qml.PauliZ(wires=i)) for i in dev.wires]

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}

    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    tf.get_logger().setLevel("ERROR")

    model = keras.Sequential([
        layers.Dense(77, activation="relu"),
        layers.Dense(60, activation="relu"),
        layers.Dense(30, activation="relu"),
        layers.Dense(20, activation="relu"),
        qlayer,
        # layers.Dense(77, activation="relu"),
        # qlayer,
        # layers.Dense(77, activation="relu"),
        # qlayer,
        # layers.Dense(77, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train_nn, Y_train, validation_data=(X_val_nn, Y_val), epochs=100, batch_size=100)


if __name__ == '__main__':
    # basic_neural_network()
    cicddos_neural_network()
    # pass
