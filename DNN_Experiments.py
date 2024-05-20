import pandas as pd
import numpy as np


import keras.callbacks
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.optimizers import Adam

import torch
from torch.nn.functional import relu

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
from pennylane.optimize import NesterovMomentumOptimizer


class DNN:
    def __init__(self, datapath, loadpath=None, layers=1, dropout=0.01, learning_rate=0.01):
        self.training_data, self.training_labels = self.read_data(datapath)
        self.testing_data, self.testing_labels = None, None
        if loadpath:
            self.model = load_model(loadpath)
        elif layers == 1:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
        elif layers == 2:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
        elif layers == 3:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
        elif layers == 4:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dropout(dropout),
                Dense(256, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
        elif layers == 5:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dropout(dropout),
                Dense(256, activation="relu"),
                Dropout(dropout),
                Dense(128, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
        else:
            print(f"Invalid number of layers specified; using the default number of layers...")
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dense(1, activation="sigmoid")
            ])

        self.lr = learning_rate
        self.opt = Adam(lr=self.lr)
        self.callbacks = [keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1)]
        self.model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def lr_scheduler(self, epoch):
        # Maintain the learning rate at a constant as in the study's experiments
        return self.lr

    def read_data(self, datapath):
        # Currently semi hard coded for the UNSW NB15 csv formats...
        dataframe = pd.read_csv(datapath, dtype={"proto": "category", "service": "category", "state": "category"})
        dataframe = dataframe.sort_values(by=["id"])  # In case the data is somehow not read in order
        x = dataframe.iloc[:, :-2]
        x_cat_columns = x.select_dtypes(["category"]).columns
        x[x_cat_columns] = x[x_cat_columns].apply(lambda n: n.cat.codes)
        y = dataframe.iloc[:, -1]
        return x, y

    def train_model(self):
        self.model.fit(self.training_data, self.training_labels, epochs=100, shuffle=False, callbacks=self.callbacks)

    def read_test_data(self, datapath):
        x, y = self.read_data(datapath)
        self.testing_data = x
        self.testing_labels = y
        print(f"Testing data successfully read...")

    def evaluate_model(self):
        if self.testing_data is not None and self.testing_labels is not None:
            self.loss, self.accuracy = self.model.evaluate(self.testing_data, self.testing_labels)

            prediction_classes = (self.model.predict(self.testing_data) > 0.5).astype("int32")

            self.precision = precision_score(self.testing_labels, prediction_classes)
            self.recall = recall_score(self.testing_labels, prediction_classes)
            self.f1_score = f1_score(self.testing_labels, prediction_classes)

            print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")
        else:
            print(f"No test data has been read in yet. Use read_test_data(path) to read the test data...")

    def save(self, savepath):
        save_model(self.model, savepath)
        print(f"Model saved to {savepath} successfully...")
        
        
class DQiNN:
    def __init__(self, datapath, loadpath=None, layers=1, dropout=0.01, learning_rate=0.01, n_qubits=16):
        self.training_data, self.training_labels = self.read_data(datapath)
        self.testing_data, self.testing_labels = None, None
        
        self.n_layers = layers
        self.n_qubits = n_qubits
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev)
        def qnode(self, inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.dev.wires)
            qml.BasicEntanglerLayers(weights, wires=self.dev.wires)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.dev.wires]
            
        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits)}
        
        self.qlayer = qml.qnn.KerasLayer(self.qnode, self.weight_shapes, output_dim=self.n_qubits)
        
        if loadpath:
            self.model = load_model(loadpath)
        elif self.n_layers == 1:
            self.model = Sequential([
                qlayer,
                Dense(1, activation="sigmoid")
            ])
        elif self.n_layers == 2:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                qlayer,
                Dense(1, activation="sigmoid")
            ])
        elif self.n_layers == 3:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                qlayer,
                Dense(1, activation="sigmoid")
            ])
        elif self.n_layers == 4:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dropout(dropout),
                qlayer,
                Dense(1, activation="sigmoid")
            ])
        elif self.n_layers == 5:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dropout(dropout),
                Dense(256, activation="relu"),
                Dropout(dropout),
                qlayer,
                Dense(1, activation="sigmoid")
            ])
        else:
            print(f"Invalid number of layers specified; using the default number of layers...")
            self.model = Sequential([
                qlayer,
                Dense(1, activation="sigmoid")
            ])

        self.lr = learning_rate
        self.opt = Adam(lr=self.lr)
        self.callbacks = [keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=1)]
        self.model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def lr_scheduler(self, epoch):
        # Maintain the learning rate at a constant as in the study's experiments
        return self.lr

    def read_data(self, datapath):
        # Currently semi hard coded for the UNSW NB15 csv formats...
        dataframe = pd.read_csv(datapath, dtype={"proto": "category", "service": "category", "state": "category"})
        dataframe = dataframe.sort_values(by=["id"])  # In case the data is somehow not read in order
        x = dataframe.iloc[:, :-2]
        x_cat_columns = x.select_dtypes(["category"]).columns
        x[x_cat_columns] = x[x_cat_columns].apply(lambda n: n.cat.codes)
        y = dataframe.iloc[:, -1]
        return x, y

    def train_model(self):
        self.model.fit(self.training_data, self.training_labels, epochs=100, shuffle=False, callbacks=self.callbacks)

    def read_test_data(self, datapath):
        x, y = self.read_data(datapath)
        self.testing_data = x
        self.testing_labels = y
        print(f"Testing data successfully read...")

    def evaluate_model(self):
        if self.testing_data is not None and self.testing_labels is not None:
            self.loss, self.accuracy = self.model.evaluate(self.testing_data, self.testing_labels)

            prediction_classes = (self.model.predict(self.testing_data) > 0.5).astype("int32")

            self.precision = precision_score(self.testing_labels, prediction_classes)
            self.recall = recall_score(self.testing_labels, prediction_classes)
            self.f1_score = f1_score(self.testing_labels, prediction_classes)

            print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")
        else:
            print(f"No test data has been read in yet. Use read_test_data(path) to read the test data...")

    def save(self, savepath):
        save_model(self.model, savepath)
        print(f"Model saved to {savepath} successfully...")


if __name__ == "__main__":
    TRAINING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_training-set.csv"
    TESTING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_testing-set.csv"
    N_LAYERS = 1
    DROPOUT = 0.01
    LEARNING_RATE = 0.01
    SAVEPATH = f"./models/DNN_{N_LAYERS}-layer.keras"

    my_model = DNN(TRAINING_PATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE)
    my_model.train_model()
    my_model.read_test_data(TESTING_PATH)
    my_model.evaluate_model()
    my_model.save(SAVEPATH)

    # my_model = DNN(TRAINING_PATH, loadpath=SAVEPATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE)
    # my_model.read_test_data(TESTING_PATH)
    # my_model.evaluate_model()
    
    N_QUBITS = 16
    Q_SAVEPATH = f"./models/DQiNN_{N_LAYERS}-layer_{N_QUBITS}-qubit.keras"
    
    my_q_model = DQiNN(TRAINING_PATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, n_qubits=N_QUBITS)
    my_q_model.train_model()
    my_q_model.read_test_data(TESTING_PATH)
    my_q_model.evaluate_model()
    my_q_model.save(Q_SAVEPATH)
