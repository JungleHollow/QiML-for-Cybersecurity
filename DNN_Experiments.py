import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import sklearn_pandas as spd
import numpy as np

import tensorflow.keras.callbacks
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.optimizers import Adam

import torch
from torch.nn.functional import relu

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
from pennylane.optimize import NesterovMomentumOptimizer

tf.keras.backend.set_floatx("float64")


class DNN:
    def __init__(self, trainpath, testpath, loadpath=None, layers=1, dropout=0.01, learning_rate=0.01, batch_size=32, epochs=100):
        self.training_data, self.training_labels, self.testing_data, self.testing_labels = self.read_data(trainpath, testpath)
        if loadpath:
            self.model = load_model(loadpath)
        elif layers == 1:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dense(10, activation="softmax")
            ])
        elif layers == 2:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dense(10, activation="softmax")
            ])
        elif layers == 3:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dense(10, activation="softmax")
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
                Dense(10, activation="softmax")
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
                Dense(10, activation="softmax")
            ])
        else:
            print(f"Invalid number of layers specified; using the default number of layers...")
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dense(10, activation="softmax")
            ])

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.opt = Adam(learning_rate=self.lr)
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=0)]
        self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def lr_scheduler(self, epoch):
        # Maintain the learning rate at a constant as in the study's experiments
        return self.lr

    def read_data(self, trainpath, testpath):
        # Currently semi hard coded for the UNSW NB15 csv formats...
        train_frame = pd.read_csv(trainpath, dtype={"proto": "category", "service": "category", "state": "category", "attack_cat": "category", "label": "category"})
        train_frame = train_frame.sort_values(by=["id"])  # In case the data is somehow not read in order

        x_train = train_frame.iloc[:, :-2]
        x_train_cat = x_train.select_dtypes(include=["category"]).columns
        x_train_num = x_train.select_dtypes(include=["number"]).columns
        y_train = train_frame.iloc[:, -2]

        test_frame = pd.read_csv(testpath, dtype={"proto": "category", "service": "category", "state": "category", "attack_cat": "category", "label": "category"})
        test_frame = test_frame.sort_values(by=["id"])

        x_test = test_frame.iloc[:, :-2]
        x_test_cat = x_test.select_dtypes(include=["category"]).columns
        x_test_num = x_test.select_dtypes(include=["number"]).columns
        y_test = test_frame.iloc[:, -2]

        scaler = StandardScaler()
        x_train[x_train_num] = scaler.fit_transform(x_train[x_train_num], y_train)
        x_test[x_test_num] = scaler.transform(x_test[x_test_num])

        x_train[x_train_cat] = x_train[x_train_cat].apply(lambda n: n.cat.codes)
        x_test[x_test_cat] = x_test[x_test_cat].apply(lambda n: n.cat.codes)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def train_model(self):
        self.model.fit(self.training_data, self.training_labels, batch_size=self.batch_size, epochs=self.epochs, shuffle=False, callbacks=self.callbacks)

    def evaluate_model(self):
        self.loss, self.accuracy = self.model.evaluate(self.testing_data, self.testing_labels, batch_size=self.batch_size)

        prediction_classes = (self.model.predict(self.testing_data)).astype("int32")

        self.precision = precision_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)
        self.recall = recall_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)
        self.f1_score = f1_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)

        print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")

    def save(self, savepath):
        save_model(self.model, savepath)
        print(f"Model saved to {savepath} successfully...")
        
        
class DQiNN:
    def __init__(self, trainpath, testpath, loadpath=None, layers=1, dropout=0.01, learning_rate=0.01, batch_size=32, epochs=100, n_qubits=16):
        self.training_data, self.training_labels, self.testing_data, self.testing_labels = self.read_data(trainpath, testpath)

        self.n_layers = layers
        self.n_qubits = n_qubits
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
            
        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits)}

        @qml.qnode(self.dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.dev.wires)
            qml.BasicEntanglerLayers(weights, wires=self.dev.wires)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.dev.wires]
        
        self.qlayer = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)
        
        if loadpath:
            self.model = load_model(loadpath)
        elif self.n_layers == 1:
            self.model = Sequential([
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
            ])
        elif self.n_layers == 2:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
            ])
        elif self.n_layers == 3:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
            ])
        elif self.n_layers == 4:
            self.model = Sequential([
                Dense(1024, activation="relu"),
                Dropout(dropout),
                Dense(768, activation="relu"),
                Dropout(dropout),
                Dense(512, activation="relu"),
                Dropout(dropout),
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
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
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
            ])
        else:
            print(f"Invalid number of layers specified; using the default number of layers...")
            self.model = Sequential([
                Dense(self.n_qubits, activation="relu"),
                Dropout(dropout),
                self.qlayer,
                Dense(10, activation="softmax")
            ])

        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.opt = Adam(learning_rate=self.lr)
        self.callbacks = [tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler, verbose=0)]
        self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def lr_scheduler(self, epoch):
        # Maintain the learning rate at a constant as in the study's experiments
        return self.lr

    def read_data(self, trainpath, testpath):
        # Currently semi hard coded for the UNSW NB15 csv formats...
        train_frame = pd.read_csv(trainpath, dtype={"proto": "category", "service": "category", "state": "category", "attack_cat": "category", "label": "category"})
        train_frame = train_frame.sort_values(by=["id"])  # In case the data is somehow not read in order

        x_train = train_frame.iloc[:, :-2]
        x_train_cat = x_train.select_dtypes(include=["category"]).columns
        x_train_num = x_train.select_dtypes(include=["number"]).columns
        y_train = train_frame.iloc[:, -2]

        test_frame = pd.read_csv(testpath, dtype={"proto": "category", "service": "category", "state": "category", "attack_cat": "category", "label": "category"})
        test_frame = test_frame.sort_values(by=["id"])

        x_test = test_frame.iloc[:, :-2]
        x_test_cat = x_test.select_dtypes(include=["category"]).columns
        x_test_num = x_test.select_dtypes(include=["number"]).columns
        y_test = test_frame.iloc[:, -2]

        scaler = StandardScaler()
        x_train[x_train_num] = scaler.fit_transform(x_train[x_train_num], y_train)
        x_test[x_test_num] = scaler.transform(x_test[x_test_num])

        x_train[x_train_cat] = x_train[x_train_cat].apply(lambda n: n.cat.codes)
        x_test[x_test_cat] = x_test[x_test_cat].apply(lambda n: n.cat.codes)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def train_model(self):
        self.model.fit(self.training_data, self.training_labels, batch_size=self.batch_size, epochs=self.epochs, shuffle=False, callbacks=self.callbacks)

    def evaluate_model(self):
        self.loss, self.accuracy = self.model.evaluate(self.testing_data, self.testing_labels, batch_size=self.batch_size)

        prediction_classes = (self.model.predict(self.testing_data)).astype("int32")

        self.precision = precision_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)
        self.recall = recall_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)
        self.f1_score = f1_score(self.testing_labels, prediction_classes, average="weighted", zero_division=np.nan)

        print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")

    def save(self, savepath):
        save_model(self.model, savepath)
        print(f"Model saved to {savepath} successfully...")


if __name__ == "__main__":
    TRAINING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_training-set.csv"
    TESTING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_testing-set.csv"
    N_LAYERS = 3
    DROPOUT = 0.01
    LEARNING_RATE = 0.01
    BATCH_SIZE = 256
    N_EPOCHS = 100
    SAVEPATH = f"./models/DNN_{N_LAYERS}-layer_{N_EPOCHS}-epochs.keras"

    my_model = DNN(TRAINING_PATH, TESTING_PATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
    my_model.train_model()
    my_model.evaluate_model()
    my_model.save(SAVEPATH)

    # my_model = DNN(TRAINING_PATH, TESTING_PATH, loadpath=SAVEPATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS)
    # my_model.evaluate_model()
    
    N_QUBITS = 10
    Q_SAVEPATH = f"./models/DQiNN_{N_LAYERS}-layer_{N_QUBITS}-qubit_{N_EPOCHS}-epochs.keras"
    
    # my_q_model = DQiNN(TRAINING_PATH, TESTING_PATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, n_qubits=N_QUBITS)
    # my_q_model.train_model()
    # my_q_model.evaluate_model()
    # my_q_model.save(Q_SAVEPATH)
    #
    # my_q_model = DQiNN(TRAINING_PATH, TESTING_PATH, loadpath=Q_SAVEPATH, layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, n_qubits=N_QUBITS)
    # my_q_model.evaluate_model()

    # TODO: Start training models with moderate layers and more qubits...
    # TODO: Test with allowing the model to shuffle instances
