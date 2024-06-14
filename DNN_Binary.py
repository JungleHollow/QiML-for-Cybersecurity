import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gc
import multiprocessing
import pandas as pd
import numpy as np
np.random.seed(1337)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.backend import clear_session
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
from pennylane.optimize import NesterovMomentumOptimizer

tf.keras.backend.set_floatx("float64")


# ==== Reset the keras session (trying to work around memory leak issues) ====
def reset_keras():
    clear_session()
    gc.collect()


# ==== Read in the stored data appropriately to use for model training and evaluation ====
def read_data(trainpath, testpath):
    log_cols = ["dur", "sbytes", "dbytes", "sload", "dload", "spkts", "stcpb", "dtcpb", "sjit", "djit"]

    train_frame = pd.read_csv(trainpath)
    train_frame["service"] = train_frame["service"].apply(lambda x: "None" if x == "-" else x)
    train_frame[log_cols].apply(np.log1p)  # Apply logarithm transformations to skewed features

    test_frame = pd.read_csv(testpath)
    test_frame["service"] = test_frame["service"].apply(lambda x: "None" if x == "-" else x)
    test_frame[log_cols].apply(np.log1p)

    train_numeric = train_frame.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns
    test_numeric = test_frame.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns

    # Standardize the numeric variables in each dataframe
    scaler = StandardScaler()
    train_frame[train_numeric] = scaler.fit_transform(train_frame[train_numeric], train_frame["label"])
    test_frame[test_numeric] = scaler.transform(test_frame[test_numeric])

    del train_numeric, test_numeric

    # Onehot encode the categorical variables in each dataframe
    for col in ["proto", "service", "state"]:
        ohe = OneHotEncoder(handle_unknown="ignore")

        train_cols = ohe.fit_transform(train_frame[col].values.reshape(-1, 1))
        test_cols = ohe.transform(test_frame[col].values.reshape(-1, 1))

        train_df = pd.DataFrame(train_cols.todense(), columns=[col + "_" + str(i) for i in ohe.categories_[0]])
        test_df = pd.DataFrame(test_cols.todense(), columns=[col + "_" + str(i) for i in ohe.categories_[0]])

        train_frame = pd.concat([train_frame.drop(col, axis=1), train_df], axis=1)
        test_frame = pd.concat([test_frame.drop(col, axis=1), test_df], axis=1)

    # Sample one in every 50 packets randomly (reasonable training times without much loss in predictive power)
    train_frame, discarded_train = train_test_split(train_frame, test_size=0.95, random_state=1337)
    test_frame, discarded_test = train_test_split(test_frame, test_size=0.95, random_state=1337)

    del discarded_train, discarded_test

    # Sort by id to maintain the temporal structure of the data
    train_frame.sort_values(by=["id"], inplace=True)
    test_frame.sort_values(by=["id"], inplace=True)

    # Split each dataframe into their respective input and target variables for use in model training and evaluation
    y_train = train_frame["label"]
    x_train = train_frame.drop(["id", "attack_cat", "label"], axis=1).to_numpy().astype("float32")

    y_test = test_frame["label"]
    x_test = test_frame.drop(["id", "attack_cat", "label"], axis=1).to_numpy().astype("float32")

    del train_frame, test_frame

    return x_train, y_train, x_test, y_test


# ==== Model classes with their appropriate implementations and configurations ====
class DNN:
    def __init__(self, trainpath, testpath, savepath, loadpath=None, layers=1, dropout=0.01,
                 learning_rate=0.01, batch_size=32, epochs=100, threshold=0.5):
        self.training_set, self.training_labels, self.testing_set, self.testing_labels = self.read_data(trainpath, testpath)
        gc.collect()

        self.save_path = savepath
        self.chk_path = f"./models_binary/checkpoints/{self.save_path.split('/')[-1]}"

        self.batch_size = batch_size
        self.prediction_threshold = threshold

        if loadpath:
            self.model = load_model(loadpath)
        else:
            self.model = Sequential()
            self.model.add(Dense(1024, activation="relu"))
            if layers >= 2:
                self.model.add(Dropout(dropout))
                self.model.add(Dense(768, activation="relu"))
            if layers >= 3:
                self.model.add(Dropout(dropout))
                self.model.add(Dense(512, activation="relu"))
            if layers >= 4:
                self.model.add(Dropout(dropout))
                self.model.add(Dense(256, activation="relu"))
            if layers >= 5:
                self.model.add(Dropout(dropout))
                self.model.add(Dense(128, activation="relu"))
            self.model.add(Dense(1, activation="sigmoid"))

        self.checkpointing = ModelCheckpoint(filepath=self.chk_path, save_best_only=True, monitor="val_loss",
                                             mode="min")

        self.lr = learning_rate
        self.epochs = epochs
        self.opt = Adam(learning_rate=self.lr)
        self.model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def read_data(self, trainpath, testpath):
        return read_data(trainpath, testpath)

    def train_model(self):
        self.model.fit(
            self.training_set, self.training_labels, validation_split=0.1, epochs=self.epochs,
            batch_size=self.batch_size, callbacks=[self.checkpointing]
        )
        gc.collect()

    def evaluate_model(self):
        best_model = load_model(self.chk_path)
        prediction_proba = best_model.predict(self.testing_set, batch_size=self.batch_size)

        prediction_proba = prediction_proba.reshape(prediction_proba.shape[0],)

        precision, recall, thresholds = precision_recall_curve(self.testing_labels, prediction_proba)
        fscore = (2 * precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(fscore)]

        prediction_classes = np.where(prediction_proba > optimal_threshold, 1, 0)

        gc.collect()

        conf_mat = confusion_matrix(self.testing_labels, prediction_classes)
        print(conf_mat)

        loss_fn = BinaryCrossentropy()
        self.loss = loss_fn(self.testing_labels, prediction_proba)
        self.accuracy = accuracy_score(self.testing_labels, prediction_classes)

        self.precision = precision_score(self.testing_labels, prediction_classes)
        self.recall = recall_score(self.testing_labels, prediction_classes)
        self.f1_score = f1_score(self.testing_labels, prediction_classes)

        print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")

    def save(self):
        self.model.save(self.save_path)
        print(f"Model saved to {self.save_path} successfully...")


# Loading and saving of DQiNN models is done with weights instead of full models to work around a layer type issue
class DQiNN:
    def __init__(self, trainpath, testpath, savepath, loadpath=None, layers=1, dropout=0.01,
                 learning_rate=0.01, batch_size=32, epochs=100, threshold=0.5, n_qubits=16):
        self.training_set, self.training_labels, self.testing_set, self.testing_labels = self.read_data(trainpath, testpath)
        gc.collect()

        self.save_path = savepath
        self.chk_path = f"./models_binary/checkpoints/{self.save_path.split('/')[-1]}"

        self.prediction_threshold = threshold
        self.batch_size = batch_size

        self.dropout = dropout
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

        self.model = Sequential()
        self.model.add(Dense(self.n_qubits, activation="relu"))
        self.model.add(self.qlayer)
        if self.n_layers >= 2:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer)
        if self.n_layers >= 3:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer)
        if self.n_layers >= 4:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer)
        if self.n_layers >= 5:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer)
        self.model.add(Dense(1, activation="sigmoid"))

        self.lr = learning_rate
        self.epochs = epochs
        self.opt = Adam(learning_rate=self.lr)
        self.model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        if loadpath:
            self.model.fit(self.training_set, self.training_labels, batch_size=1, steps_per_epoch=1, epochs=1, verbose=0)   # Needed to allow weight loading
            self.model.load_weights(loadpath)

        self.checkpointing = ModelCheckpoint(filepath=self.chk_path, save_best_only=True, monitor="val_loss",
                                             mode="min", save_weights_only=True)

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def read_data(self, trainpath, testpath):
        return read_data(trainpath, testpath)

    def train_model(self):
        self.model.fit(
            self.training_set, self.training_labels, validation_split=0.1, epochs=self.epochs,
            batch_size=self.batch_size, callbacks=[self.checkpointing]
        )
        gc.collect()

    def evaluate_model(self):
        best_model = Sequential()
        best_model.add(Dense(self.n_qubits, activation="relu"))
        best_model.add(self.qlayer)
        if self.n_layers >= 2:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer)
        if self.n_layers >= 3:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer)
        if self.n_layers >= 4:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer)
        if self.n_layers >= 5:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer)
        best_model.add(Dense(1, activation="sigmoid"))
        best_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        best_model.fit(self.training_set, self.training_labels, batch_size=1, steps_per_epoch=1, epochs=1, verbose=0)  # Needed to allow weight loading
        best_model.load_weights(self.chk_path)

        prediction_proba = best_model.predict(self.testing_set, batch_size=self.batch_size)

        prediction_proba = prediction_proba.reshape(prediction_proba.shape[0],)

        precision, recall, thresholds = precision_recall_curve(self.testing_labels, prediction_proba)
        fscore = (2 * precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(fscore)]

        prediction_classes = np.where(prediction_proba > optimal_threshold, 1, 0)

        gc.collect()

        loss_fn = BinaryCrossentropy()
        self.loss = loss_fn(self.testing_labels, prediction_proba)
        self.accuracy = accuracy_score(self.testing_labels, prediction_classes)

        self.precision = precision_score(self.testing_labels, prediction_classes)
        self.recall = recall_score(self.testing_labels, prediction_classes)
        self.f1_score = f1_score(self.testing_labels, prediction_classes)

        print(f"==== MODEL EVALUATION COMPLETE ====\n\n- Loss: {self.loss}\n- Accuracy: {self.accuracy}\n- Precision: {self.precision}\n- Recall: {self.recall}\n- F1 Score: {self.f1_score}\n")

    def save(self):
        self.model.save_weights(self.save_path)
        print(f"Model saved to {self.save_path} successfully...")


# ==== Dummy functions to enable multiprocessing trick (workaround for keras model.predict() memory leak) ====
def init_dnn():
    if os.path.isfile(SAVEPATH):
        model = DNN(TRAINING_PATH, TESTING_PATH, SAVEPATH, loadpath=SAVEPATH, layers=N_LAYERS,
                    dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    threshold=THRESHOLD)
        model.evaluate_model()
    else:
        model = DNN(TRAINING_PATH, TESTING_PATH, SAVEPATH, layers=N_LAYERS, dropout=DROPOUT,
                    learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, threshold=THRESHOLD)
        model.train_model()
        model.save()
        model.evaluate_model()


def init_dqinn():
    if os.path.isfile(Q_SAVEPATH):
        my_q_model = DQiNN(TRAINING_PATH, TESTING_PATH, Q_SAVEPATH, loadpath=Q_SAVEPATH,
                           layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                           epochs=N_EPOCHS, threshold=THRESHOLD, n_qubits=N_QUBITS)
        my_q_model.evaluate_model()
    else:
        my_q_model = DQiNN(TRAINING_PATH, TESTING_PATH, Q_SAVEPATH, layers=N_LAYERS, dropout=DROPOUT,
                           learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, threshold=THRESHOLD,
                           n_qubits=N_QUBITS)
        my_q_model.train_model()
        my_q_model.save()
        my_q_model.evaluate_model()


# ==== Parameter configuration and script initialisation ====
if __name__ == "__main__":
    reset_keras()

    TRAINING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_training-set.csv"
    TESTING_PATH = "./Datasets/UNSW_NB15/UNSW_NB15_testing-set.csv"

    DATASET = "UNSW"
    N_LAYERS = 3
    DROPOUT = 0.1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    N_EPOCHS = 1000
    THRESHOLD = 0.5
    SAVEPATH = f"./models_binary/DNN_{N_LAYERS}-layer_{N_EPOCHS}-epochs_{DATASET}.keras"

    # p = multiprocessing.Process(target=init_dnn)
    # p.start()
    # p.join()

    N_QUBITS = 8
    Q_SAVEPATH = f"./models_binary/DQiNN_{N_LAYERS}-layer_{N_QUBITS}-qubit_{N_EPOCHS}-epochs_{DATASET}.keras"

    p_q = multiprocessing.Process(target=init_dqinn)
    p_q.start()
    p_q.join()
