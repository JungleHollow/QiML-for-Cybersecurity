import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import gc
import multiprocessing
import pandas as pd
import numpy as np
np.random.seed(1337)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import save_model, load_model, Sequential
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.backend import clear_session
from tensorflow.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
import pennylane as qml
from pennylane.templates import AngleEmbedding, BasicEntanglerLayers

tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx("float64")

# ==== Initialise parameters as globals outside __name__ == __main__ so that spawned Windows processes can see them ====

global ENTANGLEMENT_POPULATION, STRONGEST_ENTANGLEMENT

ENTANGLEMENT_POPULATION = []
STRONGEST_ENTANGLEMENT = (None, np.inf)

global TRAINING_PATH, VALIDATION_PATH, TESTING_PATH
TRAINING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_training-set"
VALIDATION_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_validation-set"
TESTING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_testing-set"
# TRAINING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS17_training-set"
# VALIDATION_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS17_validation-set"
# TESTING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS17_testing-set"
# TRAINING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\IoT23_training-set"
# VALIDATION_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\IoT23_validation-set"
# TESTING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\IoT23_testing-set"
# TRAINING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICDDoS19_training-set"
# VALIDATION_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICDDoS19_validation-set"
# TESTING_PATH = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICDDoS19_testing-set"

global DATASET, N_LAYERS, DROPOUT, LEARNING_RATE, BATCH_SIZE, N_EPOCHS, THRESHOLD
DATASET = "UNSW"
# DATASET = "CICIDS17"
# DATASET = "IoT23"
# DATASET = "CICDDoS19"
N_LAYERS = 1
DROPOUT = 0.1
LEARNING_RATE = 0.01
BATCH_SIZE = 64
N_EPOCHS = 1000
THRESHOLD = 0.5

global N_QUBITS, Q_SAVEPATH, ENTANGLEMENT
N_QUBITS = 4
ENTANGLEMENT = "Basic-Entanglement"
Q_SAVEPATH = rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\DQiNN_{ENTANGLEMENT}.keras"


# ==== Reset the keras session (trying to work around memory leak issues) ====
def reset_keras():
    clear_session()
    gc.collect()


# ==== Read in the stored data appropriately to use for model training and evaluation ====
def read_data(trainpath, valpath, testpath):
    train_ds = Dataset.load(trainpath)
    val_ds = Dataset.load(valpath)

    test_ds = Dataset.load(testpath)
    test_frame = pd.read_csv(f"{testpath}.csv")

    y_test = test_frame["label"]

    del test_frame

    return train_ds, val_ds, test_ds, y_test


# ==== Workaround used to make the tensorflow csv datasets compatible with keras model functions ====
def dataset_fix(features, labels):
    return tf.stack(list(features.values()), axis=1), labels


# ==== Model classes with their appropriate implementations and configurations ====
# Loading and saving of DQiNN models is done with weights instead of full models to work around a layer type issue
class DQiNN:
    def __init__(self, trainpath, valpath, testpath, savepath, loadpath=None, layers=1, dropout=0.01,
                 learning_rate=0.01, batch_size=32, epochs=100, threshold=0.5, n_qubits=16, entanglements=None):
        self.training_set, self.validation_set, self.testing_set, self.testing_labels = self.read_data(trainpath, valpath, testpath)
        gc.collect()

        self.save_path = savepath
        tmp_name = self.save_path.split("\\")[-1]
        self.chk_path = rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\checkpoints\{tmp_name}"
        del tmp_name

        self.prediction_threshold = threshold
        self.batch_size = batch_size

        self.dropout = dropout
        self.n_layers = layers
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.weight_shapes = {"weights": (self.n_layers, self.n_qubits)}

        @qml.qnode(self.dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.dev.wires)
            qml.BasicEntanglerLayers(weights, wires=self.dev.wires)

            if entanglements:
                for entanglement in entanglements:
                    qml.apply(entanglement)

            # qml.CNOT(wires=[1, 0])
            # qml.CNOT(wires=[1, 3])
            # qml.CNOT(wires=[2, 0])
            # qml.CNOT(wires=[2, 3])

            return [qml.expval(qml.PauliZ(wires=i)) for i in self.dev.wires]

        self.qlayer = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)
        self.qlayer_2 = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)
        self.qlayer_3 = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)
        self.qlayer_4 = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)
        self.qlayer_5 = qml.qnn.KerasLayer(qnode, self.weight_shapes, output_dim=self.n_qubits)

        self.history = None

        self.model = Sequential()
        self.model.add(Dense(self.n_qubits, activation="relu"))
        self.model.add(self.qlayer)
        if self.n_layers >= 2:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer_2)
        if self.n_layers >= 3:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer_3)
        if self.n_layers >= 4:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer_4)
        if self.n_layers >= 5:
            self.model.add(Dropout(self.dropout))
            self.model.add(self.qlayer_5)
        self.model.add(Dense(1, activation="sigmoid"))

        self.lr = learning_rate
        self.epochs = epochs
        self.opt = Adam(learning_rate=self.lr)
        self.model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        if loadpath:
            self.model.fit(self.training_set.map(dataset_fix), steps_per_epoch=1, epochs=1, verbose=0)   # Needed to allow weight loading
            self.model.load_weights(loadpath)

        self.early_stop = EarlyStopping(monitor="val_loss", patience=30, mode="min")
        self.checkpointing = ModelCheckpoint(filepath=self.chk_path, save_best_only=True, monitor="val_loss",
                                             mode="min", save_weights_only=True)

        self.loss = -1
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def read_data(self, trainpath, valpath, testpath):
        return read_data(trainpath, valpath, testpath)

    def train_model(self):
        self.history = self.model.fit(
            self.training_set.map(dataset_fix), validation_data=self.validation_set.map(dataset_fix), epochs=self.epochs,
            callbacks=[self.checkpointing, self.early_stop]
        )
        gc.collect()

    def evaluate_model(self):
        best_model = Sequential()
        best_model.add(Dense(self.n_qubits, activation="relu"))
        best_model.add(self.qlayer)
        if self.n_layers >= 2:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer_2)
        if self.n_layers >= 3:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer_3)
        if self.n_layers >= 4:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer_4)
        if self.n_layers >= 5:
            best_model.add(Dropout(self.dropout))
            best_model.add(self.qlayer_5)
        best_model.add(Dense(1, activation="sigmoid"))

        best_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        best_model.fit(self.training_set.map(dataset_fix), steps_per_epoch=1, epochs=1, verbose=0)  # Needed to allow weight loading
        best_model.load_weights(self.chk_path)

        prediction_proba = best_model.predict(self.testing_set.map(dataset_fix))

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
        print(best_model.summary())

    def save(self):
        self.model.save_weights(self.save_path)

        tmp_df = pd.DataFrame()
        tmp_df["loss"] = self.history.history["loss"]
        tmp_df["accuracy"] = self.history.history["accuracy"]
        tmp_df["val_loss"] = self.history.history["val_loss"]
        tmp_df["val_accuracy"] = self.history.history["val_accuracy"]

        tmp_df.to_csv(rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\histories\DQiNN_{ENTANGLEMENT}.csv", index=False)

        print(f"Model saved to {self.save_path} successfully...")


# ==== Dummy functions to enable multiprocessing trick (workaround for keras model.predict() memory leak) ====
def init_dqinn():
    if os.path.isfile(Q_SAVEPATH):
        my_q_model = DQiNN(TRAINING_PATH, VALIDATION_PATH, TESTING_PATH, Q_SAVEPATH, loadpath=Q_SAVEPATH,
                           layers=N_LAYERS, dropout=DROPOUT, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE,
                           epochs=N_EPOCHS, threshold=THRESHOLD, n_qubits=N_QUBITS)
        my_q_model.evaluate_model()
    else:
        my_q_model = DQiNN(TRAINING_PATH, VALIDATION_PATH, TESTING_PATH, Q_SAVEPATH, layers=N_LAYERS, dropout=DROPOUT,
                           learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, threshold=THRESHOLD,
                           n_qubits=N_QUBITS)
        my_q_model.train_model()
        my_q_model.save()
        my_q_model.evaluate_model()


class GeneticAlgorithm:
    def __init__(self, generations=5):
        self.combinations = ["12", "13", "14", "21", "23", "24", "31", "32", "34", "41", "42", "43"]
        self.entanglement_population = []  # To track what combinations have already been tried
        self.strongest_entanglements = {"1": (None, {"f1_score": -np.inf}), "2": (None, {"f1_score": -np.inf}), "3": (None, {"f1_score": -np.inf}), "4": (None, {"f1_score": -np.inf})}
        self.generation = 1
        self.max_generations = generations
        self.generation_size = len(self.combinations)

    # Initialize the first generation of results
    def initialize(self):
        for combination in self.combinations:
            reset_keras()
            save_path = rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\DQiNN_{combination}.keras"
            entanglement_gates = self.create_gates(combination)

            model = DQiNN(TRAINING_PATH, VALIDATION_PATH, TESTING_PATH, save_path, layers=N_LAYERS, dropout=DROPOUT,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, threshold=THRESHOLD,
                          n_qubits=N_QUBITS, entanglements=entanglement_gates)
            model.train_model()
            model.save()
            model.evaluate_model()

            result = (combination, {"loss": model.loss, "accuracy": model.accuracy, "precision": model.precision, "recall": model.recall, "f1_score": model.f1_score})
            ranking = self.rank_entanglement(result)
            if ranking != 5:
                self.update_ranking(result, ranking)
            self.entanglement_population.append(combination)

    def update_ranking(self, result, ranking):
        if ranking == 4:
            self.strongest_entanglements["4"] = result
        elif ranking == 3:
            self.strongest_entanglements["4"] = self.strongest_entanglements["3"]
            self.strongest_entanglements["3"] = result
        elif ranking == 2:
            self.strongest_entanglements["4"] = self.strongest_entanglements["3"]
            self.strongest_entanglements["3"] = self.strongest_entanglements["2"]
            self.strongest_entanglements["2"] = result
        elif ranking == 1:
            self.strongest_entanglements["4"] = self.strongest_entanglements["3"]
            self.strongest_entanglements["3"] = self.strongest_entanglements["2"]
            self.strongest_entanglements["2"] = self.strongest_entanglements["1"]
            self.strongest_entanglements["1"] = result

    # Utility function to return the ranking of an entanglement relative to the top 4 (if it is better than any)
    def rank_entanglement(self, result):
        if result[1]["f1_score"] >= self.strongest_entanglements["1"][1]["f1_score"]:
            return 1
        elif result[1]["f1_score"] >= self.strongest_entanglements["2"][1]["f1_score"]:
            return 2
        elif result[1]["f1_score"] >= self.strongest_entanglements["3"][1]["f1_score"]:
            return 3
        elif result[1]["f1_score"] >= self.strongest_entanglements["4"][1]["f1_score"]:
            return 4
        return 5

    # Utility function that creates the appropriate CNOT gates for the model out of the given input string
    def create_gates(self, entanglement):
        gates = []
        pairs = entanglement.split("-")
        for pair in pairs:
            gate = qml.CNOT(wires=[int(pair[0]), int(pair[1])])
            gates.append(gate)
        return gates

    # Generate an appropriate entanglement configuration from a given parent combination
    def generate_combination(self, parent):
        parent_pairs = parent.split("-")
        valid_pairs = [valid_pair for valid_pair in self.combinations if valid_pair not in parent_pairs]

        possible_combinations = ["-".join([parent_pairs, valid_pair]) for valid_pair in valid_pairs]
        valid_combinations = [possible_combination for possible_combination in possible_combinations if possible_combination not in self.entanglement_population]

        if len(valid_combinations) > 0:
            generated_combination = np.random.choice(valid_combinations)
            return generated_combination
        return None

    # Continue searching until the generation cap is reached
    def search(self):
        while self.generation < self.max_generations:
            print(f"\n\n ==== Starting generation {self.generation} ==== \n\n")
            parents = [value[0] for value in self.strongest_entanglements.values()]  # Set this generation's parents as the current top 4
            for i in range(self.generation_size):
                chosen_parent = np.random.choice(parents)
                combination = self.generate_combination(chosen_parent)
                if combination is None:
                    continue

                reset_keras()
                save_path = rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\DQiNN_{combination}.keras"
                entanglement_gates = self.create_gates(combination)

                model = DQiNN(TRAINING_PATH, VALIDATION_PATH, TESTING_PATH, save_path, layers=N_LAYERS, dropout=DROPOUT,
                              learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=N_EPOCHS, threshold=THRESHOLD,
                              n_qubits=N_QUBITS, entanglements=entanglement_gates)
                model.train_model()
                model.save()
                model.evaluate_model()

                result = (combination, {"loss": model.loss, "accuracy": model.accuracy, "precision": model.precision, "recall": model.recall, "f1_score": model.f1_score})
                ranking = self.rank_entanglement(result)
                if ranking != 5:
                    self.update_ranking(result, ranking)
                self.entanglement_population.append(combination)
        print("FINISHED ALL GENERATIONS OF THE GENETIC ALGORITHM...")

    # Function to display the final results of the search
    def result_summary(self):
        print(f"\n\n ==== SUMMARY OF THE STRONGEST MODELS AFTER GENETIC ALGORITHM SEARCH ==== \n\n")
        for key, value in self.strongest_entanglements.items():
            print(f"Model {key} - {value[0]} entanglement:\n\nLoss: {value[1]['loss']}\nAccuracy: {value[1]['accuracy']}\nPrecision: {value[1]['precision']}\nRecall: {value[1]['recall']}\nF1 Score: {value[1]['f1_score']}")


# ==== Parameter configuration and script initialisation ====
if __name__ == "__main__":
    reset_keras()

    # p_q = multiprocessing.Process(target=init_dqinn)
    # p_q.start()
    # p_q.join()

    ga = GeneticAlgorithm(generations=5)
    ga.initialize()
    ga.search()
    ga.result_summary()
