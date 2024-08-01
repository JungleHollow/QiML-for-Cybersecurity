import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import shutil
import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset

paths = [
    "./Datasets/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
    "./Datasets/CICIDS2017/MachineLearningCVE/Wednesday-WorkingHours.pcap_ISCX.csv"
]

fullpath = "./Datasets/CICIDS2017/MachineLearningCVE/CICIDS17_Full.csv"

with open(fullpath, "wb") as outfile:
    for i, path in enumerate(paths):
        with open(path, "rb") as infile:
            if i != 0:
                infile.readline()  # Discard the headers after the first infile
            shutil.copyfileobj(infile, outfile)
            print(f"{path} has been copied to outfile...")

all_data = pd.read_csv(fullpath)
all_data.columns = map(str.lower, all_data.columns)
all_data.columns = map(str.strip, all_data.columns)

all_data["label"] = np.where(all_data["label"] == "BENIGN", 0, 1)

all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

train = all_data.iloc[:int(all_data.shape[0] * 0.8), :]
test = all_data.iloc[int(all_data.shape[0] * 0.8):, :]
test.reset_index(drop=True, inplace=True)  # Reset the test set's indexes

del all_data

train.dropna(inplace=True)
test.dropna(inplace=True)

print(f"\nData have been split into training, and test sets...")

train_numeric = train.drop(columns=["label"], axis=1).select_dtypes(include=["number"]).columns
test_numeric = test.drop(columns=["label"], axis=1).select_dtypes(include=["number"]).columns

train[train_numeric] = train[train_numeric].apply(abs, axis=1).apply(np.log1p, axis=1)
test[test_numeric] = test[test_numeric].apply(abs, axis=1).apply(np.log1p, axis=1)

normalizer = StandardScaler()
train[train_numeric] = normalizer.fit_transform(train[train_numeric], train["label"])
test[test_numeric] = normalizer.transform(test[test_numeric])

print(f"All data have been log-transformed and scaled...")

del train_numeric, test_numeric

train.dropna(inplace=True)
test.dropna(inplace=True)

train_y = train["label"]
train.drop(columns=["label"], inplace=True)

smote = SMOTE(random_state=1337)
train, train_y = smote.fit_resample(train, train_y)

train = pd.concat([train, train_y], axis=1)

print(f"SMOTE applied to the training data...")

del train_y

train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)

train, validation = train_test_split(train, test_size=0.1, random_state=1337)

train, train_discarded = train_test_split(train, test_size=0.999, random_state=1337)
validation, validation_discarded = train_test_split(validation, test_size=0.999, random_state=1337)

print(f"Finished sampling the training set...")

del train_discarded, validation_discarded

train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

# Ensure that each dataframe is stored as float32
train = train.astype("float32")
validation = validation.astype("float32")
test = test.astype("float32")

train.to_csv("./Datasets/Cleaned Datasets/CICIDS17_training-set.csv", index=False)
validation.to_csv("./Datasets/Cleaned Datasets/CICIDS17_validation-set.csv", index=False)
test.to_csv("./Datasets/Cleaned Datasets/CICIDS17_testing-set.csv", index=False)

del train, validation, test

print(f"All csv files have been successfully written to disk...")

train_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICIDS17_training-set.csv",
                            label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

train_ds.save("./Datasets/Cleaned Datasets/CICIDS17_training-set")

val_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICIDS17_validation-set.csv",
                          label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

val_ds.save("./Datasets/Cleaned Datasets/CICIDS17_validation-set")

test_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICIDS17_testing-set.csv",
                           label_name="label", batch_size=64, shuffle=False, num_epochs=1, ignore_errors=True)

test_ds.save("./Datasets/Cleaned Datasets/CICIDS17_testing-set")

print(f"All data sets have been successfully written to disk...\n\nPROCESS FINISHED")
