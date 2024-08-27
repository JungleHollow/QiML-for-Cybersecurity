import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import shutil
import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset
import os


def index_logic(index):
    if index % 100 == 0:
        return False
    return True


paths = [
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-14-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-15-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-16-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-21-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-22-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-23-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-28-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\03-01-2018.csv",
    r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\03-02-2018.csv",
]

problem_path = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\02-20-2018.csv"

fullpath = r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\CICIDS2018\CICIDS18_Full.csv"

if not os.path.isfile(fullpath):
    with open(fullpath, "wb") as outfile:
        for i, path in enumerate(paths):
            with open(path, "rb") as infile:
                if i != 0:
                    infile.readline()  # Discard the headers after the first infile
                shutil.copyfileobj(infile, outfile)
                print(f"{path} has been copied to outfile...")

    all_data = pd.read_csv(fullpath)

    all_data = all_data.drop(columns=["Timestamp", "Dst Port"], axis=1)
    all_data.columns = map(str.lower, all_data.columns)
    all_data.columns = map(str.strip, all_data.columns)

    all_data["label"] = np.where(all_data["label"] == "Benign", 0, 1)

    for column in all_data.columns:
        all_data[column] = pd.to_numeric(all_data[column], errors="coerce")

    all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    all_data = all_data.astype("float64")

    problem_data = pd.read_csv(problem_path)
    problem_data.drop(columns=["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"], axis=1, inplace=True)

    problem_data.columns = map(str.lower, problem_data.columns)
    problem_data.columns = map(str.strip, problem_data.columns)

    problem_data["label"] = np.where(problem_data["label"] == "Benign", 0, 1)

    problem_data = problem_data.astype("float64")

    all_data = pd.concat([all_data, problem_data], axis=0)
    all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    del problem_data

    all_data.to_csv(fullpath, index=False)
else:
    all_data = pd.read_csv(fullpath, skiprows=lambda x: index_logic(x))

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

train, train_discarded = train_test_split(train, test_size=0.98, random_state=1337)
validation, validation_discarded = train_test_split(validation, test_size=0.98, random_state=1337)

print(f"Finished sampling the training set...")

del train_discarded, validation_discarded

train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

# Ensure that each dataframe is stored as float64
train = train.astype("float64")
validation = validation.astype("float64")
test = test.astype("float64")

train.to_csv(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_training-set.csv", index=False)
validation.to_csv(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_validation-set.csv", index=False)
test.to_csv(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_testing-set.csv", index=False)

del train, validation, test

print(f"All csv files have been successfully written to disk...")

train_ds = make_csv_dataset(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_training-set.csv",
                            label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

train_ds.save(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_training-set")

val_ds = make_csv_dataset(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_validation-set.csv",
                          label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

val_ds.save(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_validation-set")

test_ds = make_csv_dataset(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_testing-set.csv",
                           label_name="label", batch_size=64, shuffle=False, num_epochs=1, ignore_errors=True)

test_ds.save(r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\Datasets\Cleaned Datasets\CICIDS18_testing-set")

print(f"All data sets have been successfully written to disk...\n\nPROCESS FINISHED")
