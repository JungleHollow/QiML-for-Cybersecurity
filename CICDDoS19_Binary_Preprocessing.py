import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import shutil
import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset


def index_logic(index):
    if index % 1000 == 0:
        return False
    return True


drop_cols = ["Unnamed: 0", "Flow ID", " Source IP", " Destination IP", " Destination Port", " Source Port", " Timestamp", "SimillarHTTP"]

paths = [
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_DNS.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_LDAP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_MSSQL.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_NetBIOS.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_NTP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_SNMP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_SSDP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/DrDoS_UDP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/Syn.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/TFTP.csv",
    "./Datasets/CICDDoS2019/01-12(Training)/UDPLag.csv"
]

test_paths = [
    "./Datasets/CICDDoS2019/03-11(Testing)/LDAP.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/MSSQL.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/NetBIOS.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/Portmap.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/Syn.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/UDP.csv",
    "./Datasets/CICDDoS2019/03-11(Testing)/UDPLag.csv"
]

fullpath = "./Datasets/CICDDoS2019/CICDDoS19_Full_Train.csv"
fullpath_test = "./Datasets/CICDDoS2019/CICDDoS19_Full_Test.csv"

with open(fullpath, "wb") as outfile:
    for i, path in enumerate(paths):
        with open(path, "rb") as infile:
            if i != 0:
                infile.readline()
            shutil.copyfileobj(infile, outfile)
            print(f"TRAIN - {path} has been copied to outfile...")

with open(fullpath_test, "wb") as outfile:
    for i, path in enumerate(test_paths):
        with open(path, "rb") as infile:
            if i != 0:
                infile.readline()
            shutil.copyfileobj(infile, outfile)
            print(f"TEST - {path} has been copied to outfile...")

print("Finished collating all input csv files successfully...")

train = pd.read_csv(fullpath, skiprows=lambda x: index_logic(x))
train.drop(columns=drop_cols, inplace=True)
train.columns = map(str.lower, train.columns)
train.columns = map(str.strip, train.columns)
train["label"] = np.where(train["label"] == "BENIGN", 0, 1)
train = train.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

train_numeric = train.drop(columns=["label"], axis=1).select_dtypes(include=["number"]).columns
train[train_numeric] = train[train_numeric].apply(abs, axis=1).apply(np.log1p, axis=1)

normalizer = StandardScaler()
train[train_numeric] = normalizer.fit_transform(train[train_numeric], train["label"])
train.dropna(inplace=True)

del train_numeric

print("Training set has been normalized...")

train_y = train["label"]
train.drop(columns=["label"], inplace=True)

smote = SMOTE(random_state=1337)
train, train_y = smote.fit_resample(train, train_y)

train = pd.concat([train, train_y], axis=1)

del train_y

print("Training set SMOTE has been applied successfully...")

train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)

train, validation = train_test_split(train, test_size=0.1, random_state=1337, stratify=train["label"])
train, train_discarded = train_test_split(train, test_size=0.95, random_state=1337, stratify=train["label"])
validation, validation_discarded = train_test_split(validation, test_size=0.95, random_state=1337, stratify=validation["label"])

del train_discarded, validation_discarded

print("Training and validation sets have finished being sampled...")

train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

train = train.astype("float64")
validation = validation.astype("float64")

train.to_csv("./Datasets/Cleaned Datasets/CICDDoS19_training-set.csv", index=False)
validation.to_csv("./Datasets/Cleaned Datasets/CICDDoS19_validation-set.csv", index=False)

del train, validation

print("Train and validation csv files have been successfully written to disk...")

test = pd.read_csv(fullpath_test)
test.drop(columns=drop_cols, inplace=True)
test.columns = map(str.lower, test.columns)
test.columns = map(str.strip, test.columns)
test["label"] = np.where(test["label"] == "BENIGN", 0, 1)
test = test.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

test_numeric = test.drop(columns=["label"], axis=1).select_dtypes(include=["number"]).columns
test[test_numeric] = test[test_numeric].apply(abs, axis=1).apply(np.log1p, axis=1)

test[test_numeric] = normalizer.transform(test[test_numeric])
test.dropna(inplace=True)

del test_numeric

print("Test set has been normalized...")

test, test_discarded = train_test_split(test, test_size=0.99, random_state=1337, stratify=test["label"])
test.reset_index(drop=True, inplace=True)

del test_discarded

print("Test set has been down-sampled...")

test = test.astype("float64")

test.to_csv("./Datasets/Cleaned Datasets/CICDDoS19_testing-set.csv", index=False)

del test

print("Test csv file has been successfully written to disk...")

train_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICDDoS19_training-set.csv", label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)
train_ds.save("./Datasets/Cleaned Datasets/CICDDoS19_training-set")

del train_ds

val_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICDDoS19_validation-set.csv", label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)
val_ds.save("./Datasets/Cleaned Datasets/CICDDoS19_validation-set")

del val_ds

test_ds = make_csv_dataset("./Datasets/Cleaned Datasets/CICDDoS19_testing-set.csv", label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)
test_ds.save("./Datasets/Cleaned Datasets/CICDDoS19_testing-set")

print(f"All data sets have been successfully written to disk... \n\nPROCESS FINISHED")
