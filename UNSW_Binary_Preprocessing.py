import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.data.experimental import make_csv_dataset

ip_cols = ["srcip", "dstip", "sport", "dsport"]


dfs = []
for i in range(1, 5):
    path = fr"E:\HonoursThesis\Code\Datasets\UNSW_NB15\UNSW-NB15_{i}.csv"
    tmp = pd.read_csv(path, header=None)
    dfs.append(tmp)

all_data = pd.concat(dfs).reset_index(drop=True)

df_col = pd.read_csv(r"E:\HonoursThesis\Code\Datasets\UNSW_NB15\UNSW-NB15_features.csv", encoding="ISO-8859-1")
df_col["Name"] = df_col["Name"].apply(lambda x: x.strip().replace(" ", "").lower())

all_data.columns = df_col["Name"]

all_data["id"] = range(1, len(all_data.index) + 1)  # To allow for the possibility of easily including indexes in training...
all_data.drop(ip_cols, axis=1, inplace=True)

all_data["attack_cat"] = all_data.attack_cat.fillna(value="normal").apply(lambda x: x.strip().lower())
all_data["ct_flw_http_mthd"] = all_data.ct_flw_http_mthd.fillna(value=0)
all_data["is_ftp_login"] = (all_data.is_ftp_login.fillna(value=0)).astype(int)
all_data["is_ftp_login"] = np.where(all_data["is_ftp_login"] >= 1, 1, 0)
all_data["ct_ftp_cmd"] = all_data.ct_ftp_cmd.replace(to_replace=" ", value=0).astype(int)
all_data["service"] = all_data["service"].apply(lambda x: "None" if x == "-" else x)
all_data["attack_cat"] = all_data["attack_cat"].replace("backdoors", "backdoor", regex=True).apply(lambda x: x.strip().lower())

all_data = all_data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
all_data = all_data.replace("-", None).dropna(axis=0)

all_data_numeric = all_data.drop(columns=["label", "id"], axis=1).select_dtypes(include=["number"]).columns
all_data[all_data_numeric] = all_data[all_data_numeric].apply(abs, axis=1).apply(np.log1p)

del df_col, all_data_numeric

for col in ["proto", "service", "state"]:
    ohe = OneHotEncoder(handle_unknown="ignore")

    all_data_cols = ohe.fit_transform(all_data[col].values.reshape(-1, 1))

    all_data_df = pd.DataFrame(all_data_cols.todense(), columns=[col + "_" + str(i) for i in ohe.categories_[0]])

    all_data.drop(columns=[col], axis=1, inplace=True)

    all_data = pd.concat([all_data, all_data_df], axis=1)

all_data.dropna(inplace=True)

train = all_data.iloc[:int(all_data.shape[0] * 0.8), :]
test = all_data.iloc[int(all_data.shape[0] * 0.8):, :]
test.reset_index(drop=True, inplace=True)  # Reset the test set's indexes to prevent dataframe issues when onehot encoding

del dfs, all_data

train.dropna(inplace=True)
test.dropna(inplace=True)

train_numeric = train.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns
test_numeric = test.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns

normalizer = Normalizer()
train[train_numeric] = normalizer.fit_transform(train[train_numeric], train["label"])
test[test_numeric] = normalizer.transform(test[test_numeric])

train.dropna(inplace=True)
test.dropna(inplace=True)

train_y = train["label"]
train.drop(columns=["label", "attack_cat", "id"], inplace=True)

oversample = SMOTE(random_state=1337)
train, train_y = oversample.fit_resample(train, train_y)

train = pd.concat([train, train_y], axis=1)

del train_y

train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)

train, validation = train_test_split(train, test_size=0.1, random_state=1337)

train, train_discarded = train_test_split(train, test_size=0.999, random_state=1337)
validation, validation_discarded = train_test_split(validation, test_size=0.999, random_state=1337)

del train_discarded, validation_discarded

train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

test = test.drop(["attack_cat", "id"], axis=1)

# Ensure that each dataframe is stored as float32
train = train.astype("float32")
validation = validation.astype("float32")
test = test.astype("float32")

train.to_csv(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_training-set.csv", index=False)
validation.to_csv(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_validation-set.csv", index=False)
test.to_csv(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_testing-set.csv", index=False)

del train, validation, test

train_ds = make_csv_dataset(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_training-set.csv",
                            label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

train_ds.save(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_training-set")

val_ds = make_csv_dataset(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_validation-set.csv",
                          label_name="label", batch_size=64, shuffle=True, num_epochs=1, ignore_errors=True)

val_ds.save(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_validation-set")

test_ds = make_csv_dataset(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_testing-set.csv",
                           label_name="label", batch_size=64, shuffle=False, num_epochs=1, ignore_errors=True)

test_ds.save(r"E:\HonoursThesis\Code\Datasets\Cleaned Datasets\UNSW_NB15_testing-set")
