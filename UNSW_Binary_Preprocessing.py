import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from imblearn.over_sampling import SMOTE

corr_cols = ["sloss", "dloss", "dpkts", "dwin", "ltime", "ct_srv_dst", "ct_src_dport_ltm", "ct_dst_src_ltm"]
log_cols = ["dur", "sbytes", "dbytes", "sload", "dload", "spkts", "stcpb", "dtcpb", "smeansz", "dmeansz", "sjit", "djit"]

dfs = []
for i in range(1, 5):
    path = f"./Datasets/UNSW_NB15/UNSW-NB15_{i}.csv"
    dfs.append(pd.read_csv(path, header=None))

all_data = pd.concat(dfs).reset_index(drop=True)

df_col = pd.read_csv("./Datasets/UNSW_NB15/UNSW-NB15_features.csv", encoding="ISO-8859-1")
df_col["Name"] = df_col["Name"].apply(lambda x: x.strip().replace(" ", "").lower())

all_data.columns = df_col["Name"]

all_data["id"] = range(1, len(all_data.index) + 1)  # To allow for the possibility of easily including indexes in training...
all_data_y = all_data["label"]
all_data.drop(columns=["srcip", "sport", "dstip", "dsport"], axis=1, inplace=True)  # Remove the src and dst ip addresses
all_data.drop(columns=corr_cols, axis=1, inplace=True)  # Remove all highly correlated features

all_data["attack_cat"] = all_data.attack_cat.fillna(value="normal").apply(lambda x: x.strip().lower())
all_data["ct_flw_http_mthd"] = all_data.ct_flw_http_mthd.fillna(value=0)
all_data["is_ftp_login"] = (all_data.is_ftp_login.fillna(value=0)).astype(int)
all_data["is_ftp_login"] = np.where(all_data["is_ftp_login"] > 1, 1, all_data["is_ftp_login"])
all_data["ct_ftp_cmd"] = all_data.ct_ftp_cmd.replace(to_replace=" ", value=0).astype(int)
all_data["service"] = all_data["service"].apply(lambda x: "None" if x == "-" else x)
all_data["attack_cat"] = all_data["attack_cat"].replace("backdoors", "backdoor", regex=True).apply(lambda x: x.strip().lower())
all_data[log_cols].apply(np.log1p)

del df_col

train = all_data.iloc[:int(all_data.shape[0] * 0.8), :]
test = all_data.iloc[int(all_data.shape[0] * 0.8):, :]
test.reset_index(drop=True, inplace=True)  # Reset the test set's indexes to prevent dataframe issues when onehot encoding

del dfs, all_data, all_data_y

train.dropna(inplace=True)
test.dropna(inplace=True)

train_numeric = train.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns
test_numeric = test.drop(columns=["id", "label"], axis=1).select_dtypes(include=["number"]).columns

normalizer = Normalizer()
train[train_numeric] = normalizer.fit_transform(train[train_numeric], train["label"])
test[test_numeric] = normalizer.transform(test[test_numeric])

train.dropna(inplace=True)
test.dropna(inplace=True)

for col in ["proto", "service", "state"]:
    ohe = OneHotEncoder(handle_unknown="ignore")

    train_cols = ohe.fit_transform(train[col].values.reshape(-1, 1))
    test_cols = ohe.transform(test[col].values.reshape(-1, 1))

    train_df = pd.DataFrame(train_cols.todense(), columns=[col + "_" + str(i) for i in ohe.categories_[0]])
    test_df = pd.DataFrame(test_cols.todense(), columns=[col + "_" + str(i) for i in ohe.categories_[0]])

    train = pd.concat([train.drop(col, axis=1), train_df], axis=1)
    test = pd.concat([test.drop(col, axis=1), test_df], axis=1)

train.dropna(inplace=True)
test.dropna(inplace=True)

train_final = train.iloc[:int(train.shape[0] * 0.9), :]
val = train.iloc[int(train.shape[0] * 0.9):, :]

del train

train_final.dropna(inplace=True)
val.dropna(inplace=True)
test.dropna(inplace=True)

train_final.sort_values(by=["id"], inplace=True)
val.sort_values(by=["id"], inplace=True)
test.sort_values(by=["id"], inplace=True)

oversample = SMOTE(random_state=1337)
train_final, train_final_y = oversample.fit_resample(train_final.drop(["label", "attack_cat", "id"], axis=1), train_final["label"])

train_final = pd.concat([train_final, train_final_y], axis=1)
val = val.drop(["attack_cat", "id"], axis=1)
test = test.drop(["attack_cat", "id"], axis=1)

del train_final_y

train_final.to_csv("./Datasets/UNSW_NB15/UNSW_NB15_binary_training-set.csv", index=False)
val.to_csv("./Datasets/UNSW_NB15/UNSW_NB15_binary_validation-set.csv", index=False)
test.to_csv("./Datasets/UNSW_NB15/UNSW_NB15_binary_testing-set.csv", index=False)
