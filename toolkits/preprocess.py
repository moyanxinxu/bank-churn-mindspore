import pandas as pd
from config import config
from sklearn.cluster import KMeans
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import trange

label_encoder = LabelEncoder()
min2max_scaler = MinMaxScaler()
kmeans = KMeans(n_clusters=10)
bayes = CategoricalNB()

random_state = config["random_state"]
unuse_columns = config["unuse_columns"]
min2max_columns = config["min2max_columns"]
label_encode_columns = config["label_encode_columns"]


def datapipe(df: pd.DataFrame, flag="Train"):
    df = df.drop(unuse_columns, axis=1)
    df["Balance"] = df["Balance"].apply(lambda x: 1 if x != 0 else 0)

    # K-means
    df[min2max_columns] = min2max_scaler.fit_transform(df[min2max_columns])
    # bayes
    df[label_encode_columns] = df[label_encode_columns].apply(
        lambda series: label_encoder.fit_transform(series)
    )

    if flag == "Train":
        export = pd.DataFrame()

        model = kmeans.fit(df[min2max_columns])
        export["f1"] = model.labels_
        export = export.join(df[label_encode_columns]).join(df[["Balance", "Exited"]])

        dev = export.sample(frac=0.2, random_state=random_state)
        train = export.drop(dev.index)

        train.to_csv(config["train_set"], index=False)
        dev.to_csv(config["dev_set"], index=False)
    else:
        export = pd.DataFrame()

        model = kmeans.fit(df[min2max_columns])
        export["f1"] = model.labels_
        export = export.join(df[label_encode_columns]).join(df[["Balance"]])

        export.to_csv(config["test_set"], index=False)


def detect_k(df: pd.DataFrame, k_range: tuple):
    min, max = k_range
    df = df[min2max_columns]
    distance = []
    for k in trange(min, max + 1):
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(df)
        distance.append(model.inertia_**0.5)
    return distance


if __name__ == "__main__":
    df = pd.read_csv(config["test_csv"])
    datapipe(df, flag="test")
