import mindspore as ms
import pandas as pd
from mindspore.dataset import GeneratorDataset


class train_dataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.X = self.data.iloc[:, :-1]
        self.Y = self.data.iloc[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values
        y = self.Y.iloc[idx]
        return ms.tensor(x, ms.float32), ms.tensor(y, ms.int32)


class test_dataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return ms.tensor(item.values, ms.float32)


def churn_dataset(csv_path, flag, shuffle=True):
    if flag != "test":
        dataset = train_dataset(csv_path)
        data = GeneratorDataset(dataset, ["features", "labels"], shuffle=shuffle)
    else:
        dataset = test_dataset(csv_path)
        data = GeneratorDataset(dataset, ["features"], shuffle=False)
    return data
