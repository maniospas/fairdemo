from fairbench.bench.loader import read_csv, features
import numpy as np


class CSVDataset:
    def __init__(self, data, numeric, categorical, labels):
        self.data = data
        self._numeric = numeric
        self._categorical = categorical
        self.labels = labels
        self.cols = numeric + categorical

    def to_features(self):
        return features(self.data, self._numeric, self._categorical).astype(np.float64)


def run(uri):
    raw_data = read_csv(
        uri,
        delimiter=";",
    )
    return CSVDataset(raw_data,
        numeric=[
            "age",
            "duration",
            "campaign",
            "pdays",
            "previous"],
        categorical=[
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "poutcome",
        ],
        labels=(raw_data["y"] != "no").values
    )