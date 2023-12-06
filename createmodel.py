"""
Run this file to train and save an sklearn model in onnx format.
"""

from skl2onnx import to_onnx
from sklearn.linear_model import LogisticRegression
from components import csv_dataset_loader

dataset = csv_dataset_loader.run("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv")

model = LogisticRegression(max_iter=1000)
x = dataset.to_features()
y = dataset.labels
model.fit(x, y)

with open("model.onnx", "wb") as f:
    f.write(to_onnx(model, x[:1], options={'zipmap': False}).SerializeToString())
