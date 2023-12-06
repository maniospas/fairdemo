from components import onnx_model_loader as model_loader
from components import csv_dataset_loader as dataset_loader

dataset_uri = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
model_path = 'model.onnx'

# this will happen within the execution engine
test_dataset = dataset_loader.run(dataset_uri)
model = model_loader.run(model_path)

predictions = model(test_dataset.to_features())

print(predictions.sum())
