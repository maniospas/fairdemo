
# select dataset
from components import csv_dataset_loader as dataset_loader
dataset_uri = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
dataset = dataset_loader.run(dataset_uri)

# select sensitive attributes based on dataset and domain
print("All column names", dataset.cols)  # select sensitive attributes from these
sensitive_attributes = ["marital"]

# select model
model_path = 'model.onnx'
from components import onnx_model_loader as model_loader  # automatically detect based on extension
model = model_loader.run(model_path)

# one of the analyses to compute
from components import fairbench_report_analysis as fairness_analysis

# run analysis
analysis_outcome = fairness_analysis.run(model, dataset, sensitive_attributes)

# we need to determine outcome formats (e.g., at first, we could say that
# every output should be an html that also uses local images saved programmatically throughout
# the analysis)
print(analysis_outcome)

