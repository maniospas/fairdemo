import fairbench as fb


def run(model, dataset, sensitive_attributes):
    # error checks
    for attr in sensitive_attributes:
        if attr not in dataset.categorical:
            raise Exception("Fairness analysis not supported on non-categorical attributes")
    # declare sensitive attributes
    labels = dataset.labels
    sensitive = fb.Fork({attr: fb.categories@dataset.data[attr] for attr in sensitive_attributes})
    # obtain predictions
    predictions = model(dataset.to_features())
    report = fb.multireport(predictions=predictions, labels=labels, sensitive=sensitive)
    return report
