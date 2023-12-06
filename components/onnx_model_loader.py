import onnxruntime as rt
import numpy as np


def run(path: str):
    with open(path, "rb") as f:
        model_bytes = f.read()
    sess = rt.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    def predict(x):
        return sess.run([label_name], {input_name: x.astype(np.float64)})[0]
    return predict
