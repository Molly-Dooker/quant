import onnx
import ipdb

# ipdb.set_trace()

model = onnx.load('resnet18_manual_qdq_final.onnx')


with open('XXX', "w") as f:
        f.write(str(model.graph))