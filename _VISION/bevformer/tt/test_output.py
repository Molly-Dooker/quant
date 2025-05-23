import ttnn

device = ttnn.open_device(0)
tensor = ttnn.load_tensor(file_name="ttnn_output.pth", device=device)
print(tensor.mean())