# testtensort.py
import torch
import torch_tensorrt

print(f"PyTorch version: {torch.__version__}")
print(f"torch_tensorrt version: {torch_tensorrt.__version__}")

# Add a simple test to ensure it works
try:
    # A simple example from torch_tensorrt documentation
    # This might require a dummy model to run, but just importing should be enough for now.
    print("Successfully imported torch_tensorrt.")
    # You might try a simple conversion if you have a dummy model
    # model = torch.nn.Linear(10, 10).cuda()
    # trt_ts_module = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 10))], enabled_precisions={torch.float})
    # print("Successfully compiled a dummy model.")
except Exception as e:
    print(f"An error occurred during torch_tensorrt usage: {e}")