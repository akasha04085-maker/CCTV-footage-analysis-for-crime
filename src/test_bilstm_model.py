import torch
from bilstm_model  import BiLSTMModel

print("Creating BiLSTM...")
model = BiLSTMModel()

print("Creating input...")
dummy_input = torch.randn(
    1,
    30,
    1280
)

print("Running...")
output = model(dummy_input)

print("Done!")
print(output.shape)