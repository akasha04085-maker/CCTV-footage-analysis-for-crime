import torch
from feature_extractor import FeatureExtractor

model = FeatureExtractor()

dummy_input = torch.randn(1, 3, 244, 244)

output = model(dummy_input)

print(output.shape)