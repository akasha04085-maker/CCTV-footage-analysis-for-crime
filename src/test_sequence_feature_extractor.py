import torch
from sequence_feature_extractor import SequenceFeatureExtractor

print("Creating model...")
model = SequenceFeatureExtractor()

print("Creating input...")
dummy_input = torch.randn(
    1,
    2,
    3,
    224,
    224
)

print("Running model...")
output = model(dummy_input)

print("Done!")
print(output.shape)