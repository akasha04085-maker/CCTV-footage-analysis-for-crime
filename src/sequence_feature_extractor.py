import torch
from feature_extractor import FeatureExtractor

class SequenceFeatureExtractor(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.feature_extractor = FeatureExtractor()
        
    def forward(self, x):
        
        batch_size, seq_length, c, h, w = x.shape
        
        # convert
        x = x.reshape(batch_size * seq_length, c, h, w)

        print("Reshaped:", x.shape)

        features = self.feature_extractor(x)

        print("Feature output:", features.shape)

        features = features.reshape(
            batch_size,
            seq_length,
            -1
        )

        print("Final output:", features.shape)

        return features