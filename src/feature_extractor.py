import torch
import torchvision.models as models

class FeatureExtractor(torch.nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        
        self.features = model.features
        
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        
        x = self.features(x)
        print("Entering EfficientNet...")
        
        x = self.pool(x)
        print("Feature shape:", x.shape)
        
        x = torch.flatten(x, 1)
        print("Flatten shape:", x.shape)
        
        return x