import torch

class BiLSTMModel(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        
        output, _ = self.lstm(x)
        
        return output