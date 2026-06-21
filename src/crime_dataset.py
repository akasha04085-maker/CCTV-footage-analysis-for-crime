import os
import numpy as np
import torch
from torch.utils.data import Dataset 

class CrimeDataset(Dataset):
    
    def __init__(self, folder_path, sequence_length=30):
        self.folder_path = folder_path
        self.sequence_length = sequence_length
        self.files = sorted(os.listdir(folder_path))
        
    def __len__(self):
        
        return len(self.files) - self.sequence_length
    
    def __getitem__(self, idx):
        
        sequence = []
        
        for i in range(idx, idx + self.sequence_length):
            
            file_path = os.path.join(
                self.folder_path,
                self.files[i]
            )
            
            image = np.load(file_path)
            
            tensor = torch.tensor(image)
            
            tensor = tensor.permute(2, 0, 1)
            
            sequence.append(tensor)
            
        sequence = torch.stack(sequence)
        
        label = 0 
        
        return sequence, label