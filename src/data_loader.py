from torch.utils.data import DataLoader
from crime_dataset import CrimeDataset

if __name__ == "__main__":

    dataset = CrimeDataset("normalized_frames")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    for sequences, labels in loader:
    
        print("Sequence shape :", sequences.shape)
        
        print("Labels shape :", labels.shape)
        
        break