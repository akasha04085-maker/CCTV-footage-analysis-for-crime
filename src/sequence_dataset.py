import os
import numpy as np
import torch


def create_sequence(folder_path, sequence_length=30):

    files = sorted(os.listdir(folder_path))

    sequence = []

    for i in range(sequence_length):

        file_path = os.path.join(folder_path, files[i])

        image = np.load(file_path)

        tensor = torch.tensor(image)

        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)

        sequence.append(tensor)

    sequence = torch.stack(sequence)

    return sequence


sequence = create_sequence("normalized_frames")

print(sequence.shape)