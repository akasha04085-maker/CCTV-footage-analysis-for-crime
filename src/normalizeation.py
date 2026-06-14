import cv2
import os
import numpy as np


def normalize_frames(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    for file in files:

        image_path = os.path.join(input_folder, file)

        image = cv2.imread(image_path)

        image = image.astype(np.float32)

        normalized_image = image / 255.0

        save_path = os.path.join(output_folder, file)

        np.save(save_path.replace(".jpg", ".npy"), normalized_image)

    print("Normalization completed.")

if __name__ == "__main__":
    normalize_frames("resized_frames", "normalized_frames")

