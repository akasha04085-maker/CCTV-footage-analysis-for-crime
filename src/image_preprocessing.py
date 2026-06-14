import cv2
import os

def resize_frames(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    
    files = os.listdir(input_folder)
    
    for file in files:
        
        image_path = os.path.join(input_folder, file)
        
        image = cv2.imread(image_path)
        
        resized_image = cv2.resize(image, (244, 244))
        
        save_path = os.path.join(output_folder, file)
        
        cv2.imwrite(save_path, resized_image)
        
    print("All image resized successfully.")