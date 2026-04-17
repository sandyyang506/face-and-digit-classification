'''
Parse image into numpy arrays (features) and integer labels
Digits: 28 width by 20 height
Faces: 60 width by 68 height
'''
import numpy as np
import os

def load_images(file_path, width, height):
    all_images = []
    
    with open(file_path, 'r') as f:
        content = f.read().splitlines()

    line_pointer = 0
    total_lines = len(content)

    while line_pointer < total_lines:
        
        # Scanning to find the next block
        if content[line_pointer].strip() == "":
            line_pointer = line_pointer + 1
            continue 

        end_of_block = line_pointer + height
        
        # Safety check
        if end_of_block > total_lines:
            break
            
        block = content[line_pointer : end_of_block]
        
        # Convert block in numbers
        pixels = []
        for line in block:
            # Safety check for consistent width
            standard_line = line.ljust(width)
            
            for char in standard_line[:width]:
                if char == '#' or char == '+':
                    pixels.append(1)
                else:
                    pixels.append(0)
    
        all_images.append(np.array(pixels))
        line_pointer = end_of_block
            
    return np.array(all_images)

def load_labels(file_path):
    labels_list = []
    
    with open(file_path, 'r') as f:
        for line in f.readlines():
            clean_line = line.strip()
            if len(clean_line) > 0:
                label_number = int(clean_line)
                labels_list.append(label_number)
                
    return np.array(labels_list)

def get_dataset(data_type, split):
    base_path = "data/"
    
    if data_type == 'digit':
        width = 28
        height = 20 
        folder = os.path.join(base_path, "digitdata")
        img_file = split + "images"
        lbl_file = split + "labels"
    else: 
        width = 60
        height = 68 
        folder = os.path.join(base_path, "facedata")
        img_file = "facedata" + split
        lbl_file = "facedata" + split + "labels"

    full_img_path = os.path.join(folder, img_file)
    full_lbl_path = os.path.join(folder, lbl_file)

    img = load_images(full_img_path, width, height)
    lbl = load_labels(full_lbl_path)
    
    num_images = len(img)
    num_labels = len(lbl)
    
    if num_images != num_labels:
        limit = min(num_images, num_labels)
        img = img[:limit]
        lbl = lbl[:limit]
        
    return img, lbl