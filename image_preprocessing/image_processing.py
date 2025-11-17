import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def get_bbox_coord(label_path):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            bboxes.append([class_id] + coords)
    
    return bboxes

def save_label(output_path, bboxes):
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            line = ' '.join(map(str, bbox))
            f.write(line + '\n')

def img_rotation(image, base_filename):
    # Rotate the images 3 times with different angle
    results = {}
    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180 : cv2.ROTATE_180,
        270 : cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    results[f'{base_filename}_rot0'] = image
    
    for angle, code in rotation_map.items():
        key = f'{base_filename}_rot{angle}'
        results[key] = cv2.rotate(image, code)
    
    return results

def bbox_rotation(bboxes):
    # Rotate the labels
    results = [bboxes]

    rotation_angles = [90, 180, 270]

    for angle in rotation_angles:
        rotated_bboxes = []

        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            
            if angle == 90:
                new_x = 1 - y_center
                new_y = x_center
                new_w = height
                new_h = width
                
            elif angle == 180:
                new_x = 1 - x_center
                new_y = 1 - y_center
                new_w = width
                new_h = height

            else: # angle == 270
                new_x = y_center
                new_y = 1 - x_center
                new_w = height
                new_h = width
            
            rotated_bboxes.append([class_id, new_x, new_y, new_w, new_h])
        results.append(rotated_bboxes)

    return results

def passthrough_label(bbox):
    # Passthrough function to pass the bbox without modification
    return bbox

def adjust_contrast(image, base_filename):
    # Adjusts the contrast of an image by multiplying the pixel values by the given factor.
    results = {}
    # results[f'{base_filename}_C100'] = image
    factors = [0.5, 1.5]

    for factor in factors:
        # Clip values to ensure they remain valid and stays between 0 and 255
        key = f'{base_filename}_C{int(factor*10)}'
        results[key] = np.clip(image * factor, 0, 255).astype(np.uint8)

    return results

def adjust_brightness(image, base_filename):
    results = {}
    # results[f'{base_filename}_B100'] = image
    levels = [-75, -35, 35, 75]
    
    for level in levels:
        key = f'{base_filename}_B{int(level)}'
        # Clip values to ensure they remain valid and stays between 0 and 255
        results[key] = np.clip(image.astype(np.int16) + level, 0, 255).astype(np.uint8)

    return results

def add_noise(image, base_filename):
    # Add noise to the image
    results = {}
    # results[f'{base_filename}_N0'] = image
    levels = [1, 2, 3]
    
    image_float = image.astype(np.float32)

    for level in levels:
        mean = 0
        sigma = 25 * level
        key = f'{base_filename}_N{level}'

        # The noise is a gaussian noise, centered around 0, with variance depending on the given level
        noise = np.random.normal(mean, sigma, image.shape)
        noised_image = image_float + noise

        # We clip the image to stay inside the 0-255 range
        results[key] = np.clip(noised_image, 0, 255).astype(np.uint8)

    return results
    

augmentation_function = {
    'rotated':{
        'image_func': img_rotation,
        'label_func': bbox_rotation
    },
    'contrasted': {
        'image_func': adjust_contrast,
        'label_func': passthrough_label
    },
    'brightned': {
        'image_func': adjust_brightness,
        'label_func': passthrough_label
    },
    'noised': {
        'image_func': add_noise,
        'label_func': passthrough_label
    }
}

def main():

    aug_types = ['resized', 'rotated', 'contrasted', 'brightned', 'noised', 'augmented']
    sub_folders = ['images', 'labels']

    # Create an empty dictionary to hold all paths
    paths = {}

    # Loop through all augmentation types and populate the dictionary
    for aug_type in aug_types:
        # Create a nested dictionary for each type
        paths[aug_type] = {} 
        
        for sub_folder in sub_folders:
            # Build the path
            current_path = Path(aug_type) / sub_folder
            
            # Store the Path object in the dictionary
            paths[aug_type][sub_folder] = current_path
            
            # We also create the directoy
            current_path.mkdir(parents=True, exist_ok=True)

    print("All directories created and paths stored.")

    augmentation_pipeline = ['rotated', 'contrasted', 'brightned', 'noised']

    for aug_step in augmentation_pipeline:

        function = augmentation_function[aug_step]

        if aug_step == 'rotated':
            input_image_dir = paths['resized']['images']
            input_label_dir = paths['resized']['labels']
        else:
            input_image_dir = paths['rotated']['images']
            input_label_dir = paths['rotated']['labels']

        # We set the output of the augmentation to the correct folder
        output_image_dir = paths[f'{aug_step}']['images']
        output_label_dir = paths[f'{aug_step}']['labels']

        for filename in os.listdir(input_image_dir):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            # We read the image
            image = cv2.imread(str(input_image_dir / filename))

            # We extract the name of the file without the extension
            base_filename = os.path.splitext(filename)[0]

            # We get the label corresponding to the image
            label_path = input_label_dir / f'{base_filename}.txt'
            bboxes = get_bbox_coord(label_path)

            if aug_step == 'rotated':

                # We call the image rotation function that create our 4 rotated images
                aug_images = function['image_func'](image, base_filename)
                # We save the names of the images for the labels later
                augmented_image_names = []

                for image_name, image in aug_images.items():
                    augmented_image_names.append(image_name)

                    # We save each image in the final folder along with its name
                    output_image_path = output_image_dir / f'{image_name}.jpg'
                    cv2.imwrite(output_image_path, image)

                # We call the label rotating function
                aug_labels = function['label_func'](bboxes)
                
                # We zip the bboxes and the names of the images so that we can just save the new bboxes with the image names
                for augmented_image_name, bbox in zip(augmented_image_names, aug_labels):
                    output_label_path = output_label_dir / f'{augmented_image_name}.txt'
                    save_label(output_label_path, bbox)

            # Now we do exactly the same with all the augmentation steps left. We always go from the rotated files
            else:
                aug_images = function['image_func'](image, base_filename)
                aug_labels = function['label_func'](bboxes)

                for image_name, image in aug_images.items():
                    augmented_image_names.append(image_name)

                    output_image_path = output_image_dir / f'{image_name}.jpg'
                    cv2.imwrite(output_image_path, image)

                    output_label_path = output_label_dir / f'{image_name}.txt'
                    save_label(output_label_path, aug_labels)

        print(f'{aug_step} done.')

    print("--- Collecting all files into 'augmented' folder ---")
    final_aug_img_dir = paths['augmented']['images']
    final_aug_lbl_dir = paths['augmented']['labels']

    for aug_type in aug_types[1:]:
        if aug_type == 'augmented':
            continue
        
        print(f"Copying files from {aug_type}...")
        src_image_dir = paths[aug_type]['images']
        src_label_dir = paths[aug_type]['labels']
        
        for img_file in src_image_dir.glob('*.jpg'):
            lbl_file = src_label_dir / f"{img_file.stem}.txt"
            shutil.copy(img_file, final_aug_img_dir)
            if lbl_file.exists():
                shutil.copy(lbl_file, final_aug_lbl_dir)

    print(f'Augmentation finished')

if __name__ == '__main__':
    main()