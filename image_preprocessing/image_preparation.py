import os
import cv2
import numpy as np

def crop_image(image_path, output_path, crop_size=(3000, 3000)):
    # Croping the image so we get a squared image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Calculate the top left corner for the crop. We want to crop relative to the center of the image
    # We will start the crop here
    start_x = (width - min(crop_size[1], width)) // 2
    start_y = (height - min(crop_size[0],height)) // 2
    
    # Crop the image
    cropped_image = image[start_y:start_y + crop_size[0], start_x:start_x + crop_size[1]]
    
    # Save the cropped image
    success = cv2.imwrite(output_path, cropped_image)
    if not success:
        print(f"Failed to save {image_path}.")

def resize_image(image_path, output_path, size=(640, 640)):
    # Resize the image by diminishing the pixels to be able to treat them easily in our DL pipeline
    image = cv2.imread(image_path)

    # Resizing the images
    resized_image = cv2.resize(image, size)
    
    # Saving the cropped images
    success = cv2.imwrite(output_path, resized_image)
    if not success:
        print(f"Failed to save {image_path}.")


def main(input_dir):

    # Defining the directory
    cropped_dir = 'new_images'
    resized_dir = 'resized_images2'

    # Create the directory if they do not exist
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(resized_dir, exist_ok=True)

    # # Step 1: Crop photos
    # for filename in os.listdir(input_dir):
    #     image_path = os.path.join(input_dir, filename)
    #     output_path = os.path.join(cropped_dir, filename)
    #     crop_image(image_path, output_path)
    # print('Cropping of the images done')

    # Step 2: Resize photos
    for filename in os.listdir(cropped_dir):
        print(f'Starting processing {filename}')
        image_path = os.path.join(cropped_dir, filename)
        output_path = os.path.join(resized_dir, filename)
        resize_image(image_path, output_path)
        print(f'Image saved in {output_path}')
    print('Resizing of the images done')

input_dir = r"C:\Users\camil\Documents\Projet_data\Object_recognition\input_images"

main(input_dir)