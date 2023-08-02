#!/usr/bin/env python
"""
Description: This is the file to combine the images vertically.
Author: Yueqian Lin
Date: 2023-05-15
"""

import os
from PIL import Image

# all code below is written by Yueqian Lin with the help of ChatGPT
####################################################################################################
def combine_images_vertically(folder_path, output_path, origin=False):
    images = []
    max_width = 0
    total_height = 0
    filenames = sorted(os.listdir(folder_path))
    # load all images in the folder
    for filename in filenames:
  
        if origin:
            if filename.endswith(".png") and filename.startswith("origin_result"):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                images.append(image)

                # update the maximum width
                max_width = max(max_width, image.width)

                # update the total height
                total_height += image.height
        else:
            if filename.endswith(".png") and filename.startswith("clean_result"):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                images.append(image)

                # update the maximum width
                max_width = max(max_width, image.width)

                # update the total height
                total_height += image.height
    # create a new blank image with the combined width and height
    combined_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    # paste each image onto the blank image vertically
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height
    print("Combined image size: ", combined_image.size)
    # save the combined image
    combined_image.save(output_path)

if __name__ == "__main__":
    # provide the folder path containing the images and the output path for the combined image
    folder_path = "output"
    output_path = "output/origin_combined.png"

    # call the function to combine the images vertically
    combine_images_vertically(folder_path, output_path)
