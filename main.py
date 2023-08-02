#!/usr/bin/env python
"""
Description: This is the main file to run the program.
Author: Yueqian Lin
Date: 2023-05-15
"""

import os
import shutil
import pre_process
import find_pitches
import combine
import argparse

# all code below is written by Yueqian Lin with the help of ChatGPT
####################################################################################################
def process_music_sheet(image_path, temp_folder='temp_output', output_dir='output', threshold=1, epsilon=None, time=False, origin=False):
    # provide the path to the image containing the music sheet
    image_path = image_path

    # remove temp_output files
    for filename in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) # Remove file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Remove folder
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # pre-process the image
    pre_process.segment(image_path, threshold=threshold, epsilon=epsilon, output_dir=temp_folder, remove_staffs=False)
    pre_process.segment(image_path, threshold=threshold, epsilon=epsilon, output_dir=temp_folder, remove_staffs=True)

    # loop through all the images in the folder
    for filename in os.listdir(temp_folder):
        if filename.endswith(".png") and not filename.endswith("origin.png"):
            # find pitches
            find_pitches.find_img_pitch(os.path.join(temp_folder, filename), out_dir=output_dir, time=time, origin=origin)

    # combine images vertically
    combine.combine_images_vertically(output_dir,'result.png', origin=origin)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process music sheet image')
    parser.add_argument('image_path', help='Path to the image containing the music sheet')
    parser.add_argument('--temp_folder', default='temp_output', help='Temporary folder path')
    parser.add_argument('--output_dir', default='output', help='Output directory path')
    parser.add_argument('--threshold', type=int, default=1, help='Segmentation threshold')
    parser.add_argument('--epsilon', type=int, default=None, help='Segmentation epsilon')
    parser.add_argument('--time', action='store_true', default=False, help='Include time during pitch finding')
    parser.add_argument('--origin', action='store_true', default=False, help='Include origin image during pitch finding')
    
    args = parser.parse_args()
    # create folder if not exist
    if not os.path.exists(args.temp_folder):
        os.makedirs(args.temp_folder)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    process_music_sheet(args.image_path, temp_folder=args.temp_folder, output_dir=args.output_dir,
                        threshold=args.threshold, epsilon=args.epsilon, time=args.time, origin=args.origin)