#!/usr/bin/env python
"""
Description: This is the file to pre-process the image.
Author: Yueqian Lin
Date: 2023-05-15
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# all code below is written by Yueqian Lin with the help of ChatGPT
####################################################################################################
def extract_lines(image_path, threshold=1):
    # load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # invert the image
    inverted_image = cv2.bitwise_not(image)

    # apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

    # binarize the image using adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # apply the Hough Line Transform
    min_line_length = 0.75 * image.shape[1]
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=min_line_length, maxLineGap=15)

    # filter lines based on angle (keep only horizontal lines)
    horizontal_lines = [line for line in lines if abs(line[0][3] - line[0][1]) < 10]

    # sort the lines by their vertical position
    sorted_lines = sorted(horizontal_lines, key=lambda line: line[0][1])

    # calculate differences between consecutive line positions
    y_diffs = np.diff([line[0][1] for line in sorted_lines])
    
    # calculate the minimum distance between consecutive lines
    y_diff_min = np.min(y_diffs)

    # combine the lines that are close to each other by average (if their distance is less than y_diff_min * threshold)
    combined_lines = []
    current_line = sorted_lines[0]
    line_count = 1
    for i, diff in enumerate(y_diffs):
        if diff <= y_diff_min * threshold:
            current_line = np.add(current_line, sorted_lines[i + 1])
            line_count += 1
        else:
            combined_lines.append(current_line[0] / line_count)
            current_line = sorted_lines[i + 1]
            line_count = 1

    # append the last line
    combined_lines.append(current_line[0] / line_count)

    return combined_lines

def draw_image(staffs, image_path):
    # load the image
    raw_image = cv2.imread(image_path)

    # draw the staffs
    for idx, staff in enumerate(staffs):
        x1, y1, x2, y2 = staff.astype(int)
        cv2.line(raw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # convert the image to RGB and show it
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    plt.imshow(raw_image)
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.axis('off')
    plt.show()

def remove_staff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # horizontal kernel
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=2)
    # find contours
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255,255,255), 2)
    # repair image
    rep_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, rep_kernel, iterations=1)
    
    return result

def segment(image_path, threshold=1, epsilon= None, output_dir= None, remove_staffs= False):
    lines = extract_lines(image_path, threshold)
    # split by 5
    staffs = [lines[i:i + 5] for i in range(0, len(lines), 5)]
    if epsilon is None:
        epsilon = np.median(np.diff([line[1] for line in lines])) * 3
    # save the images
    raw_image = cv2.imread(image_path)
    if remove_staffs:
        raw_image = remove_staff(raw_image)
    # create the output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx, staff in enumerate(staffs):
            if remove_staffs:
                filename = os.path.join(output_dir, f"{idx}.png")
            else:
                filename = os.path.join(output_dir, f"{idx}_origin.png")
            first_line_y = int(staff[0][1] - epsilon)
            last_line_y = int(staff[-1][1] + epsilon)
            x_start = np.mean([line[0] for line in staff]).astype(int) + 10
            x_end = np.mean([line[2] for line in staff]).astype(int) + 10
            cv2.imwrite(filename, raw_image[first_line_y:last_line_y, x_start:x_end])
            # save the new line location in txt
            segmented_staff = np.array(staff) - np.array([0, first_line_y, 0, first_line_y])
            segmented_staff[:, 0] = 10
            segmented_staff[:, 2] = x_end - x_start
            np.savetxt(filename.replace(".png", ".txt"), segmented_staff, fmt="%d")

    return staffs

# test case
if __name__ == "__main__":
    image_path = "test.png"
    staffs = segment(image_path, threshold=1, epsilon=None, output_dir= "temp_output", remove_staffs= False)
