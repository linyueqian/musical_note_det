#!/usr/bin/env python
"""
Description: This is the file to find pitches from an image.
Author: Yueqian Lin
Date: 2023-05-15
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import math

# all code below is written by Yueqian Lin with the help of ChatGPT
####################################################################################################
class Note(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h
        self.attr = None
        self.pitch = None
        self.duration = None
        self.cut = 0
        self.head = None

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x +
                        other.w) - max(self.x, other.x))
        overlap_y = max(0, min(self.y + self.h, other.y +
                        other.h) - max(self.y, other.y))
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return Note(x, y, w, h)

    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w), (int)(self.y + self.h))
        cv2.rectangle(img, pos, size, color, thickness)
        if self.pitch is not None:
            cv2.putText(img, self.pitch, ((int)(self.x), (int)(self.y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)
        if self.duration is not None:
            cv2.putText(img, self.duration, ((int)(self.x), (int)(
                self.y + self.h)+15), cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness, cv2.LINE_AA)

    def getCenter(self):
        return self.middle

    def getImage(self, img):
        return img[round(self.y):round(self.y+self.h), round(self.x):round(self.x+self.w)]

    def __str__(self):
        return ("Note: {}, x: {}, y: {}, w: {}, h: {}".format(self.attr, self.x, self.y, self.w, self.h))


def resize(template, dy, multiplier):
    # calculate new size based on dy and multiplier, keeping the aspect ratio constant
    new_size = (int((template.shape[1] * int(multiplier * dy)
                     ) / template.shape[0]), int(multiplier * dy))
    # resize template
    resized_template = cv2.resize(
        template, new_size, interpolation=cv2.INTER_AREA)

    return resized_template


def detect_clef(image, staff):
    # delta y
    dy = np.median(np.diff([line[1] for line in staff]))

    # load clef templates
    bass_clef = cv2.imread('symbol/clef/bass_clef.png', cv2.IMREAD_GRAYSCALE)
    bass_clef1 = cv2.imread('symbol/clef/bass_clef1.png', cv2.IMREAD_GRAYSCALE)
    treble_clef = cv2.imread(
        'symbol/clef/treble_clef.png', cv2.IMREAD_GRAYSCALE)
    treble_clef1 = cv2.imread(
        'symbol/clef/treble_clef1.png', cv2.IMREAD_GRAYSCALE)

    # resize templates
    bass_clef = resize(bass_clef, dy, 3.4)
    treble_clef = resize(treble_clef, dy, 7)

    clefs = {'bass': [bass_clef, bass_clef1],
             'treble': [treble_clef, treble_clef1]}

    # iterate through all clefs
    for clef in clefs:
        result = note_match(image, clefs[clef], 50, 150, 0.8)
        result = merge_notes(result, 0.5)

        if len(result) == 1:
            print('Clef detected:', clef)
            result[0].attr = clef
            return result[0]
    print('Clef not detected')
    return None


def find_non_zero_intervals(arr, threshold=0):
    intervals = []
    start = None

    # iterate through all elements in the array
    for i in range(len(arr)):
        if arr[i] != 0 or (start is not None and i - start <= threshold):
            if start is None:
                start = i
        elif start is not None:
            intervals.append((start, i - 1))
            start = None

    # check if there is an open interval at the end of the array
    if start is not None:
        intervals.append((start, len(arr) - 1))

    return intervals


def distance(note1, note2):
    return note2[0] - note1[0]


def find_note_position(binary_image, start_x=0):
    notes = []
    black = np.zeros(binary_image.shape[1])

    # count the number of black pixels in each column
    for y in range(0, binary_image.shape[0]):
        for x in range(start_x, binary_image.shape[1]):
            if binary_image[y, x] == 255:
                black[x] += 1

    # set the threshold
    upper_threshold = 0.85 * np.max(black)
    lower_threshold = 0

    # apply the threshold and find the intervals
    new_black = [i if i < upper_threshold and i >
                 lower_threshold else 0 for i in black]
    intervals = find_non_zero_intervals(new_black)

    # crop the image according to the intervals
    for interval in intervals:
        note_image = binary_image[:, interval[0]-1:interval[1]+1]

        # if the image is too small, skip
        if note_image.shape[1] < 5:
            continue

        # project the image to the y-axis
        projection = np.sum(note_image, axis=1)

        # find the first non-zero index
        start = 0
        for i in range(len(projection)):
            if projection[i] != 0:
                start = i
                break

        # find the last non-zero index
        end = len(projection)
        for i in range(len(projection)-1, -1, -1):
            if projection[i] != 0:
                end = i
                break

        # crop the image
        note_image = note_image[start-1:end+1, :]

        # check if the image is empty
        if np.sum(note_image) == 0:
            continue
        # create Note(x, y, w, h)
        height = end - start
        note = Note(interval[0]-1, start-1, interval[1] -
                    interval[0] + 2, height + 2)
        # check if the note's dimension is the same as the note_image
        note.attr = 'note'
        notes.append(note)
    print('Number of notes detected:', len(notes))

    # calculate the distance between each note
    distances = []
    for i in range(len(notes)-1):
        distances.append(notes[i].distance(notes[i+1]))
    min_dist = np.min(distances) + 2
    if min_dist > 30:
        min_dist = 30

    # combine if within min_dist
    new_notes = []
    i = 0
    while i < len(notes):
        if i == len(notes) - 1:
            new_notes.append(notes[i])
            break
        if distances[i] <= min_dist + 5:
            new_notes.append(
                Note(notes[i].x, notes[i].y, notes[i].w + notes[i+1].w, notes[i].h))
            i += 2
        else:
            new_notes.append(notes[i])
            i += 1
    print('Number of notes after combining:', len(new_notes))

    # save the notes
    for note in new_notes:
        cv2.imwrite('note/' + str(note.x) + '.png',
                    cv2.bitwise_not((note.getImage(binary_image))))
    return new_notes


def detect_time(image, debug=False):
    times = ['24', '34', '44', '68']

    # iterate through all time signatures
    for i in times:
        time_img = cv2.imread('symbol/time/' + str(i) +
                              '.tif', cv2.IMREAD_GRAYSCALE)
        res = note_match(image, [time_img], 20, 220, 0.5)
        res = merge_notes(res, 0.3)
        if debug:
            print(res)
        if len(res) == 1:
            print('Time signature detected:', i)
            res[0].attr = 'time_'+str(i)
            return res[0]
    print('Time not detected')
    return None


def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return image


def detect_accidental(image):
    accidentals = ['flat', 'sharp']

    # iterate through all accidentals
    for i in accidentals:
        accidental_img = cv2.imread(
            'symbol/accidental/' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        res = note_match(image, [accidental_img], 80, 200, 0.7)
        res = merge_notes(res, 0.7)
        if len(res) == 1:
            print('Accidental detected:', i)
            res[0].attr = 'accidental_'+str(i)
            return res[0]
    print('Accidental not detected')
    return None


def detect_notehead(image, debug=False):
    half0 = load_image('symbol/notehead/half0.tif')
    half1 = load_image('symbol/notehead/half1.tif')
    quarter0 = load_image('symbol/notehead/quarter0.tif')
    quarter1 = load_image('symbol/notehead/quarter1.tif')
    quarter2 = load_image('symbol/notehead/quarter2.tif')
    quarter3 = load_image('symbol/notehead/quarter3.tif')
    whole0 = load_image('symbol/notehead/whole0.png')
    whole1 = load_image('symbol/notehead/whole1.png')
    rest_quarter = load_image('symbol/rest/rest_quarter.png')
    noteheads = {'rest_quarter': [rest_quarter], 'half': [half0, half1], 'quarter': [
        quarter0, quarter1, quarter2, quarter3], 'whole': [whole0, whole1]}

    for i in noteheads:
        res = note_match(image, noteheads[i], 80, 150, 0.7)
        res = merge_notes(res, 0.7)
        if debug:
            print(res)
        if len(res) == 1:
            print('Notehead detected:', i)
            # print(res[0].x, res[0].y)
            res[0].attr = str(i)
            return res[0]

    print('Notehead not detected')
    return None


def find_pitches(binary_image, notes, staff, clef='treble', threshold=1, debug=False):
    real_notes = []

    # iterate through all notes
    for note in notes:
        note = find_pitch(binary_image, note, staff, clef, threshold, debug)
        if note is None:
            continue
        real_notes.append(note)
    return real_notes


def get_unit_width(notes):
    # extract the widths of all notes
    widths = [note.w for note in notes]

    # count the occurrences of each width
    width_counts = Counter(widths)

    # get the most common width
    most_common_width = width_counts.most_common(1)[0][0]
    return most_common_width


def cut(notes, unit_width):
    cut_notes = []

    # iterate through all notes
    for note in notes:
        width = note.w

        # conditions to cut
        if width < 0.5 * unit_width:
            continue
        elif width < 1.5 * unit_width:
            cut_notes.append(note)
            continue
        elif width < 3 * unit_width:
            cut = 2
        elif width < 4.3 * unit_width:
            cut = 3
        elif width < 6.2 * unit_width:
            cut = 4
        else:
            cut = round(width / unit_width)
        print('Find bar note, cut into {} parts'.format(cut))

        # find unit
        unit = round(width / cut)

        # cut the note
        for i in range(cut - 1):
            new_box = Note(
                round(note.x + i * unit),
                note.y,
                unit,
                note.h)
            new_box.cut = cut
            cut_notes.append(new_box)
        new_box = Note(
            round(note.x + (cut - 1) * unit),
            note.y,
            width - (cut - 1) * unit,
            note.h
        )
        new_box.cut = cut
        cut_notes.append(new_box)

    return cut_notes


def calculate_dur(notes):
    notes_new = []

    # iterate through all notes
    for note in notes:
        dur_text = note.head.attr
        if 'quarter' in dur_text:
            dur = 0.25
        elif 'half' in dur_text:
            dur = 0.5
        elif 'whole' in dur_text:
            dur = 1.0
        else:
            dur = 0.0  # Default duration for unrecognized notes
        # cut = np.float32(note.cut)
        if note.cut != 0:
            # [Note] temp fix after Xingyu's sugguestion, only applicable to one bar scenario
            # [Original] dur = dur/cut
            dur = 0.125
        note.duration = "{:.3f}".format(dur)
        notes_new.append(note)
    return notes_new


def find_img_pitch(img_path, out_dir='output', time=False, origin=False):
    # open one image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # read staff from txt file
    staff = np.loadtxt(img_path.split('.')[0] + '.txt')
    inverted_image = cv2.bitwise_not(image)

    # apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

    # binarize the image using adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    notes = find_note_position(binary_image)
    binary_image = cv2.bitwise_not(binary_image)

    # detect clef and time signature
    clef = detect_clef(binary_image, staff)
    if clef is not None:
        clef_info = clef.attr
    else:
        clef_info = 'treble'
    time = None
    if time:
        time = detect_time(binary_image)
    if time is not None:
        start_x = time.x
        time_info = time.attr  # time signature has not been used yet
    elif clef is not None:
        start_x = clef.x
    else:
        start_x = 0

    # find notes
    new_notes = []
    print('Start_x', start_x)
    for note in notes:
        if note.x <= start_x + 1:
            continue
        else:
            new_notes.append(note)
    notes = sorted(new_notes, key=lambda note: note.x)
    print('Notes after time:', len(notes))

    # cut notes
    unit = get_unit_width(notes)
    if unit > 30 or unit < 20:
        unit = 25
    print('Unit width:', unit)
    cut_notes = cut(notes, unit)
    print('Notes after cut:', len(cut_notes))
    print('Clef:', clef_info)
    cut_notes = find_pitches(binary_image, cut_notes,
                             staff, clef_info, threshold=3, debug=False)
    cut_notes = calculate_dur(cut_notes)

    # extract durations and pitches
    durs = [note.duration for note in cut_notes]
    pitchs = [note.pitch for note in cut_notes]

    # combine durations and pitches
    data = np.column_stack((durs, pitchs))

    # define the file path for saving
    file_path = out_dir + '/det_result' + img_path.split('.')[0][-1] + '.txt'
    # save data to text file
    np.savetxt(file_path, data, delimiter='|', fmt='%s')

    # draw the result
    if origin:
        image = load_image(img_path[:-4] + '_origin.png')
    else:
        image = load_image(img_path)
    for note in cut_notes:
        note.draw(image, (0, 0, 255), 2)
    if origin:
        cv2.imwrite(out_dir + '/origin_result' +
                    img_path.split('.')[0][-1]+'.png', image)
    else:
        cv2.imwrite(out_dir + '/clean_result' +
                    img_path.split('.')[0][-1]+'.png', image)

    return cut_notes


def match(img, templates, start_percent, stop_percent, threshold):
    # initialize best match
    best_location_count = -1
    best_locations = []
    best_scale = 1

    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 3)]:
        locations = [[], []]
        location_count = 0

        for template in templates:
            if (scale*template.shape[0] > img.shape[0] or scale*template.shape[1] > img.shape[1]):
                continue

            # resize template
            template = cv2.resize(template, None, fx=scale,
                                  fy=scale, interpolation=cv2.INTER_CUBIC)

            # perform template matching
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            result = np.where(result >= threshold)
            location_count += len(result[0])
            locations[0] += list(result[0])
            locations[1] += list(result[1])

        # check if this scale is better than the previous best
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
        # do nothing if this scale is worse than the previous best
        elif (location_count < best_location_count):
            pass

    return best_locations, best_scale


def note_match(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations = []

    # check if no matches were found
    if len(locations[0]) == 0:
        return img_locations
    for i in range(len(templates)):
        x, y = locations[1], locations[0]
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale

        # create a list of Note objects
        img_locations += [Note(x0, y0, w, h) for (x0, y0) in zip(x, y)]
    return img_locations


def merge_notes(notes, threshold):
    filtered_notes = []

    # iterate through all notes
    while len(notes) > 0:
        r = notes.pop(0)
        notes.sort(key=lambda box: box.distance(r))
        merged = True

        # attempt to merge the current box with all other notes
        while merged:
            merged = False
            i = 0

            # iterate through all notes
            while i < len(notes):
                if (
                    r.overlap(notes[i]) > threshold
                    or notes[i].overlap(r) > threshold
                ):
                    r = r.merge(notes.pop(i))
                    merged = True
                elif notes[i].distance(r) > r.w / 2 + notes[i].w / 2:
                    break
                else:
                    i += 1

        filtered_notes.append(r)

    return filtered_notes


def find_pitch(img, note, staff, clef='treble', threshold=1, debug=False):
    if debug:
        plt.imshow(note.getImage(img))
        plt.show()
    notehead = detect_notehead(note.getImage(img))
    if notehead is None:
        return None
    elif 'rest' in notehead.attr:
        note.head = notehead
        note.pitch = 'NA'
        return note
    else:
        note.head = notehead
        note_center_y = notehead.getCenter()[1] + note.y

    clef_info = {
        "treble": [("F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4"), (5, 3), (4, 2)],
        "bass": [("A3", "G3", "F3", "E3", "D3", "C3", "B2", "A2", "G2"), (3, 5), (2, 4)]
    }
    note_names = ["C", "D", "E", "F", "G", "A", "B"]
    dy = np.median(np.diff([line[1] for line in staff]))
    lines = [[int(line[1] - threshold), int(line[1] + threshold)]
             for line in staff]
    line_one, line_two, line_three, line_four, line_five = lines
    # Check within staff first
    if (line_one[0] <= note_center_y <= line_one[1]):
        note.pitch = clef_info[clef][0][0]
        return note
    elif (line_one[-1] <= note_center_y <= line_two[0]):
        note.pitch = clef_info[clef][0][1]
        return note
    elif (line_two[0] <= note_center_y <= line_two[1]):
        note.pitch = clef_info[clef][0][2]
        return note
    elif (line_two[-1] <= note_center_y <= line_three[0]):
        note.pitch = clef_info[clef][0][3]
        return note
    elif (line_three[0] <= note_center_y <= line_three[1]):
        note.pitch = clef_info[clef][0][4]
        return note
    elif (line_three[-1] <= note_center_y <= line_four[0]):
        note.pitch = clef_info[clef][0][5]
        return note
    elif (line_four[0] <= note_center_y <= line_four[1]):
        note.pitch = clef_info[clef][0][6]
        return note
    elif (line_four[-1] <= note_center_y <= line_five[0]):
        note.pitch = clef_info[clef][0][7]
        return note
    elif (line_five[0] <= note_center_y <= line_five[1]):
        note.pitch = clef_info[clef][0][8]
        return note
    else:
        if (note_center_y < line_one[0]):
            # check above staff
            line_below = line_one
            # go to next line above
            current_line = [int(pixel - dy) for pixel in line_one]
            # the octave number at line one
            octave = clef_info[clef][1][0]
            # line one's pitch has this index in note_names
            note_index = clef_info[clef][1][1]

            while (current_line[0] > 0):
                current_line = [int(pixel) for pixel in current_line]
                if (current_line[0] <= note_center_y <= current_line[1]):
                    # grab note two places above
                    octave = octave + \
                        1 if (note_index + 2 >= 7) else octave
                    note_index = (note_index + 2) % 7
                    note.pitch = note_names[note_index] + str(octave)
                    return note
                elif (current_line[-1] <= note_center_y <= line_below[0]):
                    # grab note one place above
                    octave = octave + \
                        1 if (note_index + 1 >= 7) else octave
                    note_index = (note_index + 1) % 7
                    note.pitch = note_names[note_index] + str(octave)
                    return note
                else:
                    # check next line above
                    octave = octave + \
                        1 if (note_index + 2 >= 7) else octave
                    note_index = (note_index + 2) % 7
                    line_below = current_line.copy()
                    current_line = [
                        pixel - dy for pixel in current_line]
            assert False, "Note was above staff, but not found"

        elif (note_center_y > line_five[-1]):
            # check below staff
            line_above = line_five
            # go to next line above
            current_line = [
                int(pixel + dy) for pixel in line_five]
            # the octave number at line five
            octave = clef_info[clef][2][0]
            # line five's pitch has this index in note_names
            note_index = clef_info[clef][2][1]

            while (current_line[-1] < img.shape[0]):
                if (current_line[0] <= note_center_y <= current_line[1]):
                    # grab note two places above
                    octave = octave - \
                        1 if (note_index - 2 <= 7) else octave
                    note_index = (note_index - 2) % 7
                    note.pitch = note_names[note_index] + str(octave)
                    return note
                elif (line_above[-1] <= note_center_y <= current_line[0]):
                    # grab note one place above
                    octave = octave - \
                        1 if (note_index - 1 >= 7) else octave
                    note_index = (note_index - 1) % 7
                    note.pitch = note_names[note_index] + str(octave)
                    return note
                else:
                    # check next line above
                    octave = octave - \
                        1 if (note_index - 2 <= 7) else octave
                    note_index = (note_index - 2) % 7
                    line_above = current_line.copy()
                    current_line = [
                        int(pixel + dy) for pixel in current_line]
            assert False, "Note was below staff, but not found"
        else:
            return None


if __name__ == "__main__":
    for i in range(0, 10):
        find_img_pitch('output/'+str(i)+'.png')
