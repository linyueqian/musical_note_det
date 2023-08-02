# Musical Note Detection
This is a repository for the course project of COMPI302: Computer Vision.
### Author: Yueqian Lin
### Instructor: Prof. Matthias Schroeter, Ph.D.


## How to Run
1. Put the image to be processed in the same directory as `main.py`.
2. Run `python main.py test.png --temp_folder 'temp_folder' --output_dir 'output' --threshold 1--origin origin`.
3. The result will be saved as `result.png` in the same directory. The pitches and durations of the notes will be printed in the terminal and saved as `result.txt` in the `output` directory.

## File Structure
```
.
├── README.md
├── combine.py
├── find_pitches.py
├── main.py
├── pre_process.py
├── slides.pdf
├── symbol
│   ├── accidental
│   │   ├── flat.png
│   │   └── sharp.png
│   ├── clef
│   │   ├── bass_clef.png
│   │   ├── bass_clef1.png
│   │   ├── treble_clef.png
│   │   └── treble_clef1.png
│   ├── notehead
│   │   ├── half0.tif
│   │   ├── half1.tif
│   │   ├── quarter0.tif
│   │   ├── quarter1.tif
│   │   ├── quarter2.tif
│   │   ├── quarter3.tif
│   │   ├── whole0.png
│   │   └── whole1.png
│   ├── rest
│   │   └── rest_quarter.png
│   └── time
│       ├── 24.tif
│       ├── 34.tif
│       ├── 44.tif
│       └── 68.tif
└── test.png
```

## File Description
- `main.py`: The main program.
- `pre_process.py`: Pre-process the image into batches of single staffs.
- `find_pitches.py`: Find the pitches and durations of the notes for each staff.
- `combine.py`: Combine the batches of single staffs into a whole piece of music.
