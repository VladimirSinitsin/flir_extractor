# FLIR Extractor
## Installation
```
cd flir_extractor
pip install -r requirements.txt
```

for Ubuntu:
```
sudo apt install libimage-exiftool-perl
```

## Structure
1) `fe_tools` contains the main functions for extracting thermal data from the FLIR .seq files
2) `demo_fff.py` is a demo script for extracting thermal data from a .fff file
3) `demo_seq.py` is a demo script for extracting thermal data from a .seq file
4) `demo_gray.py` is a demo script (like `demo_seq.py`) for extracting thermal data from a .seq file, but it saves the thermal data as a grayscale image
5) `get_all_temperture.py` is a demo script for extracting all temperature values from .seq files in a folder
6) `seq_to_tiff.py` is a script for converting .seq files to .tiff images
7) `seq_to_jpg.py` is a script for converting .seq files to .jpg grayscale images
8) `bin` contains `examples` and `exiftool.exe` for Windows

## Usage
The same for `seq_to_tiff.py` and `seq_to_jpeg.py`

- first argument: input .seq file
- second argument: minimum recording temperature (in Celsius)
- third argument: maximum recording temperature (in Celsius)
- optional argument (`--celsius`): if recording was in Celsius (default is Kelvin)
- optional argument (`--debug`): if you want to create all directories anew each time

**Examples:**
```
seq_to_tiff.py SEQ_0936.seq 200 1000
seq_to_jpeg.py SEQ_0004.seq 0 500 --celsius
```
