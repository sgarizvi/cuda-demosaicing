# cuda-demosaicing
CUDA implementation of bi-linear image demosaicing.

This application performs demosaicing of an input grayscale CFA image to an RGB image using bi-linear interpolation algorithm.
All bayer patterns are supported (i.e. `BGGR`, `RGGB`, `GRBG` and `GBRG`).

Requirements:
=
- CUDA (version 7.5 and above to support C++ 11)
- OpenCV 3 and above

Usage:
=
1. Execute make.sh as folows:

`./make.sh`
   
2.
Execute the sample code as follows:

`./example.sh`


3. To profile the code and measure execution timings, run the example_profile.sh as follows:

`./example_profile.sh`


The above mentioned steps are supposed to be executed on a supported Linux system. I have tested it in the following environment:
- Ubuntu 16.04 LTS
- CUDA 9.0
- OpenCV 4.0
