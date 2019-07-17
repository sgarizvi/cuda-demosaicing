nvcc -o demosaic -std=c++11 src/demosaicing_app.cpp src/demosaicing_cuda.cu src/main.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc
