CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++14
NVCCFLAGS = -std=c++14

OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

CPU_SOURCE = src/cpu_image_processor.cpp
GPU_SOURCE = src/gpu_image_processor.cu

CPU_EXEC = cpu_image_processor
GPU_EXEC = gpu_image_processor

INPUT_IMAGE = input/fruits.jpg

CPU_OUTPUT = output/fruits_cpu_gray.jpg
GPU_OUTPUT = output/fruits_gpu_gray.jpg

.PHONY: all cpu gpu clean run-cpu run-gpu

all: cpu gpu

cpu:
	$(CXX) $(CXXFLAGS) $(CPU_SOURCE) -o $(CPU_EXEC) $(OPENCV_FLAGS)

gpu:
	$(NVCC) $(NVCCFLAGS) $(GPU_SOURCE) -o $(GPU_EXEC) $(OPENCV_FLAGS)

clean:
	rm -f $(CPU_EXEC) $(GPU_EXEC)

run-cpu: cpu
	./$(CPU_EXEC) $(INPUT_IMAGE) $(CPU_OUTPUT)

run-gpu: gpu
	./$(GPU_EXEC) $(INPUT_IMAGE) $(GPU_OUTPUT)
