# CUDA GPU-Accelerated Image Processing

## Project Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA and OpenCV. The main task is grayscale conversion of color images.

The project contains two implementations:

- CPU implementation using standard C++ loops.
- GPU implementation using a CUDA kernel where image pixels are processed in parallel.

The GPU implementation transfers the image from CPU memory to GPU memory, processes pixels using CUDA threads, and transfers the processed image back to the CPU.

## Technologies Used

- C++
- NVIDIA CUDA
- CUDA Runtime API
- OpenCV
- Make
- NVIDIA GPU

## GPU Hardware Used

The project was tested using:

- GPU: NVIDIA Tesla T4
- Compute Capability: 7.5
- CUDA Compiler: NVCC

## How CUDA Is Used

Each image pixel is processed independently, which makes grayscale conversion suitable for parallel execution.

The CUDA kernel assigns image pixels to GPU threads using a 2D grid and 16 x 16 thread blocks.

For each pixel, the color channels are converted into a grayscale value.

The GPU execution measures the complete GPU processing pipeline:

1. Host-to-device memory transfer
2. CUDA grayscale kernel execution
3. Device-to-host memory transfer

## Project Structure

```text
CUDA_GPU_Image_Processing/
├── Makefile
├── README.md
├── src/
│   ├── cpu_image_processor.cpp
│   └── gpu_image_processor.cu
├── input/
│   ├── fruits.jpg
│   └── fruits_2048.jpg
├── output/
│   ├── fruits_cpu_gray.jpg
│   ├── fruits_gpu_gray.jpg
│   ├── fruits_2048_cpu_gray.jpg
│   └── fruits_2048_gpu_gray_v2.jpg
└── results/
    └── performance.txt
