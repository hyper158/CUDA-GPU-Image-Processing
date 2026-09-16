
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(error)            \
                      << std::endl;                           \
            return 1;                                         \
        }                                                     \
    } while (0)

__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel = (y * width + x) * 3;

        unsigned char b = input[pixel];
        unsigned char g = input[pixel + 1];
        unsigned char r = input[pixel + 2];

        unsigned char gray = static_cast<unsigned char>(
            0.299f * r +
            0.587f * g +
            0.114f * b);

        output[pixel] = gray;
        output[pixel + 1] = gray;
        output[pixel + 2] = gray;
    }
}

int main(int argc, char* argv[])
{
    std::string inputPath =
        "CUDA_GPU_Image_Processing/input/fruits.jpg";

    std::string outputPath =
        "CUDA_GPU_Image_Processing/output/output.png";

    if (argc >= 2)
    {
        inputPath = argv[1];
    }

    if (argc >= 3)
    {
        outputPath = argv[2];
    }

    std::cout << "========================================\n";
    std::cout << "CUDA GPU Image Processing Project\n";
    std::cout << "========================================\n";

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable GPU found.\n";
        return 1;
    }

    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(
        &deviceProperties, 0));

    std::cout << "GPU: "
              << deviceProperties.name << "\n";

    std::cout << "Compute Capability: "
              << deviceProperties.major << "."
              << deviceProperties.minor << "\n";

    cv::Mat inputImage = cv::imread(
        inputPath,
        cv::IMREAD_COLOR);

    if (inputImage.empty())
    {
        std::cerr << "Could not open input image: "
                  << inputPath << "\n";
        return 1;
    }

    if (inputImage.channels() != 3)
    {
        std::cerr << "Input image must have 3 color channels.\n";
        return 1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    std::cout << "Input Image: "
              << width << " x "
              << height << "\n";

    size_t imageBytes =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        3 *
        sizeof(unsigned char);

    cv::Mat outputImage(
        height,
        width,
        CV_8UC3);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_input,
        imageBytes));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        imageBytes));

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing the complete GPU processing pipeline.
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        inputImage.data,
        imageBytes,
        cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(
        d_input,
        d_output,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        outputImage.data,
        d_output,
        imageBytes,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTotalTime = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &gpuTotalTime,
        start,
        stop));

    if (!cv::imwrite(outputPath, outputImage))
    {
        std::cerr << "Failed to save output image.\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);

        return 1;
    }

    std::cout << "CUDA grayscale processing completed.\n";

    std::cout << "GPU Total Time: "
              << gpuTotalTime
              << " ms\n";

    std::cout << "Output Image: "
              << outputPath
              << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "========================================\n";
    std::cout << "Processing completed successfully.\n";
    std::cout << "========================================\n";

    return 0;
}
