
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::string inputPath =
        "CUDA_GPU_Image_Processing/input/fruits.jpg";

    std::string outputPath =
        "CUDA_GPU_Image_Processing/output/fruits_cpu_gray.jpg";

    if (argc >= 2)
    {
        inputPath = argv[1];
    }

    if (argc >= 3)
    {
        outputPath = argv[2];
    }

    std::cout << "========================================\n";
    std::cout << "CPU Image Processing\n";
    std::cout << "========================================\n";

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

    cv::Mat outputImage(
        height,
        width,
        CV_8UC3);

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const cv::Vec3b& pixel =
                inputImage.at<cv::Vec3b>(y, x);

            unsigned char b = pixel[0];
            unsigned char g = pixel[1];
            unsigned char r = pixel[2];

            unsigned char gray =
                static_cast<unsigned char>(
                    0.299f * r +
                    0.587f * g +
                    0.114f * b);

            outputImage.at<cv::Vec3b>(y, x) =
                cv::Vec3b(gray, gray, gray);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    double cpuTime =
        std::chrono::duration<double, std::milli>(
            stop - start).count();

    if (!cv::imwrite(outputPath, outputImage))
    {
        std::cerr << "Failed to save output image.\n";
        return 1;
    }

    std::cout << "CPU grayscale processing completed.\n";
    std::cout << "CPU Processing Time: "
              << cpuTime
              << " ms\n";

    std::cout << "Output Image: "
              << outputPath
              << "\n";

    std::cout << "========================================\n";
    std::cout << "Processing completed successfully.\n";
    std::cout << "========================================\n";

    return 0;
}
