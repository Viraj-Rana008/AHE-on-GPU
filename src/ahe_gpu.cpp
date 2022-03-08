#include "ahe_gpu.h"
#include <cuda_runtime.h>

#include <iostream>

extern "C" void run_sampleKernel();

void adaptiveEqualizationGPU(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  run_sampleKernel(); // Remove me!
}
