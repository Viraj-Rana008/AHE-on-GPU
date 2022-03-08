#include <cuda_runtime.h>
#include <iostream>

#include "sdt_gpu.h"

extern "C" void run_SDT_Kernel();

void computeSDT_GPU(unsigned char * bitmap, float *sdt, int width, int height)
{
  run_SDT_Kernel();
}

