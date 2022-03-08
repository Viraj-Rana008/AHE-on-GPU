#ifndef _AHE_CPU_H_
#define _AHE_CPU_H_

#include "defines.h"

void adaptiveEqualizationCPU(unsigned char* img_in, unsigned char* img_out, int width, int height);
unsigned char interp2(unsigned char v00, unsigned char v01, unsigned char v10, unsigned char v11, float x_frac, float y_frac);

#endif
