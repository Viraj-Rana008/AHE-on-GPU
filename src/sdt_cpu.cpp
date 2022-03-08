#include <iostream>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "sdt_cpu.h"

#define SQRT_2 1.4142

void computeSDT_CPU(unsigned char * bitmap, float *sdt, int width, int height)
{
  //In the input image 'bitmap' a value of 255 represents edge pixel,
  // and a value of 127 represents interior.

  //Collect all edge pixels in an array
  int sz = width*height;
  int sz_edge = 0;
  for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
  int *edge_pixels = new int[sz_edge];
  for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
  std::cout<< "\t"<<sz_edge << " edge pixels in the image of size " << width << " x " << height << "\n"<<std::flush;

  //Compute the SDT
  float min_dist, dist2;
  float _x, _y;
  float sign;
  float dx, dy;
  int x, y, k;
#pragma omp parallel for collapse(2) private(x, y, _x, _y, sign, dx, dy, min_dist, dist2, k) //Use multiple CPU cores to speedup
  for(y = 0; y<height; y++) // Compute SDT using brute force method
    for(x=0; x<width; x++)
    {
      min_dist = FLT_MAX;
      for(k=0; k<sz_edge; k++)
      {
        _x = edge_pixels[k] % width;
        _y = edge_pixels[k] / width;
        dx = _x - x;
        dy = _y - y;
        dist2 = dx*dx + dy*dy;
        if(dist2 < min_dist) min_dist = dist2;
      }
      sign  = (bitmap[x + y*width] >= 127)? 1.0f : -1.0f;
      sdt[x + y*width] = sign * sqrtf(min_dist);
    }
  delete[] edge_pixels;
}
