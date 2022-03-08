// Implement kernels here (Note: delete sample code below)
#include <stdio.h>
#include <iostream>
#include "ahe_cpu.h"

#define WIDTH 8192
#define MAP_LEN 256//(WIDTH/TILE_SIZE_X)*(WIDTH/TILE_SIZE_X)*256

__constant__ unsigned char MAP[MAP_LEN];

using namespace std;


__forceinline__ __device__ unsigned char interp(unsigned char v00, unsigned char v01, unsigned char v10, unsigned char v11, float x_frac, float y_frac)
{
  	float v0 = v00*(1 - x_frac) + v10*x_frac;
	float v1 = v01*(1 - x_frac) + v11*x_frac;
  	float v = v0*(1 - y_frac) + v1*y_frac;

	if (v < 0) v = 0;
	if (v > 255) v = 255;

	return (unsigned char)(v);
}


__global__ void adap_eq_kernel(unsigned char *img_out, unsigned char *img_in, unsigned char *mapping, int width, int height, int ntiles_x, int ntiles_y){
	// row index of thread/pixel 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// column index of thread/pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row<height && col<width){

		int tile_i0, tile_j0, tile_i1, tile_j1; // tile IDs

		tile_i0 = (row - TILE_SIZE_X/2) / TILE_SIZE_X;
		if(tile_i0 < 0) tile_i0 = 0;
		tile_j0 = (col - TILE_SIZE_Y/2) / TILE_SIZE_Y;
		if(tile_j0 < 0) tile_j0 = 0;
		tile_i1 = (row + TILE_SIZE_X/2) / TILE_SIZE_X;
		if(tile_i1 >= ntiles_x) tile_i1 = ntiles_x - 1;
		tile_j1 = (col + TILE_SIZE_Y/2) / TILE_SIZE_Y;
		if(tile_j1 >= ntiles_y) tile_j1 = ntiles_y - 1;


		// Find offsets to neighboring mappings. For no neighbors, set the nearest neighbor.
		int offset00 = 256*(tile_i0 + tile_j0*ntiles_x);
		int offset01 = 256*(tile_i0 + tile_j1*ntiles_x);
		int offset10 = 256*(tile_i1 + tile_j0*ntiles_x);
		int offset11 = 256*(tile_i1 + tile_j1*ntiles_x);


		// Compute 4 values and perform bilinear interpolation
      	unsigned char v00, v01, v10, v11;
		v00 = mapping[img_in[row+col*width] + offset00];
		v01 = mapping[img_in[row+col*width] + offset01];
		v10 = mapping[img_in[row+col*width] + offset10];
		v11 = mapping[img_in[row+col*width] + offset11];


		float x_frac = float(row - tile_i0*TILE_SIZE_X - TILE_SIZE_X/2)/float(TILE_SIZE_X);
		float y_frac = float(col - tile_j0*TILE_SIZE_Y - TILE_SIZE_Y/2)/float(TILE_SIZE_Y);

		unsigned char res = interp(v00, v01, v10, v11, x_frac, y_frac);

		img_out[row + col*width] = res;

	}
	
}


__global__ void const_adap_eq_kernel(unsigned char *img_out, unsigned char *img_in, int width, int height, int ntiles_x, int ntiles_y){
	// row index of thread/pixel 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// column index of thread/pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row<height && col<width){

		int tile_i0, tile_j0, tile_i1, tile_j1; // tile IDs

		tile_i0 = (row - TILE_SIZE_X/2) / TILE_SIZE_X;
		if(tile_i0 < 0) tile_i0 = 0;
		tile_j0 = (col - TILE_SIZE_Y/2) / TILE_SIZE_Y;
		if(tile_j0 < 0) tile_j0 = 0;
		tile_i1 = (row + TILE_SIZE_X/2) / TILE_SIZE_X;
		if(tile_i1 >= ntiles_x) tile_i1 = ntiles_x - 1;
		tile_j1 = (col + TILE_SIZE_Y/2) / TILE_SIZE_Y;
		if(tile_j1 >= ntiles_y) tile_j1 = ntiles_y - 1;


		// Find offsets to neighboring mappings. For no neighbors, set the nearest neighbor.
		int offset00 = 256*(tile_i0 + tile_j0*ntiles_x);
		int offset01 = 256*(tile_i0 + tile_j1*ntiles_x);
		int offset10 = 256*(tile_i1 + tile_j0*ntiles_x);
		int offset11 = 256*(tile_i1 + tile_j1*ntiles_x);


		// Compute 4 values and perform bilinear interpolation
      	unsigned char v00, v01, v10, v11;
		v00 = MAP[img_in[row+col*width] + offset00];
		v01 = MAP[img_in[row+col*width] + offset01];
		v10 = MAP[img_in[row+col*width] + offset10];
		v11 = MAP[img_in[row+col*width] + offset11];


		float x_frac = float(row - tile_i0*TILE_SIZE_X - TILE_SIZE_X/2)/float(TILE_SIZE_X);
		float y_frac = float(col - tile_j0*TILE_SIZE_Y - TILE_SIZE_Y/2)/float(TILE_SIZE_Y);

		unsigned char res = interp(v00, v01, v10, v11, x_frac, y_frac);

		img_out[row + col*width] = res;

	}
	
}

__global__ void calc_mapping_kernel(int* cdf, int* pdf, unsigned char *mapping, int pixels_per_tile)
{
	if(threadIdx.x==0){
		int offset = blockDim.x*blockIdx.x;

		cdf[offset] = pdf[offset];
		for(int i=offset+1; i<offset+256; i++){
			cdf[i] = cdf[i-1] + pdf[i];
		}

		int cdf_min = pixels_per_tile+1;
		for(int i=offset; i<offset+256; i++){
			if(cdf[i]!=0){
				cdf_min = cdf[i];
				break;
			}
		}

		for(int i=offset; i<offset+256; i++){
			mapping[i] = (unsigned char)round(255.0 * float(cdf[i] - cdf_min)/float(pixels_per_tile - cdf_min));
		}

	}

}

__global__ void calc_pdf_kernel(int* pdf, unsigned char* img_in, int width, int height, int ntiles_x, int ntiles_y)
{
	// row index of thread/pixel 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	// column index of thread/pixel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	if(row<height && col<width){
		int pixel_value = img_in[col + row*width];

		int tile_index_x = (int)(col/TILE_SIZE_X);
		int tile_index_y = (int)(row/TILE_SIZE_Y);
		int tileId = ntiles_x * tile_index_y + tile_index_x;


		atomicAdd(&pdf[tileId*256 + pixel_value], 1);
	}
	
}

__global__ void const_calc_mapping_kernel(int* cdf, int* pdf, int pixels_per_tile)
{
	if(threadIdx.x==0){
		int offset = blockDim.x*blockIdx.x;

		cdf[offset] = pdf[offset];
		for(int i=offset+1; i<offset+256; i++){
			cdf[i] = cdf[i-1] + pdf[i];
		}

		int cdf_min = pixels_per_tile+1;
		for(int i=offset; i<offset+256; i++){
			if(cdf[i]!=0){
				cdf_min = cdf[i];
				break;
			}
		}

		for(int i=offset; i<offset+256; i++){
			MAP[i] = (unsigned char)round(255.0 * float(cdf[i] - cdf_min)/float(pixels_per_tile - cdf_min));
		}
	}
}


extern "C" void run_sampleKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
	int pixels_per_tile = TILE_SIZE_X*TILE_SIZE_Y;
	int ntiles_x = width / TILE_SIZE_X;
	int ntiles_y = height / TILE_SIZE_Y;
	
	int img_size = width * height;
	int map_LEN = ntiles_x*ntiles_y*256;
	
	// host variables
	int *h_cdf = (int*)malloc(map_LEN*sizeof(int));
	int *h_pdf = (int*)malloc(map_LEN*sizeof(int));
	unsigned char *mapping = new unsigned char[map_LEN];
	
	//device variables
	unsigned char *d_img_in, *d_img_out;
	unsigned char *d_mapping;
	int *d_cdf;
	int *d_pdf;
	
	


	// -----allocate constant memory to MAP -----
	cudaMemcpyToSymbol(MAP, mapping, map_LEN*sizeof(unsigned char));



	// allocate memory to device img_in
	cudaMalloc((void**)&d_img_in, img_size*sizeof(unsigned char));
	cudaMemcpy(d_img_in, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	//allocate memory to device img_out
	cudaMalloc((void**)&d_img_out, img_size*sizeof(unsigned char));
	cudaMemcpy(d_img_out, img_out, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	//allocate memory to pdf array
	cudaMalloc((void**)&d_pdf, map_LEN*sizeof(int));
	cudaMemcpy(d_pdf, h_pdf, map_LEN*sizeof(int), cudaMemcpyHostToDevice);
	
	//allocate mmory to device cdf array
	cudaMalloc((void**)&d_cdf, map_LEN*sizeof(int));
	cudaMemcpy(d_cdf, h_cdf, map_LEN*sizeof(int), cudaMemcpyHostToDevice);
	
	//allocate memory to mapping
	cudaMalloc((void**)&d_mapping, map_LEN*sizeof(int));
	cudaMemcpy(d_mapping, mapping, map_LEN*sizeof(unsigned char), cudaMemcpyHostToDevice);
	


	dim3 dimGrid(1+width/32, 1+height/32);
	dim3 dimBlock(32, 32);

	calc_pdf_kernel<<< dimGrid, dimBlock >>>(d_pdf, d_img_in, width, height, ntiles_x, ntiles_y);
	cudaDeviceSynchronize();
	
	


	//-----using global mapping-----
	calc_mapping_kernel<<<1 + map_LEN/256, 256>>>(d_cdf, d_pdf, d_mapping, pixels_per_tile);
	//----using constant mapping-----
	// const_calc_mapping_kernel<<<1 + map_LEN/256, 256>>>(d_cdf, d_pdf, pixels_per_tile);
	
	cudaDeviceSynchronize();
	
	
	

	// -----using global mapping-----
	adap_eq_kernel<<<dimGrid, dimBlock>>>(d_img_out, d_img_in, d_mapping, width, height, ntiles_x, ntiles_y);
	// -----using constant mapping-----
	// const_adap_eq_kernel<<<dimGrid, dimBlock>>>(d_img_out, d_img_in, width, height, ntiles_x, ntiles_y);


	cudaDeviceSynchronize();
	
	
	// read img_out from device
	cudaMemcpy(img_out, d_img_out, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	
	// free memroy
	cudaFree(d_img_in);
	cudaFree(d_img_out);
	cudaFree(d_mapping);
	cudaFree(d_pdf);
	free(h_cdf);
	free(h_pdf);
	free(mapping);
}

