// Implement kernels here (Note: delete sample code below)

__global__ void SDT_kernel()
{
  int tx = threadIdx.x + blockDim.x*blockIdx.x;
	tx++;
}

extern "C" void run_SDT_Kernel()
{
  SDT_kernel<<<1024, 1024>>>();
	cudaDeviceSynchronize();
}
