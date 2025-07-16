%%writefile nvcc_vectorAdd.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>

// Compute vector addition on GPU
__global__ void nvcc_vectorAdd(int *d_a, int *d_b, int *d_c, int N){
  // Get our global thread ID and ensure it does not exceed array size
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    d_c[idx] = d_a[idx] + d_b[idx];

}
void init_array(int *v, int N){
  for (int i = 0; i < N; i++){
    v[i] = rand()/100;
  }
}
int main() {
  // set vector size 2^10 = 1024
  int N = 1 << 10;
  size_t size = N * sizeof(int);
  int *h_a, *h_b, *h_c;
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  h_a = (int *)malloc(size);
  h_b = (int *)malloc(size);
  h_c = (int *)malloc(size);
  init_array(h_a, N);
  init_array(h_b, N);
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  printf("threads = %d, blocks = %d\n", threads, blocks);
  //call the kernel
  nvcc_vectorAdd<<<blocks, threads>>>(d_a,d_b,d_c,N);
  //nvcc_vectorAdd<<<N, 1>>>(d_a,d_b,d_c);
  cudaDeviceSynchronize();
  //copy the result back to the host
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);  
  for (int i = 0; i < N; i++){
    //printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    std::cout << h_c[i] << " = " << h_a[i] << " + " << h_b[i] << "\n";
    //assert(h_a[i] + h_b[i] == h_c[i]);
    if (h_c[i] != h_a[i] + h_b[i])
      return -1;
  }
  return 0;
}
// To compile "!nvcc -o nvcc_vectorAdd nvcc_vectorAdd.cu -lstdc++"
// To run "!./nvcc_vectorAdd"
// To Check NVCC compiler version !nvcc --version
//NVPROF is profiling process "!nvprof ./nvcc_vectorAdd"
