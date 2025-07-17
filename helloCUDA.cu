%%cuda
#include<stdio.h>

__global__ void hello_cuda(){
    printf("Hello CUDA\n");
}

int main(){
    hello_cuda<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}

//!nvcc --version
//!pip install nvcc4jupyter
//!nvcc --version
//%load_ext nvcc4jupyter
