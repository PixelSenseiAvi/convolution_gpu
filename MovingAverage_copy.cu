#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <random>

#define TOTAL_INPUT_SIZE 10000
#define SAMPLE_SIZE 3

#define THREADS_PER_BLOCK 512

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//TOCHECK: Using int
__global__ void CalculateSMA_Shared(int* input, int input_size, int* result, int result_size, int sample_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < input_size){

        /*Shared memory. Size passed in with kernel parameters*/
        extern __shared__ int cache[];

        int cachedDataSize = sample_size + blockDim.x;

        /*Copy the data that will be used by the block into shared memory using all threads in the block.*/
        for (int i = 0; i < cachedDataSize/blockDim.x+1; i++){
            int cacheId = threadIdx.x+ i*blockDim.x;
            if (cacheId < cachedDataSize && cacheId+blockDim.x *blockIdx.x < input_size)
                cache[cacheId] = input[cacheId+blockDim.x *blockIdx.x];
        }
        __syncthreads();

        /*compute the sum using shared memory*/
        int sum = 0;
        for (int i = 0; i < sample_size; i++){
            if(i + threadIdx.x < cachedDataSize && i + idx < input_size)
                sum += cache[i+threadIdx.x];
        }
        sum /= sample_size;

        /*store in global memory*/
        if (idx < result_size)
            result[idx] = sum;
    }

}


__global__ void normalize_vec(int* arr, int* result, int MAX)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    result[idx] = arr[idx] / MAX;
}


int main()
{
  printf("Step 00 ");
    //HOST variables
    int* input_arr;
    int* result_arr;

    //Generate random numbers
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist50(0,50); 

    input_arr = (int*)(malloc(TOTAL_INPUT_SIZE*sizeof(int)));

    //Loading variables on CPU
    for(size_t i = 0; i < TOTAL_INPUT_SIZE; i++)
    {
        input_arr[i] = dist50(rng);
    }

    printf("Step 01");

    int RESULT_SIZE = TOTAL_INPUT_SIZE - SAMPLE_SIZE +1;
    result_arr = (int*)malloc(sizeof(int)*(RESULT_SIZE));

    printf("Step 1");
    //DEVICE variables
    int* device_input;
    int* device_result;

    gpuErrchk(cudaMalloc((void **)&device_input, TOTAL_INPUT_SIZE*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_result, RESULT_SIZE*sizeof(int)));

    printf("Step 2");
    cudaMemcpy(device_input, input_arr, TOTAL_INPUT_SIZE*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, result_arr, RESULT_SIZE*sizeof(int),cudaMemcpyHostToDevice);

    printf("Step 3");
    int shared_memory_allocation_size = sizeof(int)*(THREADS_PER_BLOCK+SAMPLE_SIZE);
    CalculateSMA_Shared<<<RESULT_SIZE/ THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK, shared_memory_allocation_size>>>(input_arr, TOTAL_INPUT_SIZE, result_arr, RESULT_SIZE, SAMPLE_SIZE);
    cudaDeviceSynchronize();

    //Copy result back to host
    cudaMemcpy(result_arr, device_result, RESULT_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Results copied back to HOST");

/*
    int* normal_vec;
    normal_vec = (int*)malloc(sizeof(int)*(RESULT_SIZE));

    int* device_normal;
    gpuErrchk(cudaMalloc((void **)&device_normal, RESULT_SIZE*sizeof(int)));
    cudaMemcpy(device_normal, normal_vec, RESULT_SIZE*sizeof(int), cudaMemcpyHostToDevice);

    normalize_vec<<<RESULT_SIZE/ THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(device_result, device_normal, 50);

    //Copy result back to host
    cudaMemcpy(normal_vec, device_normal, RESULT_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    free(input_arr);
    free(result_arr);
   // free(normal_vec);

    cudaFree(device_input);
    cudaFree(device_result);
   // cudaFree(device_normal);

   */
}
