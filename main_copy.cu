#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda.h>

#define pi 3.14159265359

//Load image of M x N dimensions from path fpath
void load_image(char *fpath, int M, int N, float  *img)
{
  FILE *fp;
  
  fp=fopen(fpath,"r");
  
  for (int i=0;i<N;i++){
    for(int j=0;j<M;j++)
      fscanf(fp,"%f ",&img[i*M+j]);
     fscanf(fp,"\n");
  }
  
  fclose(fp);
}


void save_image(char *fpath, int M, int N, float  *img)
{
  FILE *fp;
  
  fp=fopen(fpath,"w");
  
  for (int i=0;i<N;i++){
    for(int j=0;j<M;j++)
      fprintf(fp,"%10.3f ",img[i*M+j]);
     fprintf(fp,"\n");
  }
  
  fclose(fp);
}

void init_matrix(float *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
  }
}

void load_kernel(int kernel_dim, float sigma, float* kernel)
{
  int total_kernel_size = kernel_dim*kernel_dim;
  float x,y, center;

  center = (kernel_dim-1)/2.0;
  
  for (int i = 0; i<total_kernel_size; i++){
    x = (float)(i%kernel_dim)-center;
    y =(float)(i/kernel_dim)-center;
    kernel[i] = -(1.0/pi*pow(sigma,4))*(1.0 - 0.5*(x*x+y*y)/(sigma*sigma))*exp(-0.5*(x*x+y*y)/(sigma*sigma));
  }
}

 __global__ 
 void convolution_gpu(float *img, float *kernel, float *imgf, int M, int N, int kernel_size)
{
  
  //thread id for each block
  int tid = threadIdx.x;    
                   
  int iy = blockIdx.x + (kernel_size - 1)/2;  
  int ix = threadIdx.x + (kernel_size - 1)/2; 
  
  //idx of pixel
  int idx = iy*M +ix;                        
 
 //kernel total size
  int kernel_total_size = kernel_size*kernel_size; 
  
  //center of kernel         
  int center = (kernel_size -1)/2;		
  
  int ii, jj;
  float sum = 0.0;
 
  //_sKernel will shared in among each block 
  extern __shared__ float _sKernel[];         

  if (tid<kernel_total_size)
    _sKernel[tid] = kernel[tid];             
  
  __syncthreads();			  
  						
  
  if (idx<M*N){
    for (int ki = 0; ki<kernel_size; ki++)
      for (int kj = 0; kj<kernel_size; kj++){
	      ii = kj + ix - center;
	      jj = ki + iy - center;
	      sum+=img[jj*M+ii]*_sKernel[ki*kernel_size + kj];
      }
  
    imgf[idx] = sum;
  }  
}



int main(){
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
float milliseconds = 0;
  int M, N;
  int kernel_size;
  float sigma;
  char finput[256], foutput[256];
  int Nblocks, Nthreads;
  
  sprintf(finput,"lux_bw.dat");
  sprintf(foutput,"lux_output.dat") ;

  M = 600;
  N = 570;

  kernel_size = 5;
  sigma = 0.8;

  float *img, *imgf, *kernel;
  
  img = (float*)malloc(M*N*sizeof(float));
  imgf = (float*)malloc(M*N*sizeof(float));
  kernel = (float*)malloc(kernel_size*kernel_size*sizeof(float));  
  
  float *d_img, *d_imgf, *d_kernel;
  
  cudaMalloc(&d_img,M*N*sizeof(float));
  cudaMalloc(&d_imgf,M*N*sizeof(float));
  cudaMalloc(&d_kernel,kernel_size*kernel_size*sizeof(float));
  
  load_image(finput, M, N, img);
  //init_matrix(kernel, kernel_size);
  load_kernel(kernel_size, sigma, kernel);

  cudaMemcpy(d_img, img, M*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel,kernel, kernel_size*kernel_size*sizeof(float),cudaMemcpyHostToDevice);

  Nblocks = N - (kernel_size-1);
  Nthreads = M - (kernel_size-1);
  
  cudaEventRecord(start);
  convolution_gpu<<<Nblocks, Nthreads, kernel_size*kernel_size*sizeof(float)>>>(d_img, d_kernel, d_imgf, M, N, kernel_size);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  cudaMemcpy(imgf, d_imgf, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  save_image(foutput, M, N, imgf);
  
  printf("\n");
  printf("Convolution Completed !!! \n");
  printf("Ellapsed Time (GPU): %16.10f ms\n", milliseconds);
  printf("\n");
  
  
  
  free(img);
  free(imgf);
  free(kernel);

  cudaFree(d_img);
  cudaFree(d_imgf);
  cudaFree(d_kernel);
}