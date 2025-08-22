# Parallel and Distributed Computing

### Colab Setup : 
```python
# check the Nvidia CUDA compiler driver install or not on T4 GPU of colab
!nvcc --version
# installing necessary package for running cuda kernel on the colab gpu in notebook
!pip install nvcc4jupyter --quiet

# loading the package extension
%load_ext nvcc4jupyter
```

Then Run the following code to check everything is setuped and working as expected 

```python 
%%cuda
#include <stdio.h>

__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{
  helloGPU<<<1, 1>>>();
  cudaDeviceSynchronize();
  return 0;
}
```


## References and Resources
[1]. https://iaee.substack.com/p/cuda-for-machine-learning-intuitively
