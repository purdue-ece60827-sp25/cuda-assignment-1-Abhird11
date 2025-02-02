
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size) {
		y[idx] += x[idx] * scale;
	}
}

int verifyVectorWithTolerance(float* a, float* b, float* c, float scale, int size) {
	int errorCount = 0;
	for (int idx = 0; idx < size; ++idx) {
		if (abs(c[idx] - (scale * a[idx] + b[idx])) / c[idx] > 0.00001) {
			++errorCount;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << idx << " expected " << scale * a[idx] + b[idx] 
					<< " found " << c[idx] << " = " << a[idx] << " + " << b[idx] << "\n";
			#endif
		}
	}
	return errorCount;
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here

	// Allocate host memory
	float *h_a, *b, *h_c;

	h_a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	h_c = (float *) malloc(vectorSize * sizeof(float));

	if (h_a == NULL || b == NULL || h_c == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	// Initialize vectors with random floats
	for (int i = 0; i < vectorSize; i++) {
		h_a[i] = (float) rand();
		b[i] = (float) rand();
	}

	std::memcpy(h_c, b, vectorSize * sizeof(float));
	float scale = 2.1111f;

	// Allocate device memory
	float *d_a, *d_c;
	cudaMalloc((void**) &d_a, sizeof(float) * vectorSize);
	cudaMalloc((void**) &d_c, sizeof(float) * vectorSize);

	// Copy from host to device
	cudaMemcpy(d_a, h_a, sizeof(float) * vectorSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, sizeof(float) * vectorSize, cudaMemcpyHostToDevice);
	

	// TODO: Call saxpy_gpu here
	int threads_per_block = 1024;
	int num_blocks = ceil((float) vectorSize / threads_per_block);
	saxpy_gpu<<<num_blocks, threads_per_block>>>(d_a, d_c, scale, vectorSize);


	// Copy from device back to host (a is not needed since it's unchanged)
	cudaMemcpy(h_c, d_c, sizeof(float) * vectorSize, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_a);
	cudaFree(d_c);

	int error_count = verifyVectorWithTolerance(h_a, b, h_c, scale, vectorSize);
	std::cout << "Found " << error_count << " / " << vectorSize << " errors \n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadId < pSumSize) {
		curandState_t rng;
		float x, y;
		curand_init(clock64(), threadId, 0, &rng);

		pSums[threadId] = 0;
		for (int i = 0; i < sampleSize; i++) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);

			pSums[threadId] += !int(x*x + y*y);
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	for (int i = 0; i < reduceSize; i++) {
		if (threadId * reduceSize + i < pSumSize) {
			totals[threadId] += pSums[threadId * reduceSize + i];
		}
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here

	// Allocate host memory
	uint64_t *h_totals;
	h_totals = (uint64_t*) malloc(sizeof(uint64_t) * generateThreadCount);

	// Allocate device memory
	uint64_t *d_pSums, *d_totals;
	cudaMalloc((void**) &d_pSums, sizeof(uint64_t) * generateThreadCount);
	cudaMalloc((void**) &d_totals, sizeof(uint64_t) * reduceThreadCount);


	// Run generate kernel
	int threads_per_block = 1024;
	int num_blocks = ceil((float) generateThreadCount / threads_per_block);
	generatePoints<<<num_blocks, threads_per_block>>> (d_pSums, generateThreadCount, sampleSize);

	// Run reduce kernel
	num_blocks = ceil((float) reduceThreadCount / threads_per_block);
	reduceCounts<<<num_blocks, threads_per_block>>> (d_pSums, d_totals, generateThreadCount, reduceSize);

	// Copy memory from device to host
	cudaMemcpy(h_totals, d_totals, sizeof(uint64_t) * reduceThreadCount, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_pSums);
	cudaFree(d_totals);

	// Calculate PI
	for (int i = 0; i < reduceThreadCount; i++) {
		approxPi += h_totals[i];
	}
	approxPi /= generateThreadCount * sampleSize;
	approxPi *= 4.0f;

	return approxPi;
}
