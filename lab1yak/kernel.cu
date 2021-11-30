#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <cctype>
#include <chrono>
#include <iostream>


#define BLOCK_SIZE 32
//#define N 1024

__global__ void matMult(float* c, float* a, float* b, int size) {
	int   bx = blockIdx.x;
	int   by = blockIdx.y;
	int   tx = threadIdx.x;
	int   ty = threadIdx.y;
	float sum = 0.0f;              // Промежуточная переменная
	int   ia = size * blockDim.y * by + size * ty;   // a [i][0]
	int   ib = blockDim.x * bx + tx;

	// Умножение матриц;
	for (int k = 0; k < size; k++)
		sum += a[ia + k] * b[ib + k * size];

	//индекс вычисляемого элемента матрицы C 
	int ic = size * blockDim.y * by + blockDim.x * bx;
	c[ic + size * ty + tx] = sum;
}
//Умножение матриц послед.
void matrixMultCPU(float* A, float* B, float* C, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			C[i*n + j] = 0;
			for (int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}
void printMatrix(float* matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%4.1lf ", matrix[i * n + j]);
		}
		printf("\n");
	}
}
float* generateRandMatrix(int n, size_t sizeMatrix) {
	float* matrix = (float*)malloc(sizeMatrix);			// выделение памяти под массив
	for (int i = 0; i < n * n; i++) {
		matrix[i] = (float)rand() /100;		// заполнение массива случайными числами
	}
	return matrix;											// возврат заполненной матрицы
}
//Проверка матриц
bool checkMult(float* C1, float* C2, int n) {
	float accuracy = 1.e-6;
	for (int i = 0; i < n * n; i++) {
		if (abs(C1[i] - C2[i]) >= accuracy)

			return false;
	}
	return true;
}
// основная функция
int main(int argc, char *  argv[])
{
	char str[256], *p = str;
	bool isd = true;
	int N;
	std::cin >> str;
	while (*p)
		if (!std::isdigit(*p++))
		{
			isd = false;
			break;
		}
	if (isd)
		N = atoi(str);
	else
	{
		std::cout << "Not a number\n";
		return 0;
	}
	int numBytes = N * N * sizeof(float);

	// Выделяем память на хосте
	float * a = new float[N*N];
	float * b = new float[N*N];
	float * c = new float[N*N];
	float * c1 = new float[N*N];

	size_t sizeMat = sizeof(float)*N*N;
	a = generateRandMatrix(N, sizeMat);
	b = generateRandMatrix(N, sizeMat);
	//printMatrix(a,N);
	//printMatrix(b, N);


	// Выделяем память на устройстве 
	float * adev = NULL;
	float * bdev = NULL;
	float * cdev = NULL;

	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);

	// настраиваем число нитей и блоков
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);


	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	//Копируем массивы на девайс
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	//Вызываем функцию
	matMult << <blocks, threads >> > (cdev, adev, bdev, N);
	//копируем полученную матрицу
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);


	std::cout << "time spent executing by the GPU: " << gpuTime << " millseconds\n";

	// освобождаем память
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
	matrixMultCPU(a, b, c1, N);
	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> search_time = end_time - start_time;
	std::cout << "time spent executing by the CPU: " << search_time.count() << " millseconds\n";
	std::cout << "Acceleration: " << search_time.count() / gpuTime << std::endl;
	if (checkMult(c, c1, N))
	{
		std::cout << "Ok\n";
	}
	else
	{
		std::cout << "Wrong\n";
		printMatrix(c, N);
		printMatrix(c1, N);
	}
	delete a;
	delete b;
	delete c;

	return 0;
}