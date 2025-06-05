#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void iteration(
	__global double* A,
	__global double* I,
	int n,
	int k)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	double ratio = 0;

	if (i != k)
	{
		if (A[k * n + k] != 0)
		{
			ratio = A[i * n + k] / A[k * n + k];
		}

		A[i * n + j] -= A[k * n + j] * ratio;
		I[i * n + j] -= I[k * n + j] * ratio;
	}
}

__kernel void normalization(
	__global double* A,
	__global double* I,
	int n)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	I[i * n + j] /= A[i * n + i];
	A[i * n + i] /= A[i * n + i];
}
