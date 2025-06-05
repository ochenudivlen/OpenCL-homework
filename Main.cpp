#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "CL/cl.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include "profiler.h"
#include <algorithm>
#include <random>
#include <functional>
#include <iomanip>

int checkError(int err, const char* mes)
{
    if (err != CL_SUCCESS)
        std::cout << mes << std::endl;
    return 1;
}

int main(int argc, char** argv)
{
    const size_t n = 1024;
    std::vector<double> h_a(n * n), h_b(n * n);
    profiler prof;

    //���������� ������
    std::default_random_engine engine{ std::random_device{}() };
    std::uniform_int_distribution<int> distribution{ 1, 9 };
    auto gen = std::bind(distribution, engine);
    std::generate_n(h_a.begin(), n * n, gen);

    for (int i = 0; i < n; i++)
    {
        h_b[i * n + i] = 1;
    }

    //����� ����������� ������
#if 0
    std::cout << std::setprecision(3) << std::fixed;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << h_a[i * n + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << h_b[i * n + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
#endif

    //������ � �����
    std::ifstream ifs("kernels.cl");
    //������ ��� ������� �� ������ ifs � ������� �� ��� ������ source
    std::string source((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    //���������� ��������� �� ������� ������, ��� �������� ������
    const char* ProgramSource = source.c_str();

    int err;

    cl_context       context;
    cl_command_queue commands;
    cl_program       program;
    cl_kernel        iteration;
    cl_kernel        normalization;
    cl_platform_id   platform_id;
    cl_device_id     device_id;
    cl_uint          num_platforms;
    cl_uint          num_devices;

    cl_mem d_a;
    cl_mem d_b;

    prof.tic("opencl");

    // ��������� ��������� � ������������ ����������
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    checkError(err, "Finding platforms");
    if (num_platforms == 0)
    {
        printf("Found 0 platforms!\n");
        return 1;
    }

    cl_platform_id platforms = platform_id;
    char buffer[1024];
    clGetPlatformInfo(platforms, CL_PLATFORM_NAME, 1024, buffer, NULL);
    printf("%s\n", buffer);

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (device_id == NULL)
        checkError(err, "Finding a device");
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, buffer, NULL);
    printf("%s\n\n", buffer);

    // �������� ��������������� ��������
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // �������� ������� ������
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // �������� ��������������� ������ ���������� �� ��������� ������
    program = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource, NULL, &err);
    checkError(err, "Creating program");

    // ���������� ������ ����������
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
        return 1;

    // �������� ��������������� ����
    iteration = clCreateKernel(program, "iteration", &err);
    checkError(err, "Creating kernel");

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(double) * n * n, NULL, &err);
    checkError(err, "Creating buffer d_a");

    d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * n, NULL, &err);
    checkError(err, "Creating buffer d_�");

    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(double) * n * n, h_a.data(), 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(double) * n * n, h_b.data(), 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    // ����������� ������� ������������
    size_t global[2] = { n, n };
    // ����������� ��������� ������� ������
    size_t local_work_size[2] = { 1, std::min(256, (int)n) };

    for (int k = 0; k < n; k++)
    {
        // ������ ��������� ��� ������ ��������������� ����
        err  = clSetKernelArg(iteration, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(iteration, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(iteration, 2, sizeof(int),    &n);
        err |= clSetKernelArg(iteration, 3, sizeof(int),    &k);
        checkError(err, "Setting kernel arguments");

        // ���������� ���� �� ����� ���������
        err = clEnqueueNDRangeKernel(commands, iteration, 2, NULL, global, local_work_size, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel");
    }

    //�������� ���� ��� ������������ �������� �������
    normalization = clCreateKernel(program, "normalization", &err);
    checkError(err, "Creating kernel");

    // ������ ��������� ��� ������ ����
    err  = clSetKernelArg(normalization, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(normalization, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(normalization, 2, sizeof(int),    &n);
    checkError(err, "Setting kernel arguments");

    // ���������� ���� �� ����� ���������
    err = clEnqueueNDRangeKernel(commands, normalization, 2, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // �������� ���������� ������, ������ ��� ����������� ������
    err = clFinish(commands);
    checkError(err, "Waiting for kernel to finish");

    // ������ ������������� ��������
    err = clEnqueueReadBuffer(commands, d_a, CL_TRUE, 0, sizeof(double) * n * n, h_a.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
        return 1;

    err = clEnqueueReadBuffer(commands, d_b, CL_TRUE, 0, sizeof(double) * n * n, h_b.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS)
        return 1;

    prof.toc("opencl");

    //����� ����������� ������
#if 0
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << h_a[i * n + j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << h_b[i * n + j] << "\t";
        }

        std::cout << std::endl;
    }
#endif

    // �������, ����� ����������
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseProgram(program);
    clReleaseKernel(iteration);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    prof.report();

    return 0;
}