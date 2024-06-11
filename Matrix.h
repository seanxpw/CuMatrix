#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

#include <thrust/fill.h>
#include <thrust/device_vector.h>

#define TILE_SIZE 16
enum DataPlace
{
    HOST,
    DEVICE
};
template <typename T>
__global__ void matAdd(int mat_sz, const T *A, const T b, T *C);

template <typename T>
__global__ void matAdd(int mat_sz, const T *A, const T *B, T *C);

template <typename T>
class Matrix
{
private:
    size_t totalSize;
    size_t dim1, dim2, dim3;
    T *data;
    DataPlace dataPlace;

    void initialize();
    void initialize(bool isOnDevice = true);
    void allocateMemory();
    void freeMemory();

    // Helper method for broadcasting
    Matrix<T> broadcastTo(size_t new_dim1, size_t new_dim2, size_t new_dim3, int axis = -1) const;

public:
    Matrix(size_t d1, bool isOnDevice = true);
    Matrix(size_t d1, size_t d2, bool isOnDevice = true);
    Matrix(size_t d1, size_t d2, size_t d3, bool isOnDevice = true);
    // Matrix(size_t d1);
    // Matrix(size_t d1, size_t d2);
    // Matrix(size_t d1, size_t d2, size_t d3);
    ~Matrix();
    Matrix(const Matrix<T> &other); // 拷贝构造函数

    std::string shapeString() const;
    std::vector<size_t> shape() const;
    size_t shapeD1() const;
    size_t shapeD2() const;
    size_t shapeD3() const;

    // return the value at according index
    T &operator()(size_t i);
    T &operator()(size_t i, size_t j);
    T &operator()(size_t i, size_t j, size_t k);

    void transferToDevice();
    void transferToHost();

    void matrixMemset(T value);

    // Overload addition operators with optional broadcasting axis
    Matrix<T> operator+(const T &num) const;
    // Matrix<T> operator+(const Matrix<T> &other) const;
    // Matrix<T> operator+(const Matrix<T> &other, int axis) const;

    Matrix<T> &operator=(const Matrix<T> &other);     // Copy assignment
    Matrix<T> &operator=(Matrix<T> &&other) noexcept; // Move assignment
};

// Constructor Definitions
template <typename T>
Matrix<T>::Matrix(size_t d1, bool isOnDevice) : dim1(d1), dim2(1), dim3(1)
{
    initialize(isOnDevice);
}

template <typename T>
Matrix<T>::Matrix(size_t d1, size_t d2, bool isOnDevice) : dim1(d1), dim2(d2), dim3(1)
{
    initialize(isOnDevice);
}

template <typename T>
Matrix<T>::Matrix(size_t d1, size_t d2, size_t d3, bool isOnDevice) : dim1(d1), dim2(d2), dim3(d3)
{
    initialize(isOnDevice);
}

// ambiguity
// template <typename T>
// Matrix<T>::Matrix(size_t d1) : dim1(d1),isOnDevice(true)
// {
//     initialize();
// }

// template <typename T>
// Matrix<T>::Matrix(size_t d1, size_t d2) : dim1(d1), dim2(d2),isOnDevice(true)
// {
//     initialize();
// }

// template <typename T>
//  Matrix<T>::Matrix(size_t d1, size_t d2, size_t d3) : dim1(d1), dim2(d2), dim3(d3),isOnDevice(true)
// {
//     initialize();
// }

// Destructor Definition
template <typename T>
Matrix<T>::~Matrix()
{
    freeMemory();
}

// 拷贝构造函数
template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other)
{
    dim1 = other.dim1;
    dim2 = other.dim2;
    dim3 = other.dim3;
    totalSize = other.totalSize;
    dataPlace = other.dataPlace;
    data = other.data;
    // allocateMemory();
    // if (dataPlace == HOST)
    // {
    //     std::copy(other.data, other.data + totalSize, data);
    // }
    // else
    // {
    //     cudaMemcpy(data, other.data, totalSize * sizeof(T), cudaMemcpyDeviceToDevice);
    // }
}

// Initialize Method
template <typename T>
void Matrix<T>::initialize(bool isOnDevice)
{
    if (isOnDevice)
    {
        this->dataPlace = DEVICE;
    }
    else
    {
        this->dataPlace = HOST;
    }
    totalSize = dim1 * dim2 * dim3;
    data = nullptr;
    allocateMemory();
}
// Initialize Method
template <typename T>
void Matrix<T>::initialize()
{
    this->dataPlace = DEVICE;
    totalSize = dim1 * dim2 * dim3;
    data = nullptr;
    allocateMemory();
}

// Allocate Memory
template <typename T>
void Matrix<T>::allocateMemory()
{
    if (dataPlace == HOST)
    {
        data = new T[totalSize];
    }
    else
    {
        cudaMalloc(&data, totalSize * sizeof(T));
    }
}

// Free Memory
template <typename T>
void Matrix<T>::freeMemory()
{
    if (dataPlace == HOST)
    {
        delete[] data;
    }
    else
    {
        cudaFree(data);
    }
}

// Transfer to Device
template <typename T>
void Matrix<T>::transferToDevice()
{
    if (dataPlace == HOST)
    {
        T *deviceData;
        cudaMalloc(&deviceData, totalSize * sizeof(T));
        cudaMemcpy(deviceData, data, totalSize * sizeof(T), cudaMemcpyHostToDevice);
        freeMemory();
        data = deviceData;
        dataPlace = DEVICE;
    }
}

// Transfer to Host
template <typename T>
void Matrix<T>::transferToHost()
{
    if (dataPlace == DEVICE)
    {
        T *hostData = new T[totalSize];
        cudaMemcpy(hostData, data, totalSize * sizeof(T), cudaMemcpyDeviceToHost);
        freeMemory();
        data = hostData;
        dataPlace = HOST;
    }
}

template <typename T>
void Matrix<T>::matrixMemset(T value)
{
    if (dataPlace == HOST)
    {
        // cannot use memset here
        // https://codeforces.com/blog/entry/68747
        std::fill(data, data + totalSize, value);
    }
    else
    {
        thrust::device_ptr<T> dev_ptr(data);
        thrust::fill(dev_ptr, dev_ptr + totalSize, value);
        cudaDeviceSynchronize();
    }
}

// Shape Methods
template <typename T>
std::string Matrix<T>::shapeString() const
{
    return "(" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " + std::to_string(dim3) + ")";
}

template <typename T>
std::vector<size_t> Matrix<T>::shape() const
{
    return {dim1, dim2, dim3};
}

template <typename T>
size_t Matrix<T>::shapeD1() const
{
    return dim1;
}

template <typename T>
size_t Matrix<T>::shapeD2() const
{
    return dim2;
}

template <typename T>
size_t Matrix<T>::shapeD3() const
{
    return dim3;
}

// Overloaded Operator Methods
template <typename T>
T &Matrix<T>::operator()(size_t i)
{
    return data[i];
}

template <typename T>
T &Matrix<T>::operator()(size_t i, size_t j)
{
    return data[i * dim2 + j];
}

template <typename T>
T &Matrix<T>::operator()(size_t i, size_t j, size_t k)
{
    return data[(i * dim2 * dim3) + (j * dim3) + k];
}

// Helper method for broadcasting
template <typename T>
Matrix<T> Matrix<T>::broadcastTo(size_t new_dim1, size_t new_dim2, size_t new_dim3, int axis) const
{
    if ((dim1 != 1 && dim1 != new_dim1) ||
        (dim2 != 1 && dim2 != new_dim2) ||
        (dim3 != 1 && dim3 != new_dim3))
    {
        throw std::invalid_argument("Incompatible shapes for broadcasting");
    }

    Matrix<T> result(new_dim1, new_dim2, new_dim3);
    for (size_t i = 0; i < new_dim1; ++i)
    {
        for (size_t j = 0; j < new_dim2; ++j)
        {
            for (size_t k = 0; k < new_dim3; ++k)
            {
                size_t src_i = (dim1 == 1) ? 0 : i;
                size_t src_j = (dim2 == 1) ? 0 : j;
                size_t src_k = (dim3 == 1) ? 0 : k;
                result(i, j, k) = data[(src_i * dim2 * dim3) + (src_j * dim3) + src_k];
            }
        }
    }
    return result;
}

// Addition Operator Overloads
template <typename T>
Matrix<T> Matrix<T>::operator+(const T &num) const
{
    Matrix<T> result(dim1, dim2, dim3, false);
    if (dataPlace == HOST)
    {
        printf("in host\n");
        for (size_t i = 0; i < totalSize; ++i)
        {
            result.data[i] = data[i] + num;
        }
    }
    else
    {
        result.transferToDevice();
        size_t blockSize = TILE_SIZE * TILE_SIZE;
        size_t numBlocks = ceil(float(totalSize) / blockSize);
        matAdd<<<numBlocks, blockSize>>>(blockSize, data, num, result.data);
        cudaDeviceSynchronize();
    }
    return result;
}

// template <typename T>
// Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
// {
//     return *this + other, -1;
// }

// template <typename T>
// Matrix<T> Matrix<T>::operator+(const Matrix<T> &other, int axis) const
// {
//     size_t new_dim1 = std::max(dim1, other.dim1);
//     size_t new_dim2 = std::max(dim2, other.dim2);
//     size_t new_dim3 = std::max(dim3, other.dim3);

//     Matrix<T> broadcasted_other = other.broadcastTo(new_dim1, new_dim2, new_dim3, axis);

//     Matrix<T> result(new_dim1, new_dim2, new_dim3);
//     for (size_t i = 0; i < new_dim1; ++i)
//     {
//         for (size_t j = 0; j < new_dim2; ++j)
//         {
//             for (size_t k = 0; k < new_dim3; ++k)
//             {
//                 result(i, j, k) = (*this)(i, j, k) + broadcasted_other(i, j, k);
//             }
//         }
//     }
//     return result;
// }
// Copy assignment operator
template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
    if (this != &other)
    {
        freeMemory();
        dim1 = other.dim1;
        dim2 = other.dim2;
        dim3 = other.dim3;
        totalSize = other.totalSize;
        dataPlace = other.dataPlace;
        allocateMemory();
        if (dataPlace == HOST)
        {
            std::copy(other.data, other.data + totalSize, data);
        }
        else
        {
            cudaMemcpy(data, other.data, totalSize * sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }
    return *this;
}

// Move assignment operator
template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept
{
    if (this != &other)
    {
        freeMemory();
        dim1 = other.dim1;
        dim2 = other.dim2;
        dim3 = other.dim3;
        totalSize = other.totalSize;
        data = other.data;
        dataPlace = other.dataPlace;

        other.data = nullptr;
        other.totalSize = 0;
    }
    return *this;
}
template <typename T>
__global__ void matAdd(int mat_sz, const T *A, const T *B, T *C)
{ // C = A + B
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_sz)
    {
        C[i] = B[i] + A[i];
    }
}
template <typename T>
__global__ void matAdd(int mat_sz, const T *A, const T b, T *C)
{ // C = A + B
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mat_sz)
    {
        C[i] = b + A[i];
    }
}

// zeros and ones

template <typename T>
Matrix<T> zeros(size_t d1, bool isOnDevice = true)
{
    printf("in zeros\n");
    Matrix<T> result = Matrix<T>(d1, isOnDevice);
    result.matrixMemset(0);
    return result;
}

template <typename T>
Matrix<T> zeros(size_t d1, size_t d2,bool isOnDevice = true)
{
    printf("in zeros\n");
    Matrix<T> result = Matrix<T>(d1,d2, isOnDevice);
    result.matrixMemset(0);
    return result;
}

template <typename T>
Matrix<T> zeros(size_t d1, size_t d2,size_t d3,bool isOnDevice = true)
{
    printf("in zeros\n");
    Matrix<T> result = Matrix<T>(d1,d2,d3, isOnDevice);
    result.matrixMemset(0);
    return result;
}

template <typename T>
Matrix<T> ones(size_t d1, bool isOnDevice = true)
{
    printf("in ones\n");
    Matrix<T> result = Matrix<T>(d1, isOnDevice);
    result.matrixMemset(1);
    return result;
}

template <typename T>
Matrix<T> ones(size_t d1, size_t d2,bool isOnDevice = true)
{
    printf("in ones\n");
    Matrix<T> result = Matrix<T>(d1,d2, isOnDevice);
    result.matrixMemset(1);
    return result;
}

template <typename T>
Matrix<T> ones(size_t d1, size_t d2,size_t d3,bool isOnDevice = true)
{
    printf("in ones\n");
    Matrix<T> result = Matrix<T>(d1,d2,d3, isOnDevice);
    result.matrixMemset(1);
    return result;
}

#endif