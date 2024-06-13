#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>

// replace cudamemset
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
__global__ void broadcastKernel(const T *src, T *dst, size_t dim1, size_t dim2, size_t dim3, size_t new_dim1, size_t new_dim2, size_t new_dim3);

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

    // Helper method for broadcasting, it'll try to broadcast "this" to the given matrix
    Matrix<T> _broadcastTo(const std::vector<size_t> &shapes) const;
    std::pair<T *, T *> _broadcast(const Matrix<T> &m1, const Matrix<T> &m2, std::vector<size_t> &broadcastedShape) const;
    int _reshape(size_t new_d1, size_t new_d2 = 1, size_t new_d3 = 1);

public:
    Matrix();
    Matrix(size_t d1, bool isOnDevice = true);
    Matrix(size_t d1, size_t d2, bool isOnDevice = true);
    Matrix(size_t d1, size_t d2, size_t d3, bool isOnDevice = true);

    ~Matrix();

    // copy constructor this is crazy!!
    Matrix(const Matrix<T> &other);

    // move constructor
    Matrix(Matrix<T> &&other) noexcept;

    size_t getTotalSize() const;

    std::string shapeString() const;
    std::vector<size_t> shape() const;
    size_t shapeD1() const;
    size_t shapeD2() const;
    size_t shapeD3() const;
    void reshape(size_t new_d1, size_t new_d2 = 1, size_t new_d3 = 1);

    // return the value at according index
    T &operator()(size_t i, size_t j = 0, size_t k = 0);
    const T &operator()(size_t i, size_t j = 0, size_t k = 0) const;

    void transferToDevice();
    void transferToHost();

    void matrixMemset(T value);

    // Overload addition operators with optional broadcasting axis

    // all elements add a num
    Matrix<T> operator+(const T &num) const;

    // all element add to another matrix
    Matrix<T> operator+(const Matrix<T> &other) const;

    Matrix<T> &operator=(const Matrix<T> &other);     // Copy assignment
    Matrix<T> &operator=(Matrix<T> &&other) noexcept; // Move assignment
};

// Constructor Definitions

template <typename T>
Matrix<T>::Matrix()
{
    dim1 = dim2 = dim3 = 0;
    initialize(false);
}

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

// Destructor Definition
template <typename T>
Matrix<T>::~Matrix()
{
    freeMemory();
}

// copy constructor
// Matrix<float> matrix3D(4, 4, 1, false);
// Matrix<float> matrix2 = matrix3D; // Calls the copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other)
{
    printf("in copy constructor\n");
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

// move constructor
template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
    : dim1(other.dim1), dim2(other.dim2), dim3(other.dim3), totalSize(other.totalSize), data(other.data), dataPlace(other.dataPlace)
{
    printf("in move constructor\n");
    other.data = nullptr;
    other.totalSize = 0;
}

template <typename T>
size_t Matrix<T>::getTotalSize() const
{
    return this->totalSize;
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
    if (totalSize == 0)
    {
        return;
    }
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

// return -1 on fail, 0 on success
template <typename T>
int Matrix<T>::_reshape(size_t new_d1, size_t new_d2, size_t new_d3)
{
    if (this->totalSize != new_d1 * new_d2 * new_d3)
    {
        return -1;
    }
    else
    {
        this->dim1 = new_d1;
        this->dim2 = new_d2;
        this->dim2 = new_d2;
        return 0;
    }
}

template <typename T>
void Matrix<T>::reshape(size_t new_d1, size_t new_d2, size_t new_d3)
{
    int result = _reshape(new_d1, new_d2, new_d3);
    if (result == -1)
    {
        printf("reshape faild %s cannot be reshaped to %ld, %ld, %ld", this->shapeString(), new_d1, new_d2, new_d3);
    }
}

// Overloaded Operator Methods

template <typename T>
T &Matrix<T>::operator()(size_t i, size_t j, size_t k)
{
    // printf("i = %d, j = %d, k = %d, index = %d\n",i,j,k,k * dim1*dim2 + j * dim1 + i);
    return this->data[k * dim1 * dim2 + j * dim1 + i];
}

template <typename T>
const T &Matrix<T>::operator()(size_t i, size_t j, size_t k) const
{
    return this->data[k * dim1 * dim2 + j * dim1 + i];
}

// Helper method for broadcasting
template <typename T>
Matrix<T> Matrix<T>::_broadcastTo(const std::vector<size_t> &otherShapes) const
{
    printf("in broad cast\n");

    size_t new_dim1 = otherShapes[0];
    size_t new_dim2 = otherShapes[1];
    size_t new_dim3 = otherShapes[2];
    printf("try to bradcast this: %s to %d, %d, %d \n", this-> shapeString().c_str(),new_dim1,new_dim2,new_dim3);

    // same dimension
    if (dim1 == new_dim1 && dim2 == new_dim2 && dim3 == new_dim3)
    {
        return *this;
    }
    // either the same dim as this matrix or 1
    if ((dim1 == new_dim1 || dim1 == 1) &&
        (dim2 == new_dim2 || dim2 == 1) &&
        (dim3 == new_dim3 || dim3 == 1))
    {
    }
    else
    {
        printf("bc failed\n");
        throw std::invalid_argument("Dimensions are not compatible for broadcasting");
    }
    if (this->dataPlace == HOST)
    {
        Matrix<T> result(new_dim1, new_dim2, new_dim3, false);

        // Iterate through the dimensions and copy values from the original matrix
        for (size_t i = 0; i < new_dim1; ++i)
        {
            for (size_t j = 0; j < new_dim2; ++j)
            {
                for (size_t k = 0; k < new_dim3; ++k)
                {
                    size_t src_i = dim1 == 1 ? 0 : i;
                    size_t src_j = dim2 == 1 ? 0 : j;
                    size_t src_k = dim3 == 1 ? 0 : k;
                    result(i, j, k) = (*this)(src_i, src_j, src_k);
                }
            }
        }
        printf("bc end\n");
        return result;
    }
    else
    {
        Matrix<T> result(new_dim1, new_dim2, new_dim3, true);
        result.transferToDevice();
        // ::dim3 means use cuda dim3 defined in glocal space 
        ::dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
        ::dim3 gridDim(
            (new_dim1 + blockDim.x - 1) / blockDim.x,
            (new_dim2 + blockDim.y - 1) / blockDim.y,
            (new_dim3 + blockDim.z - 1) / blockDim.z
        );
        broadcastKernel<<<gridDim, blockDim>>>(this->data, result.data, dim1, dim2, dim3, new_dim1, new_dim2, new_dim3);
        cudaDeviceSynchronize();
        printf(" out _broadcastTo\n");
        return result;
    }

    // so what can be broadcasted?
    // at least one dim should be the same?

    // broadcast with one by one here
    // A 3 3 3  B 1 1 3 => B 3 3 3
    // A 3 3 3  B 3 1 1 => B 3 3 3
    // A 3 3 3  B 1 3 1 => B 3 3 3
    // A 3 3 3  B 1 1 1 => B 3 3 3
    // one by one is special

    // A 9 9 3  B 3 3 3 => connot be broadcasted, even though it's possible to do 3*3 to 9*9

    // broadcast with one dim difference
    // A 3 3 3  B 3 1 3 => B 3 3 3
    // A 3 3 3  B 1 3 3 => B 3 3 3
    // A 3 3 3  B 3 3 1 => B 3 3 3
    // A 3 3 1  B 3 1 1 => B 3 3 1
}

// broadcast m1 or m2 to keep them the same dimension
// usually m1 = *this
template <typename T>
std::pair<T *, T *> Matrix<T>::_broadcast(const Matrix<T> &m1, const Matrix<T> &m2, std::vector<size_t> &broadcastedShape) const
{
    if (m1.getTotalSize() == m2.getTotalSize())
    {
        // No need to broadcast, return the original pointers
        broadcastedShape = m1.shape();
        return std::make_pair(m1.data, m2.data);
    }

    Matrix<T> broadCasted;
    if (m1.getTotalSize() > m2.getTotalSize())
    {
        // Broadcast m2 to m1
        printf(" Broadcast m2 to m1\n");
        broadCasted = m2._broadcastTo(m1.shape());
        broadcastedShape = broadCasted.shape();
        auto result = std::make_pair(m1.data, broadCasted.data);
        broadCasted.data = nullptr;
        return result;
    }
    else
    {
        // Broadcast m1 to m2
        // printf(" Broadcast m1 to m2\n");
        broadCasted = m1._broadcastTo(m2.shape());
        broadcastedShape = broadCasted.shape();
        // printf("after get shape from_broadcastTo \n\
        broadcastedDim1 = %d broadcastedDim2 =%d broadcastedDim3 = %d\n", broadcastedShape[0],broadcastedShape[1],broadcastedShape[2]);
        auto result = std::make_pair(broadCasted.data, m2.data);
        broadCasted.data = nullptr;
        // printf(" finish Broadcast m1 to m2\n");
        return result;
    }
}

// Addition Operator Overloads
template <typename T>
Matrix<T> Matrix<T>::operator+(const T &num) const
{
    // Matrix<T> result(dim1, dim2, dim3, false);
    // this is crazy if defined here, it won't call move constructor
    // may be Return Value Optimization (RVO) and Named Return Value Optimization (NRVO)
    if (dataPlace == HOST)
    {
        Matrix<T> result(dim1, dim2, dim3, false);
        printf("in host\n");
        for (size_t i = 0; i < totalSize; ++i)
        {
            result.data[i] = data[i] + num;
        }
        return result;
    }
    else
    {
        Matrix<T> result(dim1, dim2, dim3, true);
        result.transferToDevice();
        size_t blockSize = TILE_SIZE * TILE_SIZE;
        size_t numBlocks = ceil(float(totalSize) / blockSize);
        matAdd<<<numBlocks, blockSize>>>(blockSize, data, num, result.data);
        cudaDeviceSynchronize();
        return result;
    }
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
{
    // printf("this shape %s\n", this->shapeString().c_str());
    // printf("other shape %s\n", other.shapeString().c_str());
    std::vector<size_t> broadcastedShape;
    auto [data1, data2] = this->_broadcast(*this, other, broadcastedShape);
    size_t broadcastedDim1 = broadcastedShape[0], broadcastedDim2 = broadcastedShape[1], broadcastedDim3 = broadcastedShape[2];
    size_t broadcastedTotalSize = broadcastedDim1 * broadcastedDim2 * broadcastedDim2;

    if (dataPlace == HOST)
    {
        Matrix<T> result(broadcastedDim1, broadcastedDim2, broadcastedDim3, false);
        for (size_t i = 0; i < broadcastedTotalSize; ++i)
        {
            // printf(" data = %f, other data = %f\n", data[i], other.data[i]);
            result.data[i] = data1[i] + data2[i];
        }
        delete []data1;
        delete []data2;
        return result;
    }
    else
    {
        Matrix<T> result(broadcastedDim1, broadcastedDim2, broadcastedDim3, true);
        size_t blockSize = TILE_SIZE * TILE_SIZE;
        size_t numBlocks = ceil(float(broadcastedTotalSize) / blockSize);
        matAdd<<<numBlocks, blockSize>>>(blockSize, data1, data2, result.data);
        cudaDeviceSynchronize();
        // printf("out kernel\n");
        cudaFree(data1);
        cudaFree(data2);
        return result;
    }
}

// Copy assignment operator
// Matrix<float> matrix3D(4, 4, 1, false);
// Matrix<float> matrix2(4, 4, 1, false);
// matrix2 = matrix3D; // Calls the copy assignment operator
template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
    printf("out copy operator =\n");
    if (this != &other)
    {
        // printf("in operator =\n");
        // printf("old memory %f, other memory %f", data[0], other.data[0]);
        freeMemory();
        this->dim1 = other.dim1;
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
// man like A = A + B, the result of A + B is a right value
// matrix3D = matrix3D + 100;
template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept
{
    printf("in move operator= \n");
    if (this != &other)
    {
        // printf("old memory %f, other memory %f\n", data[0], other.data[0]);
        freeMemory();
        // printf("in move = broadcastedDim1 = %d broadcastedDim2 =%d broadcastedDim3 = %d\n", dim1,dim2,dim3);
        dim1 = other.dim1;
        dim2 = other.dim2;
        dim3 = other.dim3;
        // printf("in move = after copy broadcastedDim1 = %d broadcastedDim2 =%d broadcastedDim3 = %d\n", dim1,dim2,dim3);
        totalSize = other.totalSize;
        data = other.data;
        dataPlace = other.dataPlace;
        // printf("old memory %f, other memory %f\n", data[0], other.data[0]);

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

template <typename T>
__global__ void broadcastKernel(const T *src, T *dst, size_t dim1, size_t dim2, size_t dim3, size_t new_dim1, size_t new_dim2, size_t new_dim3)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < new_dim1 && j < new_dim2 && k < new_dim3)
    {
        // Determine the source indices considering broadcasting rules
        // For example, if dim1 = 1 and new_dim1 = 3,
        // the value from the original matrix at src_i = 0
        // should be used for all i in the destination matrix.
        size_t src_i = dim1 == 1 ? 0 : i;
        size_t src_j = dim2 == 1 ? 0 : j;
        size_t src_k = dim3 == 1 ? 0 : k;

        // Calculate the flattened index for the destination and source
        size_t dst_index = k * new_dim1 * new_dim2 + j * new_dim1 + i;
        size_t src_index = src_k * dim1 * dim2 + src_j * dim1 + src_i;

        dst[dst_index] = src[src_index];
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
Matrix<T> zeros(size_t d1, size_t d2, bool isOnDevice = true)
{
    printf("in zeros\n");
    Matrix<T> result = Matrix<T>(d1, d2, isOnDevice);
    result.matrixMemset(0);
    return result;
}

template <typename T>
Matrix<T> zeros(size_t d1, size_t d2, size_t d3, bool isOnDevice = true)
{
    printf("in zeros\n");
    Matrix<T> result = Matrix<T>(d1, d2, d3, isOnDevice);
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
Matrix<T> ones(size_t d1, size_t d2, bool isOnDevice = true)
{
    printf("in ones\n");
    Matrix<T> result = Matrix<T>(d1, d2, isOnDevice);
    result.matrixMemset(1);
    return result;
}

template <typename T>
Matrix<T> ones(size_t d1, size_t d2, size_t d3, bool isOnDevice = true)
{
    printf("in ones\n");
    Matrix<T> result = Matrix<T>(d1, d2, d3, isOnDevice);
    result.matrixMemset(1);
    return result;
}

#endif