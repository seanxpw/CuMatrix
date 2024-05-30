enum DataPlace { HOST, DEVICE };

template <typename T>
class Matrix {
public:
    Matrix(size_t d1);
    Matrix(size_t d1, size_t d2);
    Matrix(size_t d1, size_t d2, size_t d3);
    ~Matrix();

    T& operator()(size_t i);
    T& operator()(size_t i, size_t j);
    T& operator()(size_t i, size_t j, size_t k);

    void allocateMemory();
    void freeMemory();

    void transferToDevice();
    void transferToHost();

private:
    T* data;
    DataPlace dataPlace;
    size_t dim1, dim2, dim3;
    size_t totalSize;

    void initialize();
};

template <typename T>
Matrix<T>::Matrix(size_t d1) : dim1(d1), dim2(1), dim3(1), dataPlace(HOST) {
    initialize();
}

template <typename T>
Matrix<T>::Matrix(size_t d1, size_t d2) : dim1(d1), dim2(d2), dim3(1), dataPlace(HOST) {
    initialize();
}

template <typename T>
Matrix<T>::Matrix(size_t d1, size_t d2, size_t d3) : dim1(d1), dim2(d2), dim3(d3), dataPlace(HOST) {
    initialize();
}

template <typename T>
Matrix<T>::~Matrix() {
    freeMemory();
}

template <typename T>
void Matrix<T>::initialize() {
    totalSize = dim1 * dim2 * dim3;
    data = nullptr;
    allocateMemory();
}

template <typename T>
void Matrix<T>::allocateMemory() {
    if (dataPlace == HOST) {
        data = new T[totalSize];
    } else {
        cudaMalloc(&data, totalSize * sizeof(T));
    }
}

template <typename T>
void Matrix<T>::freeMemory() {
    if (dataPlace == HOST) {
        delete[] data;
    } else {
        cudaFree(data);
    }
}

template <typename T>
void Matrix<T>::transferToDevice() {
    if (dataPlace == HOST) {
        T* deviceData;
        cudaMalloc(&deviceData, totalSize * sizeof(T));
        cudaMemcpy(deviceData, data, totalSize * sizeof(T), cudaMemcpyHostToDevice);
        freeMemory();
        data = deviceData;
        dataPlace = DEVICE;
    }
}

template <typename T>
void Matrix<T>::transferToHost() {
    if (dataPlace == DEVICE) {
        T* hostData = new T[totalSize];
        cudaMemcpy(hostData, data, totalSize * sizeof(T), cudaMemcpyDeviceToHost);
        freeMemory();
        data = hostData;
        dataPlace = HOST;
    }
}

template <typename T>
T& Matrix<T>::operator()(size_t i) {
    return data[i];
}

template <typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    return data[i * dim2 + j];
}

template <typename T>
T& Matrix<T>::operator()(size_t i, size_t j, size_t k) {
    return data[(i * dim2 * dim3) + (j * dim3) + k];
}