#include <iostream>
#include "Matrix.h"




int main() {
    Matrix<float> matrix3D(4, 4, 4,false);
    matrix3D(0, 0, 0) = 1.0f;
    matrix3D = matrix3D + 10;
    Matrix<float> matrix1 = matrix3D + 100;

    // matrix3D.transferToDevice();
    // matrix3D.transferToHost();

    std::cout << "Matrix(0, 0, 0): " << matrix1(0, 0, 0) << std::endl;
    return 0;
}
