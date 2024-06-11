#include <iostream>
#include "Matrix.h"




int main() {
    Matrix<float> matrix3D(4, 4, 4,false);
    matrix3D(0, 0, 0) = 1.0f;
    matrix3D + 1;

    // matrix3D.transferToDevice();
    // matrix3D.transferToHost();

    std::cout << "Matrix(0, 0, 0): " << matrix3D(0, 0, 0) << std::endl;
    return 0;
}
