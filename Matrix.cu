#include <iostream>
#include "Matrix.h"


int main() {
    Matrix<float> matrix3D(4, 4,false);
    std::cout << matrix3D.shapeString();
    // matrix3D(0, 0, 0) = 1.0f;
    matrix3D = matrix3D + 100;
    Matrix<float> matrix1 =  zeros<float>(1000,false);


    std::cout << "Matrix(0, 0, 0): " << matrix1(5) << std::endl;
    return 0;
}
