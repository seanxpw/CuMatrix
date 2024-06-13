#include <iostream>
#include "Matrix.h"

int main()
{
    const bool GPU = true;
    const bool CPU = false;

    // create matrix

    // 1D
    Matrix<int> matrix1D_1(10, GPU);
    Matrix<double> matrix1D_2(100, CPU);

    // 2D 3D
    Matrix<int> matrix2D_1(10, 50, GPU);
    Matrix<double> matrix3D_2(10, 50, 300, CPU);
    Matrix<float> matrix1 = ones<float>(32, 32, 1, GPU);
    Matrix<float> matrix0 = zeros<float>(5, 1, GPU);

    // print a matrix, must in host to print
    matrix1.transferToHost();
    std::cout << matrix1;

    // ADD
    Matrix<float> mat1(4, 4, 2, GPU);
    mat1.matrixMemset(5.2);
    Matrix<float> mat2(4, 4, 1, GPU);
    mat2.matrixMemset(0.2);

    Matrix<float> mat3 = mat1 + mat2; // add with broadcasting
    mat3 = mat3 + 50;
    mat3.transferToHost();
    std::cout << mat3;
    /*
Matrix(4x4x2):
Slice 0:
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4

Slice 1:
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4
      55.4       55.4       55.4       55.4
    */

    mat3.transferToDevice();
    mat3 = mat1 /*5.2*/ + mat3 /*55.4*/ + 6 + mat1 /*5.2*/; // all add together
    mat3.transferToHost();
    std::cout << mat3;
    /*Matrix(4x4x2):
 Slice 0:
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8

 Slice 1:
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8
       71.8       71.8       71.8       71.8*/

    // element-wise Multiplication
    Matrix<float> mat10(4, 4, 2, GPU);
    mat10.matrixMemset(3);
    Matrix<float> mat11(1, 4, 2, GPU); 
    mat11.matrixMemset(2);
    Matrix<float> mat12 = mat11 * mat10;
    mat12.transferToHost();
    std::cout << mat12;

    return 0;
}
