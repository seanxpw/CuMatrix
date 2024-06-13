#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;

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

    // assign values only works on host (CPU)
    matrix1.transferToHost();
    matrix1(0) = 50.0f;

    // print a matrix, must in host to print
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

    // ELEMENT-WISE Multiplication
    Matrix<float> mat10(4, 4, 2, GPU);
    mat10.matrixMemset(3);
    Matrix<float> mat11(1, 4, 1, GPU);
    mat11.matrixMemset(2);
    Matrix<float> mat12 = mat11 * mat10;
    mat12.transferToHost();
    std::cout << mat12;
    /*Matrix(4x4x2):
Slice 0:
         6          6          6          6
         6          6          6          6
         6          6          6          6
         6          6          6          6

Slice 1:
         6          6          6          6
         6          6          6          6
         6          6          6          6
         6          6          6          6 */

    // COMBINE ADDITION AND MNULTIPLICATION
    mat12 = mat1 /*5.2*/ + mat10 /*3*/ * mat11 /*2*/; // 5.2 + 6 = 11.2
    mat12.transferToHost();
    std::cout << mat12;
    /*Matrix(4x4x2):
Slice 0:
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2

Slice 1:
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2
      11.2       11.2       11.2       11.2 */
    mat12 = (mat1 /*5.2*/ + mat10 /*3*/) * mat11 /*2*/; // 8.2*2=16.4
    mat12.transferToHost();
    std::cout << mat12;
    /*Matrix(4x4x2):
Slice 0:
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4

Slice 1:
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4
      16.4       16.4       16.4       16.4 */

    // DOT PRODUCT
    Matrix<float> matrix3232 = ones<float>(32, 32, 1, GPU);
    Matrix<float> matrix3201 = ones<float>(32, 1, 1, GPU);
    matrix3232 = matrix3232.dot(matrix3201);
    matrix3232.transferToHost();
    std::cout << matrix3232;
    /*Matrix(32x1x1):
        32
        32
        32
  ...
        32
        32
        32
    */

    // compare CPU and GPU speed
    mat1 = ones<float>(1000, 1000, 50, CPU);
    mat2 = ones<float>(1000, 1000, 1, CPU);
    mat3 = ones<float>(1000, 1000, 50, CPU);
    mat3.matrixMemset(5.5);
    Matrix<float> mat4 = ones<float>(1000, 1000, 50, CPU);

    auto start = system_clock::now();
    mat4 = mat2 + mat1 * 20 * (mat3 + 4);
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "CPU: "
         << double(duration.count()) * microseconds::period::num / microseconds::period::den
         << "sec" << endl;
    /*CPU: 1.21958sec*/
    cout << mat4 << endl;

    mat1.transferToDevice();
    mat2.transferToDevice();
    mat3.transferToDevice();
    start = system_clock::now();
    mat1 = mat2 + mat1 * 20 * (mat3 + 4);
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "GPU: "
         << double(duration.count()) * microseconds::period::num / microseconds::period::den
         << "sec" << endl;
    /*GPU: 0.063841sec*/
    mat4.transferToHost();
    cout << mat4 << endl;
    return 0;
}
