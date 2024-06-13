#include <iostream>
#include "Matrix.h"

int main()
{
    const bool GPU = true;
    const bool CPU = false;
    Matrix<float> matrix3D(40, 40, 4, GPU);
    Matrix<float> matrix1 = ones<float>(32, 32, 1, GPU);
    Matrix<float> matrix2 = ones<float>(32, 1, 1, GPU);
    matrix1 = matrix1 +1;
    // matrix1(5) = 100;
    printf("begin dot\n");
    matrix1 = matrix1.dot(matrix2);
    // matrix3D(1,1,3) = 888;
    // matrix3D = matrix3D + 100;
    // std::cout << matrix3D.shapeString();
    // matrix3D.matrixMemset(0);
    // matrix3D = matrix3D + 100;
    // // printf("start to copy things!!!\n");
    // Matrix<float> matrix2 = matrix3D;
    // printf("matrix3d\n");
    // for (int i = 0; i < matrix3D.getTotalSize(); i++)
    // {
    //     printf("%.1f, ",  matrix3D(i));
    // }
    // matrix2.transferToHost();
    // printf("matrix2\n");
    // for (int i = 0; i < matrix2.getTotalSize(); i++)
    // {
    //     printf("%.1f, ", matrix2(i));
    // }

    printf("\n");
   
    // printf("matrix1\n");
    // for (int i = 0; i < matrix1.getTotalSize(); i++)
    // {
    //     printf("%.1f, ", matrix1(i));
    // }
    // matrix1 =  1.5 + matrix1 ;
    // printf("matrix3D = matrix3D + matrix1\n");
    // matrix3D =   matrix3D  + ( matrix1 + 1.5 ) * matrix3D * 5 ;
    matrix1.transferToHost();

    printf("matrix3d\n");
    for (int i = 0; i < matrix1.getTotalSize(); i++)
    {
        printf("%.1f, ", matrix1(i));
    }
    // matrix1.transferToHost();
    // matrix3D.transferToHost();

    // std::cout << "Matrix1: " << matrix1(0) << std::endl;
    // std::cout << "Matrix3d: " << matrix3D(3, 2) << std::endl;
    // std::cout << "Matrix3d: " << matrix2(3, 2) << std::endl;
    return 0;
}
