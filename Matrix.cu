#include <iostream>
#include "Matrix.h"

int main()
{
    Matrix<float> matrix3D(4, 4, false);
    std::cout << matrix3D.shapeString();
    matrix3D.matrixMemset(0);
    matrix3D = matrix3D + 100;
    // printf("start to copy things!!!\n");
    Matrix<float> matrix2 = matrix3D;
    printf("matrix3d\n");
    for (int i = 0; i < matrix3D.getTotalSize(); i++)
    {
        printf("%.1f, ", matrix3D(i));
    }
        printf("matrix2\n");
    for (int i = 0; i < matrix2.getTotalSize(); i++)
    {
        printf("%.1f, ", matrix2(i));
    }

    printf("\n");
    Matrix<float> matrix1 = ones<float>(4, 1, false);
    printf("matrix1\n");
    for (int i = 0; i < matrix1.getTotalSize(); i++)
    {
        printf("%.1f, ", matrix1(i));
    }
    matrix3D = matrix3D + matrix1;
    printf("matrix3D = matrix3D + matrix1\n");
    printf("matrix3d\n");
    for (int i = 0; i < matrix3D.getTotalSize(); i++)
    {
        printf("%.1f, ", matrix3D(i));
    }
    matrix1.transferToHost();
    matrix3D.transferToHost();

    std::cout << "Matrix1: " << matrix1(0) << std::endl;
    std::cout << "Matrix3d: " << matrix3D(3, 2) << std::endl;
    std::cout << "Matrix3d: " << matrix2(3, 2) << std::endl;
    return 0;
}
