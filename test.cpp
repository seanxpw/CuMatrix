#include <iostream>
#include <Eigen/Dense>
#include <cstring>
using Eigen::MatrixXd;
 
int main()
{
int a[100];
memset(a, 1, sizeof(a));
for(int i =0;i< 100;i++){
    printf("%d\n", a[i]);
}
}