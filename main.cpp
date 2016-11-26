#include <iostream>
#include "LR.h"
#include <time.h>
int main(int argc,char *argv[]) {

    const double  ALPHA = 0.01;
   /* char * inputPath = argv[0];
    char * testPath = argv[1];
    char * outputPath = argv[2];
    int specIteration = 20;
    int feature_dimension ;
    if (argc > 4) {
        feature_dimension = atoi(argv[3]);
    }
    if (argc > 5) {
        specIteration = atof(argv[4]);
    }
*/
    char  inputPath[] = "/home/kxw/研一开题/data/logisticRegression/iris_training";
    char  testPath[] = "/home/kxw/研一开题/data/logisticRegression/iris_testing";
    char  outputPath[] = "/home/kxw/研一开题/data/logisticRegression/iris_testing_res";
    int specIteration = 200;
    int feature_dimension =5;
    clock_t start_time=clock();

    int currentIteration = 0;
    LR lr = LR(feature_dimension);
    lr.init_wb();//初始化wb
// Driver calls previous b and w, so it set b w for map get function
    while (currentIteration < specIteration) {
       int num_of_sample = lr.training_theta(inputPath,ALPHA);
        currentIteration++;
    }
    lr.predictLabel(testPath,outputPath);
    clock_t end_time=clock();
    cout<< "Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    return 0;
}

