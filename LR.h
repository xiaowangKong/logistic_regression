//
// Created by kxw on 11/25/16.
//

#ifndef LOGISTIC_REGRESSION_LR_H
#define LOGISTIC_REGRESSION_LR_H

#include <iostream>
#include <string>
#include <list>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <random>
using namespace std;

class LR {
public:
     int dimension;////属性维度+1
     double * wb;//w和b
     LR(int dimension);
     double * splitByMyChar(string line,string mychar);
     double * calGradient( double expVector[]);
     double sigmoid(double expVector [] );
     double *  sumArray(double x[], double y[]);
     double GetRandomReal(double low, double up);
     void init_wb();
     int  training_theta(char* fname,double alpha);
     int  predictLabel(char *sourcename, char *outfile);
};


#endif //LOGISTIC_REGRESSION_LR_H
