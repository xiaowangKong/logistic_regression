//
// Created by kxw on 11/25/16.
//

#include "LR.h"
LR::LR(int dimension) {
    this->dimension = dimension;
    wb=new double[dimension];
}
double * LR::splitByMyChar(string line,string mychar){

    string::size_type index = line.find_first_of(mychar,0);//找到空格、回车、换行等空白符
    if(index<line.size()){
        double *res = new double[dimension+1];//先new一个结果double数组
        int i=0;
        while (index != string::npos) {
            //      cout<<"放入list元素是"<<line.substr(0,index)<<endl;
            res[i]=atof(line.substr(0,index).c_str());
            //s  cout<<res[i]<<endl;
            //  feature_value.push_back(line.substr(0, index));
            line = line.substr(index + 1);
            //     cout<<"去掉一个dest_id剩余子串是："<<line<<endl;
            index = line.find_first_of(mychar, 0);
            //    cout<<"空格键下标是"<<index<<endl;
            ++i;
        }
        res[i]=atof(line.c_str());
        //cout<<res[i]<<endl;
        return  res;
    }
    else{//给定字符串line中没有要分割的字符mychar,返回长度为1的源字符串
        double *res = new double[1];//先new一个结果double数组
        res[0]=atof(line.c_str());
        return res;
    }
}
 double * LR::calGradient( double expVector[]){//wb
     double * bw_vec_result = new double[dimension];
     double y = expVector[dimension-1];
     int i = 0;
     double temp = y-sigmoid(expVector);
     for(i=0;i<dimension;i++)
        {
            if(i==dimension-1)
                bw_vec_result[i] = temp; // (4) for b
            else
                bw_vec_result[i] = (temp*expVector[i]); // (3) for w vectors
        }

    return bw_vec_result;
}
// (2) function
 double LR::sigmoid( double expVector [] ){
        if(wb==NULL || expVector ==NULL ){
            cout<<"Mismatch in array dimensions in sigmoid function"<<endl;
            exit(0);
        }
        double sum=0;
// Inner Product
        for(int i =0; i<dimension-1;i++){//去除最后一个标签
            sum+= (wb[i]*expVector[i]);
        }
        return 1/(1+exp(-sum-wb[dimension-1]));
}
 double *  LR::sumArray(double x[], double y[])  {
        if(x==NULL|| y ==NULL ){
            cout<<"Mismatch in array dimensions in sigmoid function"<<endl;
            exit(0);
        }
        double * sum = new double[dimension];
        for(int i=0; i<dimension ; i++)
        {
            sum[i]= x[i]+y[i];
        }
        return sum;
}

void LR::init_wb() {
    //Type of random number distribution
    std::uniform_real_distribution<double> dist(-1.0, 1.0);  //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;
    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());
    for(int i=0;i<dimension;i++) {
        wb[i] = dist(rng);
    }
}
int LR::training_theta(char *fname, double alpha) {//返回本次迭代中生成的新的梯度求和
    ifstream source_in;//用来读源文件
    // ofstream source_out;//将文件根据分类分成几个文件
    source_in.open(fname, ios::in);//首先打开属性描SS述文件
    if(!source_in){
        cout<<"文件"<<fname<<"不能打开！"<<endl;
        exit(0);
    }
    string line;
    int traincount=0;
    double * sum =new double[dimension];//用来存梯度求和，要记得delete sum
    // char pg_message[256]="/home/kxw/研一开题/data/pagerank/pg_message";
    getline(source_in,line);//以文件句柄读取一行
    while(!source_in.eof()){
        traincount++;//记录文件里有多少条记录
        double * res=splitByMyChar(line,",");///根据features记录下的属性个数，包括标签，将源数据的每一行都分解成单个的属性数组，分割符号是“，”，有可能会变哦，要注意！
        double * gradient ;
        gradient = calGradient(res);//要记得delete gradient
        for(int i=0; i<dimension ; i++)//求和这些梯度值
        {
            sum[i]= sum[i]+gradient[i];
        }
        getline(source_in,line);//以文件句柄读取一行
        delete res;
        delete gradient;
     }//否则说明是连续的，values没有赋值
    source_in.close();//此时应该是离散属性的频率统计和连续属性的mean求和项已经结束

    //double[] minusArray(double[] x, double[] y, double alpha, double trainCount){LRutil.minusArray(parseStringToVector(pre_v),sum,ALPHA,trainCount);

    for(int i=0; i<dimension; i++) {
            wb[i]= wb[i]+(alpha*sum[i]);//利用之前的wb和计算好的梯度增量求和，更新wb为新的wb
     }
    delete sum;
    return traincount;
    }

int LR::predictLabel(char *sourcename, char *outfile) {
    ifstream source_in;//用来读源文件
    ofstream source_out;//将文件根据分类分成几个文件
    source_in.open(sourcename, ios::in);//首先打开属性描SS述文件
    if (!source_in) {
        cout << "文件" << sourcename << "不能打开！" << endl;
        exit(0);
    }
    string line;
    int count_correct = 0;
    int count = 0;
    // char pg_message[256]="/home/kxw/研一开题/data/pagerank/pg_message";
    getline(source_in, line);//以文件句柄读取一行
    while (!source_in.eof()) {
        count++;
        double c=0;//二分类标签
        double *res = splitByMyChar(line, ",");///根据features记录下的属性个数，包括标签，将源数据的每一行都分解成单个的属性数组，分割符号是“，”，有可能会变哦，要注意！
        //map<string,vector<map<string,double >>> model;//
        double p1 = sigmoid(res);
        if(p1>0.5) c=1;

        ///////////////////////////////////////将当前预测完的属性值和预测标签值写入输出文件（作为一行）

        source_out.open(outfile,ios::ate|ios::app);//,ios::app);
        if(!source_out){
            cout<<"不能打开要写的文件"<<outfile<<endl;
            exit(0);
        }
        if(res[dimension-1]==c){//原来的label == predict label
            count_correct++;
            source_out<<line<<endl;
        }
        else{ source_out<<line.substr(0,line.size()-1)<<c<<endl;//将本点指向本点值置为0，防止丢失pagerank，并且不影响分数统计
        }
        source_out.flush();
        source_out.close();
        getline(source_in, line);//以文件句柄读取一行
        delete res;
    }
    source_in.close();//此时应该是离散属性的频率统计和连续属性的mean求和项已经结束
    cout<<"======================================================="<<endl;
    cout<<"Summary                                                "<<endl;
    cout<<"-------------------------------------------------------"<<endl;
    cout<<"Correctly Classified Instances          :       "<<count_correct<<"       "<<(100.0*count_correct/count)<<"%"<<endl;
    cout<<"Incorrectly Classified Instances        :       "<<count-count_correct<<"       "<<((100.0-100.0*count_correct/count))<<"%"<<endl;
    cout<<"Total Classified Instances              :       "<<count<<"       "<<endl;
    cout<<"======================================================="<<endl;
    return count_correct;
}
