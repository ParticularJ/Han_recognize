#include <iostream>
#include <opencv2/opencv.hpp>  
#include <fstream>
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int readformtext(vector<string> &path,vector<int> &catg) {
	int nLine = 0;
	string buf;
	//读入路径，标签文件
	ifstream svm_data("C://Myself//example//opencv_C++//characterSample//Train//han.txt");
	//将训练样本文件依次读取进来
	while (svm_data) {
		if (getline(svm_data, buf)) {
			nLine++;
			//	if (nLine % 2 == 0)//奇数行是图片全路径，偶数行是标签   
			{
			//			catg.push_back(atoi(buf.c_str()));//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错      
			}
			//else
			{
				path.push_back(buf);//图像路径      
			}
		}
	}
	for (int i = 0; i != 500; i++) {
		for (int j = 0; j < 5; j++) {
			catg.push_back(i);
		}
	}
//	for (vector<int>::iterator i = catg.begin(); i != catg.end(); ++i) {
//		cout << *i << endl;
//	}
	//cout << path.size();
	//cout << nLine;
	svm_data.close();//关闭文件
	return nLine;
}

void hogFeatur(Mat &data,Mat &lable,int line,vector<int> &catg,vector<string> &path) {
	int widht = 24;
	int height = 24;
	int nImgNum = line; //读入样本数量  
	//样本矩阵，nImgNum：横坐标是样本数量， WIDTH * HEIGHT：样本特征向量，即图像大小    
//	data = Mat::zeros(nImgNum, 36, CV_32FC1);
	//类型矩阵,存储每个样本的类型标志    
	lable = Mat::zeros(nImgNum, 1, CV_32SC1);
	//src:图片矩阵；trainImg分析的图片矩阵
	Mat src;
	Mat trainImg = Mat::zeros(height, widht, CV_8UC3);//需要分析的图片    

	for (int i = 0; i != catg.size(); i++){
		src = imread(path[i].c_str(), 1);
		cvtColor(src, src, CV_BGR2GRAY);
		resize(src, trainImg, Size(height, widht), 0, 0, CV_INTER_LINEAR);
		//imshow(" ", trainImg);
		//waitKey();

		HOGDescriptor *hog = new HOGDescriptor(cvSize(height, widht), cvSize(8, 8), cvSize(4, 4), cvSize(4, 4), 9);       
		vector<float> descriptors;//结果数组       
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //调用计算函数开始计算  
																
		if (i == 0)
		{
			data = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1); //根据输入图片大小进行分配空间   
		}
		cout << "HOG dims: " << descriptors.size() << endl;
		unsigned long n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++) {
			data.at<float>(i, n) = *iter;
			n++;
		}
		lable.at<int>(i, 0) = catg[i];

		//PCA
		int count = 0;
		float sum = 0;
		float sum_eigevalue = 0;
		PCA pca(data, Mat(), 0);
		//cout << pca.eigenvalues << endl;
		for (int i = 0; i < pca.eigenvalues.cols; ++i) {
			for (int j = 0; j < pca.eigenvalues.rows; ++j) {
				sum += pca.eigenvalues.at<float>(i, j);
				if (pca.eigenvalues.at<float>(i,j) > 0) {
					sum_eigevalue += pca.eigenvalues.at<float>(i, j);
					count++;
				}
			}
		}
		cout << sum_eigevalue/sum;
		cout << count;
		cout << endl;
		//cout << pca.eigenvectors << endl;
	}
}

void trainTemplate(Mat &a,Mat &b) {
	//SVM algorithm
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	//svm->setDegree(10.0);
	svm->setGamma(0.09);
	//svm->setCoef0(1.0);
	svm->setC(10.0);
	//svm->setNu(0.5);
	//svm->setP(1.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆   
	Ptr<TrainData> tData = TrainData::create(a, ROW_SAMPLE, b);
	svm->train(tData);
	//☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆       
	svm->save("C://Myself//example//opencv_C++//characterSample//Train//Svm_data.xml");
}

int main(){
	//图片路径
	vector<string> img_path;     
	//图片类别
	vector<int> img_catg;
	//图片特征以及标签
	Mat data_mat, label_mat;
	//读取文件
	int nLine=readformtext(img_path,img_catg);
	
	//特征提取
	hogFeatur(data_mat,label_mat,nLine,img_catg, img_path);

	//SVM训练
	long beginTime = clock();
//	trainTemplate(data_mat, label_mat);
	long endTime = clock();
	cout << "Time:"<<endTime - beginTime;
	return 0;
}
