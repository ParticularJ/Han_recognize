#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>
#include<time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

//读取样本
void readSample(vector<string> &path) {
	string buf;
	ifstream img_tst("C://Myself//example//opencv_C++//characterSample//test//test.txt");
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			path.push_back(buf);
		}
	}
	img_tst.close();
}

//识别汉字
void recognize(vector<int> a) {  
	vector<string> str;
	vector<string> str1;  
	vector<string> model;    
	string buf1; 
	string buf;
	ifstream han_directory("C://Myself//example//opencv_C++//characterSample//character.txt");
	ifstream compare_han("C://Myself//example//opencv_C++//characterSample//test//sample.txt");

	while (han_directory) {
		if (getline(han_directory, buf)) { 
			str.push_back(buf);
		}
	}
	han_directory.close(); 
	 
	while (compare_han) {
		if (getline(compare_han, buf1)) {
			model.push_back(buf1);
		}
	} 
	compare_han.close();

	for (vector<int>::iterator iter = a.begin(); iter != a.end(); iter++) {
		str1.push_back(str[*iter]);
	}

	int count = 0;
	for (int i = 0; i < str1.size(); ++i) {
		if (str1[i] == model[i])
			count++;
		else
			cout << str1[i];
	}
	//cout << count;
	cout << static_cast<float>(100*count / str1.size()) << "%" << endl;
}

//样本处理
void processSample(int a, int b, vector<string> &path,/* Ptr<SVM> svm*/Ptr<RTrees> rf) {
	Mat test=Mat::zeros(a,b,CV_32F);
	Mat trainImg = Mat::zeros(a, b, CV_8UC3);
	vector<int> sample;
	int ret;
	//ofstream predict_txt("C://Myself//example//opencv_C++//trainTemplate//train//testData//SVM_PREDICT.txt");
	for (int j = 0; j != path.size(); j++) {
		test = imread(path[j].c_str(), 1);//读入图像   
	//	imshow("predict", test);
		vector<float> descriptors;//维度
								  //预处理，灰度图和resize
		cvtColor(test, test, CV_BGR2GRAY);
		resize(test, trainImg, cv::Size(a, b), 0, 0, CV_INTER_LINEAR);
		
		waitKey(0);
		//提取hog
		HOGDescriptor *hog = new HOGDescriptor(cvSize(a, b), cvSize(8, 8), cvSize(4, 4), cvSize(4, 4), 9);
		//调用计算函数开始计算 
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0));
		//获得图片的特征矩阵
		Mat trainMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		int n = 0;
		//将图片特征存入矩阵  
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++) {
			trainMat.at<float>(0, n) = *iter;
			n++;
		}
		FileStorage fs("C://Myself//example//opencv_C++//characterSample//Train//pca_eigenvectors.xml", FileStorage::READ);
		Mat pca_eigenvectors;
		fs["vectors"] >> pca_eigenvectors;
		//cout << pca_eigenvectors.size();
		trainMat = (pca_eigenvectors*trainMat.t()).t();
		//cout << SVMtrainMat.size() << endl;
		//predict with svm
//		ret = svm->predict(SVMtrainMat); 
		//predict with rf  
		ret = rf->predict(trainMat); 
		//cout << ret;
		sample.push_back(ret);
	}
	recognize(sample);
}
 
int main() {
	int ImgWidht = 24;
	int ImgHeight = 24;
	vector<string> img_tst_path;
	//加载训练模型
	long beginTime = clock();

	//RF模型
	Ptr<RTrees> rf = StatModel::load<RTrees>("C://Myself//example//opencv_C++//characterSample//Train//RF_data_pca.xml");

	//SVM模型
//	Ptr<SVM> svm = StatModel::load<SVM>("C://Myself//example//opencv_C++//characterSample//Train//Svm_data_pca.xml");

	long endTime = clock();
	cout << "Time" << endTime - beginTime << endl;
	//读取样本
	long beginTime1 = clock();
	readSample(img_tst_path);
	//处理样本与识别with RF
	processSample(ImgWidht, ImgHeight, img_tst_path, rf);


	//处理样本与识别with SVM
//	processSample(ImgWidht, ImgHeight, img_tst_path,svm);
	long endTime1 = clock();
	cout << "Time:" << endTime1 - beginTime1 << endl;
	return 0;
}
