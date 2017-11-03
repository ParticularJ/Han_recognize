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
	//����·������ǩ�ļ�
	ifstream svm_data("C://Myself//example//opencv_C++//characterSample//Train//han.txt");
	//��ѵ�������ļ����ζ�ȡ����
	while (svm_data) {
		if (getline(svm_data, buf)) {
			nLine++;
			//	if (nLine % 2 == 0)//��������ͼƬȫ·����ż�����Ǳ�ǩ   
			{
			//			catg.push_back(atoi(buf.c_str()));//atoi���ַ���ת�������ͣ���־(0,1��2��...��9)��ע����������Ҫ��������𣬷�������      
			}
			//else
			{
				path.push_back(buf);//ͼ��·��      
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
	svm_data.close();//�ر��ļ�
	return nLine;
}

void hogFeatur(Mat &data,Mat &lable,int line,vector<int> &catg,vector<string> &path) {
	int widht = 24;
	int height = 24;
	int nImgNum = line; //������������  
	//��������nImgNum�������������������� WIDTH * HEIGHT������������������ͼ���С    
//	data = Mat::zeros(nImgNum, 36, CV_32FC1);
	//���;���,�洢ÿ�����������ͱ�־    
	lable = Mat::zeros(nImgNum, 1, CV_32SC1);
	//src:ͼƬ����trainImg������ͼƬ����
	Mat src;
	Mat trainImg = Mat::zeros(height, widht, CV_8UC3);//��Ҫ������ͼƬ    

	for (int i = 0; i != catg.size(); i++){
		src = imread(path[i].c_str(), 1);
		cvtColor(src, src, CV_BGR2GRAY);
		resize(src, trainImg, Size(height, widht), 0, 0, CV_INTER_LINEAR);
		//imshow(" ", trainImg);
		//waitKey();

		HOGDescriptor *hog = new HOGDescriptor(cvSize(height, widht), cvSize(8, 8), cvSize(4, 4), cvSize(4, 4), 9);       
		vector<float> descriptors;//�������       
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //���ü��㺯����ʼ����  
																
		if (i == 0)
		{
			data = Mat::zeros(nImgNum, descriptors.size(), CV_32FC1); //��������ͼƬ��С���з���ռ�   
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
	//����������(5)SVMѧϰ�������������   
	Ptr<TrainData> tData = TrainData::create(a, ROW_SAMPLE, b);
	svm->train(tData);
	//�������ѵ�����ݺ�ȷ����ѧϰ����,����SVMѧϰ�����       
	svm->save("C://Myself//example//opencv_C++//characterSample//Train//Svm_data.xml");
}

int main(){
	//ͼƬ·��
	vector<string> img_path;     
	//ͼƬ���
	vector<int> img_catg;
	//ͼƬ�����Լ���ǩ
	Mat data_mat, label_mat;
	//��ȡ�ļ�
	int nLine=readformtext(img_path,img_catg);
	
	//������ȡ
	hogFeatur(data_mat,label_mat,nLine,img_catg, img_path);

	//SVMѵ��
	long beginTime = clock();
//	trainTemplate(data_mat, label_mat);
	long endTime = clock();
	cout << "Time:"<<endTime - beginTime;
	return 0;
}
