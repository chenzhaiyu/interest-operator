#pragma once

// Harris算子头文件

#ifndef HARRIS_OPERATOR_H
#define HARRIS_OPERATOR_H
#endif

#include <stdio.h>
#include <math.h>
#include "mymath.h"
#include "opencv2/core/core.hpp"  
#include"opencv2/highgui/highgui.hpp"  
#include"opencv2/imgproc/imgproc.hpp"

# define Harris_Window_Size 21  // 窗口尺寸

using namespace cv;
using namespace std;

class Harris_Operator
{

public:
	Harris_Operator();
	~Harris_Operator();

	void extract(const Mat&); // 点提取函数
	void draw(Mat&);		  // 点绘制函数

private:

	int if_any_point_extracted; // 有无特征点标识符，缺省为0，若当前已有点提出，则置为1

	vector<CvPoint> points_of_interest; // 用于存兴趣点的容器

	Mat image_input;	// 输入的图像
	Mat max_in_window;  // 当前窗口中最大兴趣值
	Mat gx;				// x方向梯度（原始）
	Mat gy;				// y方向梯度（原始）
	Mat Gx;				// x方向梯度（经过高斯滤波）
	Mat Gy;				// y方向梯度（经过高斯滤波）
	Mat GxGy;			// 协方差GxGy
	Mat GxGx;			// 协方差GxGx
	Mat GyGy;			// 协方差GyGy
	Mat Intensity;		// 强度值
	Mat Gausiankernel;  // 高斯核

};