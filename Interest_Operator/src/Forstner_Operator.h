#pragma once

// Forstner算子头文件

#ifndef FORSTNER_OPERATOR_H
#define FORSTNER_OPERATOR_H
#endif

#include <stdio.h>
#include <math.h>
#include "mymath.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"

# define Forstner_Window_Size 5				// 窗口尺寸
# define Forstner_Threshold_q 0.9			// 圆度q阈值
# define Forstner_Threshold_w 7000			// 权值w阈值
# define Forstner_Refined_Window_Size 11	// 精选窗口尺寸

using namespace cv;
using namespace std;


class Forstner_Operator
{

public:
	Forstner_Operator();
	~Forstner_Operator();

	void extract(const Mat&); // 点提取函数
	void draw(Mat&);		  // 点绘制函数

private:

	int if_any_point_extracted;  // 有无特征点标识符，缺省为0，若当前已有点提出，则置为1
	int window_j_present;		 // 当前窗口点列号，用于拉着窗口进行遍历
	int window_i_present;		 // 当前窗口点行号，用于拉着窗口进行遍历
	int window_j_before;		 // 上一次窗口点列号
	int window_i_before;		 // 上一次窗口点行号

	Mat gradient_u;				 // 行方向的梯度
	Mat gradient_v;				 // 列方向的梯度
	Mat image_Q;				 // 所有q所构成的Mat阵
	Mat image_W;				 // 所有w所构成的Mat阵
	Mat max_q_in_window;		 // 当前窗口中最大的q值
	Mat max_w_in_window;		 // 当前窗口中最大的w值
	Mat N;						 // 协方差矩阵

	vector<CvPoint> points_of_interest;  // 用于存兴趣点的容器

};

