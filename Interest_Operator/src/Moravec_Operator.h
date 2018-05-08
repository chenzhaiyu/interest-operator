#pragma once

// Moravec算子头文件

#ifndef MORAVEC_OPERATOR_H
#define MORAVEC_OPERATOR_H
#endif

#include <stdio.h>
#include <math.h>
#include "mymath.h"
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"

# define Moravec_Interest_Window_Size 5		// 提取窗口尺寸
# define Moravec_Filter_Window_Size 21		// 精选窗口尺寸
# define Moravec_Threshold 20				// 兴趣值阈值

using namespace cv;
using namespace std;

class Moravec_Operator
{

public:
	Moravec_Operator();
	~Moravec_Operator();

	void extract(const Mat&);	// 点提取函数
	void draw(Mat&);			// 点绘制函数


private:

	Mat image_input;			// 输入的图像
	Mat max_in_window;			// 窗口中最大兴趣值

	int if_any_point_extracted; // 有无特征点标识符，缺省为0，若当前已有点提出，则置为1

	int window_j_present;		// 当前窗口点列号，用于拉着窗口进行遍历
	int window_i_present;		// 当前窗口点行号，用于拉着窗口进行遍历
	int window_j_before;		// 上一次窗口点列号
	int window_i_before;		// 上一次窗口点行号

	// 4个方向上分别计算的梯度值
	int interest_value_1;	
	int interest_value_2;
	int interest_value_3;
	int interest_value_4;

	vector<CvPoint> points_of_interest; // 用于存兴趣点的容器

};

