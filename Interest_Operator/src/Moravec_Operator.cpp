#include "Moravec_Operator.h"

// Moravec算子cpp文件

// 初始化Moravec算子实例
Moravec_Operator::Moravec_Operator()
{
	if_any_point_extracted = 0;
	window_j_before = 0;
	window_i_before = 0;
	window_j_present = 0;
	window_i_present = 0;
}


Moravec_Operator::~Moravec_Operator()
{

}

// 点提取函数
void Moravec_Operator::extract(const Mat& image_input)
{
	max_in_window = Mat(image_input.rows, image_input.cols, CV_8UC1, Scalar(0)); // 这里是个坑，如果也用8位，大于255的兴趣值将无法存入max_in_window阵
	CV_Assert(image_input.depth() == CV_8U);							         // 仅接受对CV_8U图像的处理，不满足则抛出异常

	// 逐像素对行进行遍历，第一行和最后一行不进行计算
	for (int j = 2; j < image_input.rows - 2; ++j)                               
	{
		for (int i = 2; i < (image_input.cols - 2); ++i)				       
		{
			// 将4个方向的梯度初始化为0
			interest_value_1 = 0;
			interest_value_2 = 0;
			interest_value_3 = 0;
			interest_value_4 = 0;

			// 4个方向上梯度值计算
			for (int w = -int(Moravec_Interest_Window_Size / 2); w <= int(Moravec_Interest_Window_Size / 2) - 1; w++)
			{
				interest_value_1 += pow(image_input.ptr<uchar>(j + w)[i] - image_input.ptr<uchar>(j + w + 1)[i], 2);			 // 竖直方向
				interest_value_2 += pow(image_input.ptr<uchar>(j)[i + w] - image_input.ptr<uchar>(j)[i + w + 1], 2);			 // 水平方向
				interest_value_3 += pow(image_input.ptr<uchar>(j + w)[i + w] - image_input.ptr<uchar>(j + w + 1)[i + w + 1], 2); // 左下到右上
				interest_value_4 += pow(image_input.ptr<uchar>(j + w)[i + w] - image_input.ptr<uchar>(j + w + 1)[i + w - 1], 2); // 左上到右下
			}
			// 4个方向上梯度值最小值
			int interest_value_min = min(interest_value_1, interest_value_2, interest_value_3, interest_value_4) / 100; // 除以100是为了不让兴趣值超过255，这样就不用改动max_in_window的类型
			
			// 当前窗口的行列序号（以窗口行列索引来编号，e.g.1，2，3，4）
			window_j_present = j / Moravec_Filter_Window_Size;
			window_i_present = i / Moravec_Filter_Window_Size;

			if (interest_value_min > Moravec_Threshold && interest_value_min > max_in_window.at<uchar>(window_j_present, window_i_present))
			{
				// 直接在这里利用条件判断进行抑制局部最大
				max_in_window.at<uchar>(window_j_present, window_i_present) = interest_value_min;
				// 条件：仍在当前窗口中且有更大兴趣值的点进来，操作：剔除前面的点
				if (window_i_before == window_i_present && window_j_before == window_j_present && if_any_point_extracted != 0)
				{
					points_of_interest.pop_back();
				}
				// 将满足条件的当前点加入容器中
				points_of_interest.push_back(cvPoint(i, j));
				if_any_point_extracted = 1;
			}
			// 为了判断是不是还是同一个格子，通过比较格子坐标新旧值
			window_i_before = window_i_present;
			window_j_before = window_j_present;
		}
	}
}

// 点绘制函数
void Moravec_Operator::draw(Mat& Result)
{
	for (int i = 0; i < points_of_interest.size(); i++)
		cv::circle(Result, points_of_interest[i], 4, cv::Scalar(255, 255, 100));
}