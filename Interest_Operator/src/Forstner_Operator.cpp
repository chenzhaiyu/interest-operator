#include "Forstner_Operator.h"

// Forstner算子cpp文件

// 初始化Forstner算子实例
Forstner_Operator::Forstner_Operator()
{
	if_any_point_extracted = 0;
	window_j_present = 0;
	window_i_present = 0;
	window_j_before = 0;
	window_i_before = 0;
}

Forstner_Operator::~Forstner_Operator()
{

}

// 点提取函数
void Forstner_Operator::extract(const Mat& myImage)
{
	gradient_u = Mat(myImage.rows, myImage.cols, CV_16SC1, Scalar(0)); // 存行方向梯度图像
	gradient_v = Mat(myImage.rows, myImage.cols, CV_16SC1, Scalar(0)); // 存列方向梯度图像
	image_Q = Mat(myImage.rows, myImage.cols, CV_32FC1, Scalar(0));    // 存整幅图像的圆度，32浮点型
	image_W = Mat(myImage.rows, myImage.cols, CV_32FC1, Scalar(0));    // 存整幅图像的权值，32浮点型
	
	CV_Assert(myImage.depth() == CV_8U); // 仅处理灰度图像

	max_q_in_window = Mat(myImage.rows, myImage.cols, CV_32FC1, Scalar(0)); // 存当前窗口中q的最大值
	max_w_in_window = Mat(myImage.rows, myImage.cols, CV_32FC1, Scalar(0)); // 存当前窗口中w的最大值

	// 计算每个像素处的Robert梯度
	// 第一行和最后一行不进行计算
	for (int j = 0; j < myImage.rows - 1; ++j) 
	{
		for (int i = 0; i < (myImage.cols - 1); ++i)				      
		{
			gradient_u.at<short>(j, i) = myImage.ptr<uchar>(j + 1)[i + 1] - myImage.ptr<uchar>(j)[i];
			gradient_v.at<short>(j, i) = myImage.ptr<uchar>(j)[i + 1] - myImage.ptr<uchar>(j + 1)[i];
		}
	}

	// 在每次对不同窗口的遍历中将是否提取到点的标识符重置为0
	if_any_point_extracted = 0; 

	// 计算窗口中灰度的协方差矩阵
	// 下面两个for遍历框框的左上角坐标，代表窗口序号
	for (int j = 0; j < myImage.rows - Forstner_Window_Size; j++)
	{
		for (int i = 0; i < (myImage.cols - Forstner_Window_Size); i++)
		{
			int sum_gradient_u_2 = 0;
			int sum_gradient_v_2 = 0;
			int sum_gradient_v_u = 0;
			float w = 0, q = 0;

			// 建立当前像元处的梯度协方差矩阵
			N = Mat(2, 2, CV_16SC1, Scalar(0)); 

			// 下面两个for遍历每个窗口内的像素
			for (int n = 0; n < Forstner_Window_Size; n++)
			{
				for (int m = 0; m < Forstner_Window_Size; m++)
				{
					// Forstner算子似乎就这么相乘就行
					sum_gradient_u_2 += gradient_u.ptr<short>(j + n)[i + m] * gradient_u.ptr<short>(j + n)[i + m];
					sum_gradient_v_2 += gradient_v.ptr<short>(j + n)[i + m] * gradient_v.ptr<short>(j + n)[i + m];
					sum_gradient_v_u += gradient_v.ptr<short>(j + n)[i + m] * gradient_u.ptr<short>(j + n)[i + m];
				}
			}
			// 求N矩阵
			N.at<short>(0, 0) = sum_gradient_u_2;
			N.at<short>(0, 1) = sum_gradient_v_u;
			N.at<short>(1, 0) = sum_gradient_v_u;
			N.at<short>(1, 1) = sum_gradient_v_2;

			// 求兴趣值q和w
			if ((N.at<short>(0, 0) + N.at<short>(1, 1)) != 0)
			{
				w = 1.0 * (N.at<short>(0, 0) * N.at<short>(1, 1) - N.at<short>(0, 1) * N.at<short>(1, 0)) / (N.at<short>(0, 0) + N.at<short>(1, 1));
				q = 4.0 * (N.at<short>(0, 0) * N.at<short>(1, 1) - N.at<short>(0, 1) * N.at<short>(1, 0)) / ((N.at<short>(0, 0) + N.at<short>(1, 1)) * (N.at<short>(0, 0) + N.at<short>(1, 1)));
			}

			// w取经验值
			// 将提取的兴趣值赋给新的Mat

			// 计算像素所处的窗口
			window_j_present = j / Forstner_Refined_Window_Size;
			window_i_present = i / Forstner_Refined_Window_Size;

			if (q > Forstner_Threshold_q && w > Forstner_Threshold_w && q > max_q_in_window.at<float>(window_j_present, window_i_present) && w > max_w_in_window.at<float>(window_j_present, window_i_present))
			{
				// 直接在这里面进行候选点中极值点的选取
				// 更新窗口中最大值的值
				max_q_in_window.at<float>(window_j_present, window_i_present) = q;
				max_w_in_window.at<float>(window_j_present, window_i_present) = w;

				if (window_i_before == window_i_present && window_j_before == window_j_present && if_any_point_extracted != 0)
				{
					points_of_interest.pop_back();
				}

				points_of_interest.push_back(cvPoint(i + Forstner_Window_Size / 2, j + Forstner_Window_Size / 2));
				if_any_point_extracted = 1;
			}

			// 将q和w存入Q和W的Mat中
			image_Q.at<float>(j + Forstner_Window_Size / 2, i + Forstner_Window_Size / 2) = q;
			image_W.at<float>(j + Forstner_Window_Size / 2, i + Forstner_Window_Size / 2) = w;

			// 更新窗口序号缓存，用于判断下一次遍历是不是还在当前窗口
			window_i_before = window_i_present;
			window_j_before = window_j_present;
		}
	}
}

// 点绘制函数
void Forstner_Operator::draw(Mat& Result)
{
	for (int i = 0; i < points_of_interest.size(); i++)
		cv::circle(Result, points_of_interest[i], 4, cv::Scalar(255, 255, 100));
}