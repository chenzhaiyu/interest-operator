#include "Harris_Operator.h"

// Harris算子cpp文件

// 初始化Harris算子实例
Harris_Operator::Harris_Operator()
{
	// 保留下来方便以后实现统一划线，抑制极大值
	// 缺省为0，默认为没有提取出兴趣点
	if_any_point_extracted = 0;		
}

Harris_Operator::~Harris_Operator()
{

}

// 点提取函数
void Harris_Operator::extract(const Mat& image_input)
{
	// 仅接受对灰度图像的处理
	CV_Assert(image_input.depth() == CV_8U);

	// 建立高斯核，由X和Y方向的高斯核组成
	Mat GausiankernelX = getGaussianKernel(Harris_Window_Size, 0.5, CV_32FC1);
	Mat GausiankernelY = getGaussianKernel(Harris_Window_Size, 0.5, CV_32FC1);
	Gausiankernel = GausiankernelX * GausiankernelY.t();

	// 定义x和y方向的梯度矩阵
	gx = Mat(Harris_Window_Size, Harris_Window_Size, CV_16SC1, Scalar(0)); // 梯度可能为负，可能小于-128，Signed Char是short类型
	gy = Mat(Harris_Window_Size, Harris_Window_Size, CV_16SC1, Scalar(0)); // TODO: 梯矩阵初始值换换，可能对整体效果有影响
	Gx = Mat(Harris_Window_Size, Harris_Window_Size, CV_32F, Scalar(0));   // 经过高斯滤波的梯度矩阵
	Gy = Mat(Harris_Window_Size, Harris_Window_Size, CV_32F, Scalar(0));

	Intensity = Mat(Harris_Window_Size, Harris_Window_Size, CV_32F, Scalar(0));   // 强度值矩阵
	max_in_window = Mat(image_input.rows, image_input.cols, CV_32F, Scalar(0));	  // 当前窗口的最大强度值

	// 对不同的窗口进行遍历
	for (int j = 0; j < image_input.rows - 2 - Harris_Window_Size; j += Harris_Window_Size - 1)
	{
		for (int i = 0; i < (image_input.cols - 2 - Harris_Window_Size); i += Harris_Window_Size - 1)
		{
			static int j_buffer = 0;
			static int i_buffer = 0;

			// 对当前窗口的每个像元进行遍历
			for (int n = 0; n < Harris_Window_Size; n++)
			{
				for (int m = 0; m < Harris_Window_Size; m++)
				{
					gx.at<short>(n, m) = image_input.ptr<uchar>(j + n)[i + m + 1] - image_input.ptr<uchar>(j + n)[i + m];
					gy.at<short>(n, m) = image_input.ptr<uchar>(j + n + 1)[i + m] - image_input.ptr<uchar>(j + n)[i + m];
				}
			}
			// 对这个小窗口中的gx和gy分别进行高斯滤波，用不着在这边滤
			// GaussianBlur(gx, Gx, Size(3, 3), 0.9, 0.9);
			// GaussianBlur(gy, Gy, Size(3, 3), 0.9, 0.9);

			// 计算gxgy，矩阵乘法
			Mat _GxGx = gx.mul(gx);
			Mat _GyGy = gy.mul(gy);
			Mat _GxGy = gx.mul(gy);

			Mat __GxGx;
			Mat __GxGy;
			Mat __GyGy;

			_GxGx.convertTo(__GxGx,CV_32F);
			_GxGy.convertTo(__GxGy, CV_32F);
			_GyGy.convertTo(__GyGy, CV_32F);


			// 尝试着对M阵再进行滤波，后面计算det(M)不再为0，是0没有意义
			GaussianBlur(__GxGx, GxGx, Size(3, 3), 0.1, 0.1);
			GaussianBlur(__GyGy, GyGy, Size(3, 3), 0.1, 0.1);
			GaussianBlur(__GxGy, GxGy, Size(3, 3), 0.1, 0.1);
			
			
			// 下面的代码逐像素进行特征值计算，书上写的有问题，det(M)恒为0，强度值恒为负，没有意义
			// 开下面的小窗口循环是为了计算当前窗口中强度值Intensity
			for (int n = 0; n < Harris_Window_Size - 1; n++)
			{
				for (int m = 0; m < Harris_Window_Size - 1; m++)
				{
					float det_M = GxGx.at<float>(n, m) * GyGy.at<float>(n, m) + GxGy.at<float>(n, m) * GxGy.at<float>(n, m);
					float k_tr_2_M = GxGx.at<float>(n, m) + GyGy.at<float>(n, m);
					Mat Covariance(2, 2, CV_32F, Scalar(0));
					Mat eigenValue;
					Mat eigenVector;

					/*
					Covariance.at<float>(0, 0) = GxGx.at<short>(n, m) * GxGx.at<short>(n, m);
					Covariance.at<float>(0, 1) = GxGy.at<short>(n, m) * GxGy.at<short>(n, m);
					Covariance.at<float>(1, 0) = GxGy.at<short>(n, m) * GxGy.at<short>(n, m);
					Covariance.at<float>(1, 1) = GyGy.at<short>(n, m) * GyGy.at<short>(n, m);
					*/

					//printf("%f\n", Covariance.at<float>(0, 0));
					//printf("%f\n", Covariance.at<float>(0, 1));
					//printf("%f\n", Covariance.at<float>(1, 0));
					//printf("%f\n", Covariance.at<float>(1, 1));

					// 计算特征值和特征向量，只是为了和Forstner简化后的式子计算的结果进行比较，发现结果差不多
					eigen(Covariance, eigenValue, eigenVector);
					
					// 分别为两个特征值
					float lambda_1 = eigenValue.at<float>(0, 0);
					float lambda_2 = eigenValue.at<float>(1, 0);
					
					k_tr_2_M = 0.04 * pow(k_tr_2_M, 2);
					Intensity.at<float>(n, m) = det_M + k_tr_2_M;

					// 比现有小窗中最大强度值还大才更新
					if (Intensity.at<float>(n, m) > max_in_window.at<float>(j, i) && Intensity.at<float>(n, m) > 100000)
					{
						// 更新最大强度值
						max_in_window.at<float>(j, i) = Intensity.at<float>(n, m);
						if (i_buffer == i && j_buffer == j && if_any_point_extracted != 0)
						{
							points_of_interest.pop_back();
						}
						points_of_interest.push_back(cvPoint(i + m, j + n));
						if_any_point_extracted = 1;
					}
					// 更新窗口索引
					i_buffer = i;
					j_buffer = j;
				}
			}
			
			
		}
	}
}

// 点绘制函数
void Harris_Operator::draw(Mat& Result)
{
	for (int i = 0; i < points_of_interest.size(); i++)
		cv::circle(Result, points_of_interest[i], 4, cv::Scalar(255, 255, 100));
}