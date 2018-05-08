#include "Harris_Operator.h"
#include "Moravec_Operator.h"
#include "Forstner_Operator.h"

// 调用Moravec算子，Forstner算子，Harris算子的主函数

int main(int argc, char* argv[])
{
	char imageName[] = "data/image1.jpg";
				
	Mat image;		// 灰度图像，用于实际提取处理
	Mat colorImage; // 彩色图像，用于显示提取效果

	// 读入图像
	image = imread(imageName, CV_8U);	
	colorImage = imread(imageName, CV_LOAD_IMAGE_UNCHANGED);
	
	if (image.empty())                     
	{
		printf("Could not open or find the image. \n");
		return -1;
	}

	Mat result1 = colorImage.clone();
	Mat result2 = colorImage.clone();
	Mat result3 = colorImage.clone();

	// Moravec兴趣点算子调用
	Moravec_Operator moravec_operator;
	moravec_operator.extract(image);
	moravec_operator.draw(result1);

	// Forstner兴趣点算子调用
	Forstner_Operator forstner_operator;
	forstner_operator.extract(image);
	forstner_operator.draw(result2);

	// Harris兴趣点算子调用
	Harris_Operator harris_operator;
	harris_operator.extract(image);
	harris_operator.draw(result3);

	// 兴趣点结果输出
	namedWindow("Moravec兴趣点提取结果", WINDOW_AUTOSIZE);     
	imshow("Moravec兴趣点提取结果", result1);

	namedWindow("Forstner兴趣点提取结果", WINDOW_AUTOSIZE);      
	imshow("Forstner兴趣点提取结果", result2);

	namedWindow("Harris兴趣点提取结果", WINDOW_AUTOSIZE);     
	imshow("Harris兴趣点提取结果", result3);

	waitKey(0);
	return 0;
}