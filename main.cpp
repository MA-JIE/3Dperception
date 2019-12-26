#include<fstream>
#include<string>
#include <3D_perception.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main(int argc, char ** argv)
{
    string label_file ="../label.txt";
    // 设置相机内部参数
    vpCameraParameters cam(7.215377000000e+02, 7.215377000000e+02,6.095593000000e+02,1.728540000000e+02);
    VisualOdom vo(cam);
    vo.setNormalGuess(0,1,0);
    vector<cv::Mat> images;
    //读取图像信息
    cv::Mat img1 = cv::imread("../images/000000.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat img2 = cv::imread("../images/000001.png", CV_LOAD_IMAGE_COLOR);
    images.push_back(img1);
    images.push_back(img2);
    std::vector<vpRowVector> bbox;
    //读取检测框信息
    ifstream fin(label_file);
    if (!fin) {
       cout << "cannot find trajectory file at " << label_file << endl;
       return 1;
     }
    while (!fin.eof()) {
      double number, xmin,ymin,xmax,ymax;
      fin >> number>>xmin>>ymin>>xmax>>ymax;
      vpRowVector bbox_(4);
      bbox_[0]=xmin;  bbox_[1]=ymin;  bbox_[2]=xmax;  bbox_[3]=ymax;
      bbox.push_back(bbox_);
    }
    vpColVector X_left(3);
    vpColVector X_right(3);
    //输入图像(kitti)以及检测框，输出车辆3D位置(X_left,X_right)
    vo.process(img1,img2,bbox[0],bbox[1],X_left,X_right);
    cout<<"左图车辆的三维空间位置为"<<endl<<"x坐标为："<<X_left[0]<<
          endl<<"y坐标为："<<X_left[1]<<endl<<"距离为："<<X_left[2]<<endl;
    cout<<"右图车辆的三维空间位置为"<<endl<<"x坐标为："<<X_right[0]<<
          endl<<"y坐标为："<<X_right[1]<<endl<<"距离为："<<X_right[2]<<endl;

}
