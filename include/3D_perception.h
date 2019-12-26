#ifndef VISUALODOM_H
#define VISUALODOM_H
#include <visp/vpPoint.h>
#include <visp/vpHomogeneousMatrix.h>
#include <vector>
#include <algorithm>
#include <visp/vpSubColVector.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <visp/vpHomography.h>
#include <time.h>
#include <visp/vpMeterPixelConversion.h>
#include <visp/vpSubMatrix.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
//g2o
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
// 单应性矩阵的结构体
struct Homography
{
    vpRotationMatrix R;
    vpTranslationVector t, n;
   //用于重构H矩阵
    void buildFrom(cv::Mat1d _R, cv::Mat1d _t, cv::Mat1d _n)
    {
        for(unsigned int i=0;i<3;++i)
        {
            t[i] = _t(i,0);
            n[i] = _n(i,0);
            for(unsigned int j=0;j<3;++j)
                R[i][j] = _R(i,j);
        }
    }
    vpMatrix H()
    {
        return ((vpMatrix) R) + t*n.t();
    }

    void print()
    {
        vpThetaUVector tu(R);
        std::cout << "R: " << tu.t()*180/CV_PI << std::endl;
        std::cout << "t: "  << t.t() << std::endl;
        std::cout << "n: " << n.t() << std::endl;
    }
};
class VisualOdom
{
public:
    // 构造函数，传入相机内部参数
    VisualOdom(const vpCameraParameters cam);

    // 初始n向量的预测函数
    inline void setNormalGuess(double x, double y, double z)
    {
        n_guess[0] = x;
        n_guess[1] = y;
        n_guess[2] = z;
        n_guess /= n_guess.euclideanNorm();
    }
    // 3D感知具体细节
    bool process(cv::Mat &im1,cv::Mat&im2,vpRowVector &bbox_im1,vpRowVector &bbox_im2,vpColVector &X_left, vpColVector &X_right);
    //最优化部分
    vpMatrix optim_d(const std::vector<cv::Point3f> &points_3d,
                   const std::vector<cv::Point2f> &points_right,
                   g2o::SE3Quat &pose,g2o::SE3Quat &pose_init);
    //相机高度估计值函数
   inline void get_d_estimated(const vpMatrix& X)
    {
        vpRowVector d(X.getCols());
        for(unsigned int i=0;i<X.getCols();++i)
        {
            d[i]=n_estimated * X.getCol(i);
        }
        double d_esti = vpRowVector::mean(d);
        d_estimated = d_esti;
    }
    //得到左图中车的空间位置
   inline void get_left_position(const double &xmin, const double & xmax, const double & ymax)
    {
       cv::Point2f car_image;
       car_image.x = (xmin + xmax)/2.0;
       car_image.y = ymax;
       vpMatrix car_homo(3,1);
       car_homo[0][0] = car_image.x;
       car_homo[1][0] = car_image.y;
       car_homo[2][0] = 1.0;
       auto car_norm = Ki*car_homo;
       double depth =d_estimated/(n_estimated.t()*car_norm.getCol(0));
       space_coord.resize(3);
       space_coord[0] = car_norm.getCol(0)[0]*depth;
       space_coord[1] = car_norm.getCol(0)[1]*depth;
       space_coord[2] = car_norm.getCol(0)[2]*depth;
//       std::cout<<"n估计值: "<<n_estimated<<std::endl;
//       std::cout<<"d估计值: "<<d_estimated<<std::endl;
//       std::cout<<"左车深度为: "<<depth<<std::endl;
//       std::cout<<"左车归一化坐标为: "<<car_norm.getCol(0)<<std::endl;


    }
   //得到右图中车的空间位置
  inline void get_right_position(const double &xmin, const double & xmax, const double & ymax)
   {
      cv::Point2f car_image;
      car_image.x = (xmin + xmax)/2.0;
      car_image.y = ymax;
      vpMatrix car_homo(3,1);
      car_homo[0][0] = car_image.x;
      car_homo[1][0] = car_image.y;
      car_homo[2][0] = 1.0;
      auto car_norm = Ki*car_homo;
      double depth =d_guess/(n_guess.t()*car_norm.getCol(0));
      space_coord_right.resize(3);
      space_coord_right[0] = car_norm.getCol(0)[0]*depth;
      space_coord_right[1] = car_norm.getCol(0)[1]*depth;
      space_coord_right[2] = car_norm.getCol(0)[2]*depth;
//      std::cout<<"n估计值: "<<n_guess<<std::endl;
//      std::cout<<"d估计值: "<<d_guess<<std::endl;
//      std::cout<<"右车深度为: "<<depth<<std::endl;
//      std::cout<<"右车归一化坐标为: "<<car_norm.getCol(0)<<std::endl;

   }

protected:
    //定义的保护变量引用了Opencv以及visp(视觉伺服)的库
    cv::Mat1d Kcv;
    vpMatrix K, Ki;

    // 是否是第一帧图片进入
    bool first_time = true;

    // 用于特征匹配的变量
    cv::Mat im1,imatches;
    cv::Ptr<cv::Feature2D> feature2d;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2, Hp, mask;
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    // 单应性矩阵相关变量
    double d_guess = 1.7, d_estimated;//由kitti数据可知预测值为1.7米
    vpColVector n_guess , n_estimated;
    std::vector<Homography> H;
    std::vector<cv::Mat> R, t, nor;
    Eigen::Matrix3d R_estimated;
    Eigen::Vector3d t_estimated;
    //存放汽车左右帧空间坐标的变量
    vpColVector space_coord, space_coord_right;


    //将像素点转化为归一化坐标
    vpMatrix cvPointToNormalized(const std::vector<cv::Point2f> &p)
    {
        vpMatrix X(3,p.size());
        // build homogeneous pixel coordinates
        for(unsigned int i=0;i<p.size();++i)
        {
            X[0][i] = p[i].x;
            X[1][i] = p[i].y;
            X[2][i] = 1;
        }
        // to normalized coordinates
        return Ki*X;
    }
    //将归一化坐标转化为空间坐标
    std::vector<cv::Point3f> cvToVector3d(vpMatrix &X1)
    {
        std::vector<cv::Point3f> points_3d;
        for(unsigned int i=0;i<X1.getCols();++i)
        {
            //获取深度信息
            double Z1 = d_guess/(n_estimated.t()*X1.getCol(i));
            points_3d.push_back(cv::Point3f(X1.getCol(i)[0],X1.getCol(i)[1],X1.getCol(i)[2])*Z1);

        }
        return points_3d;
    }
};
#endif // VISUALODOM_H
