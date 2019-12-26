#include <3D_perception.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
using std::vector;
using namespace std;
using namespace cv;
using namespace Eigen;
static double threshol=1.0;
VisualOdom::VisualOdom(const vpCameraParameters cam) :matcher(cv::NORM_L2)
{
    //导入相机内参
    Kcv = (cv::Mat1d(3, 3) <<
           cam.get_px(), 0, cam.get_u0(),
           0, cam.get_py(), cam.get_v0(),
           0, 0, 1);
    K = cam.get_K();
    Ki = cam.get_K_inverse();//inv(K)，方便求解对应的归一化坐标
    //如果之间没有图像进入函数，设置为true
    first_time = true;
    //relative_to_initial = _relative_to_initial;
    // 创建SIFT匹配
    feature2d = cv::xfeatures2d::SIFT::create();
    // 设置向量n的猜测值(预测值)
    n_guess.resize(3);
    n_guess[2] = 0;
}
//图优化部分，优化空间坐标以及相机位姿
vpMatrix VisualOdom::optim_d(const std::vector<cv::Point3f> &points_3d,
                             const std::vector<cv::Point2f> &points_right,
                             g2o::SE3Quat &pose,g2o::SE3Quat &pose_init)
{
    g2o::SparseOptimizer optimizer;
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver (new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>());
    std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg * algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm( algorithm );
    optimizer.setVerbose( false );
    g2o::VertexSE3Expmap* cam_pose = new g2o::VertexSE3Expmap();
    cam_pose->setId(0);
    cam_pose->setEstimate( pose_init );
    optimizer.addVertex(cam_pose);
    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true );
        optimizer.addVertex ( point );
    }
    // 设定相机内部参数
    g2o::CameraParameters* camera = new g2o::CameraParameters(K[0][0],Eigen::Vector2d(K[0][2], K[1][2]), 0);
    camera->setId(0);
    optimizer.addParameter( camera );
    // 设定边
    index = 1;
    for ( const Point2f p:points_right )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, cam_pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization(0);
    optimizer.optimize ( 1000 );
    cout<<endl<<"after optimization:"<<endl;
    Eigen::Isometry3d T =cam_pose ->estimate();
    //得出相机位姿估计值
    R_estimated = T.rotation();
    t_estimated = T.translation();
    vpMatrix points_3d_estimated(3,points_3d.size());
    for(int i=0; i<points_3d.size();i++)
    {
        g2o::VertexSBAPointXYZ* point = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+1));
        Eigen::Vector3d pos = point->estimate();
        points_3d_estimated[0][i] = pos(0);
        points_3d_estimated[1][i] = pos(1);
        points_3d_estimated[2][i] = pos(2);
        //cout<<"优化后的空间点坐标为："<<pos(0)<<""<<pos(1)<<""<<pos(2)<<endl;
    }
    return points_3d_estimated;
}
// 处理图像并输出车辆三维位置
bool VisualOdom::process(cv::Mat &im1, cv::Mat&im2,vpRowVector &bbox_im1,vpRowVector &bbox_im2,vpColVector &X_left, vpColVector &X_right)
{
    cv::Mat img_gray1;
    cv::Mat img_gray2;
    cv::cvtColor(im1, img_gray1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(im2, img_gray2, cv::COLOR_RGB2GRAY);
    //检测兴趣点，绘制描述子
    feature2d->detectAndCompute(img_gray1,cv::noArray(),kp1,des1);
    //检测兴趣点，绘制描述子
    feature2d->detectAndCompute(img_gray2,cv::noArray(), kp2, des2);
    // 匹配并存储特征点
    matcher.match(des1, des2, matches);
    std::vector<cv::Point2f> matched1, matched2;
    for(auto &m: matches)
    {
        matched1.push_back(kp1[m.queryIdx].pt);
        matched2.push_back(kp2[m.trainIdx].pt);
    }
    // 使用RANSAC求解H矩阵
    double xmin, xmax,ymin,ymax;
    xmin = bbox_im1[0]; ymin=bbox_im1[1];xmax=bbox_im1[2];ymax = bbox_im1[3];
    //xmin = 296.744956; xmax = 455.226042; ymin = 161.752147; ymax=292.372804;//还要改进，读取参数
    double d = xmax - xmin;
    double xmin1, xmax1,ymin1,ymax1;
    xmin1 = bbox_im2[0]; ymin1=bbox_im2[1];xmax1=bbox_im2[2];ymax1 = bbox_im2[3];
    // xmin1 = 294.898777; xmax1 = 452.199718; ymin1 = 156.024256; ymax1=284.621269 ;
    double d1 = xmax1 - xmin1;
    std::vector<cv::Point2f>matched1_ROI, matched2_ROI;
    for(unsigned int i=0; i<matched1.size();i++)
    {
        if(matched1[i].x > (xmin-threshol*d) && matched1[i].x < (xmax+threshol*d) && matched1[i].y > ymax
                &&matched2[i].x > (xmin1 -threshol*d1) && matched2[i].x< (xmax1+threshol*d1)&&matched2[i].y>ymax1
                &&abs(matched1[i].x-matched2[i].x)<10.0 && abs(matched1[i].y-matched2[i].y)<10.0)
        {
            matched1_ROI.push_back(matched1[i]);
            matched2_ROI.push_back(matched2[i]);
        }
    }
    cout<<"updated matchers 1 is:"<<endl<<matched1_ROI<<endl;
    cout<<"updated matchers 2 is:"<<endl<<matched2_ROI<<endl;
    std::vector<char> inliers;
    Hp = findHomography(matched1_ROI,matched2_ROI,inliers,cv::RANSAC,3);
    cout<<"ROI中匹配点个数为: "<<matched1_ROI.size()<<endl;
    int n = 0;//解的个数,一般有四个解
    n=cv::decomposeHomographyMat(Hp,Kcv,R,t,nor);
    cout << "我们发现了 " << n << " 个解" << endl;
    //重构H矩阵
    H.resize(n);
    for(unsigned int i=0;i<n;++i)
    {
        H[i].buildFrom(R[i], t[i], nor[i]);
        cout<<"n"<<i<<" is "<<endl<<nor[i]<<endl;
        cout<<"t"<<i<<" is "<<endl<<t[i]<<endl;

    }
    //归一化坐标 X1 = inv(K).p
    vpMatrix X1 = cvPointToNormalized(matched1_ROI);
    for(unsigned int j=0;j<matched1_ROI.size();++j)
    {
        for(unsigned int i=0;i<H.size();++i)
        {
            if(H[i].n.t() * X1.getCol(j) < 0)
            {
                cout << "Z为负值，此解为无效解！！！" << endl;
                cout << "无效解为： "<<H[i].n.t()<<endl;
                H.erase(H.begin()+i);
                break;
            }
        }
    }
    if(H.size() == 0)
    {
        if (first_time)
        {
            cout<<"第一帧检测失败！"<<endl;
            first_time = false;
            return false;
        }
        else
            X_right =  space_coord_right;//返回上一帧空间坐标
        X_left = space_coord_right;
        return true;
    }

    //我们假设最优解是H[0]
    int idx = 0;
    // 如果剩下两个解，继续检查
    if(H.size() == 2)
    {
        //选择与n_guess更接近的解
        if((H[0].n.t() * n_guess < H[1].n.t() * n_guess))
        {
            idx = 1;
        }
    }
    //如果剩下一个解，默认为最优解
    if(H.size() == 1)
    {
        cout << "Best solution found" << endl;
    }
    //得出左图n向量的估计值
    n_estimated = H[idx].n;
    cout<<"向量n的最终解为： "<<endl<<n_estimated<<endl;
    vpRotationMatrix R_ = H[idx].R;
    vpTranslationVector t_ = H[idx].t;
    Eigen::Vector3d translation(t_[0]*d_guess,t_[1]*d_guess,t_[2]*d_guess);
    vector<Point3f> pts_3d;//存放左帧的空间坐标
    //转换为基于左相机的空间坐标
    pts_3d = cvToVector3d(X1);
    //得出初始的相机位姿
    Matrix3d R__=Matrix3d::Zero();
    R__(0,0)=1;R__(1,1)=1;R__(2,2)=1;
    g2o::SE3Quat pose_init(R__,translation);
    g2o::SE3Quat pose;
    //图优化(优化相对于左图的空间坐标以及相机位姿)
    vpMatrix X1_estimated=optim_d(pts_3d,matched2_ROI,pose,pose_init);
    //得出左图高度d的估计值,d_guess = 1.7
    get_d_estimated(X1_estimated);
    cout << "初始预测的R is :" <<endl<< R_ << endl;
    cout<<"R的估计值 is :"<<endl<<R_estimated<<endl;
    cout<<"初始预测的t is :  "<< endl<<translation<<endl;
    cout<<"t的估计值 is :"<<endl<<t_estimated<<endl;
    cout<<"n 的估计值为 : "<<endl<<n_estimated << endl;
    cout<<"d 的估计值为"<<endl<<d_estimated<<endl;
    //得出下一帧(右图)n向量的预测值
    vpRotationMatrix R_cv;
    for(int i=0; i<R_estimated.cols();i++)
    {
        R_cv[0][i] = R_estimated(0,i);
        R_cv[1][i] = R_estimated(1,i);
        R_cv[2][i] = R_estimated(2,i);
    }
    //右帧d的估计值
    vpColVector X2;
    vpRowVector depth_2(X1_estimated.getCols());
    vpTranslationVector trans_esti;
    trans_esti[0]=t_estimated[0];trans_esti[1]=t_estimated[1];trans_esti[2]=t_estimated[2];
    for (int i =0; i< X1_estimated.getCols(); i++)
    {
        X2 = R_cv*X1_estimated.getCol(i)+trans_esti;
        depth_2[i] = n_estimated.t()*X2;
    }
    //右帧的预测值，因为有优化后的R，t，以及空间坐标得到，所以也可看作为右帧的估计值
    d_guess = vpRowVector::mean(depth_2);//右帧d的估计值
    n_guess =R_cv*n_estimated;//右帧n的预测值(估计值）
    //取左右图检测框下边中间像素点,并计算出左图车的三维坐标
    get_left_position(xmin,xmax,ymax);
    get_right_position(xmin1,xmax1,ymax1);
    cout<<"右帧d的预测值为："<<d_guess<<endl;//1.67
    X_left = space_coord;
    X_right = space_coord_right;
    return true;

}
