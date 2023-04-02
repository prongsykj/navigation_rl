#include "EntryClass.hpp"
#include <image_transport/image_transport.h>

using namespace std;
using namespace ros;
using namespace tf;

#define H_LIDAR_INTERPOLATION  (true) //是否对雷达数据水平方向插值
#define H_LIDAR_INTERPOLATION_IN_RADIAN  (0.00349)  //弧度
#define H_LIDAR_INTERPOLATION_IN_RADIAN_3  (0.002)  //弧度
#define V_LIDAR_INTERPOLATION  (true) //是否对雷达数据垂直方向插值
#define V_UP_LIDAR_INTERPOLATION_IN_PIXEL  (200)  //向上垂直插值个数
#define V_DOWN_LIDAR_INTERPOLATION_IN_PIXEL  (30)  //向下垂直插值个数

#define DEG2RAD(x) ((x)*M_PI/180.)
#define RAD2DEG(x) ((x)*180./M_PI)

EntryClass::EntryClass() : 
    nh_private("~"), 
    camera_lidar_tf_ok_(false), 
    camera_info_ok_(false), 
    usingObjs(false), 
    image_frame_id("")
{
    double freq;
    nh_private.param("freq", freq, 1.0);

    timer_ = nh_private.createTimer(ros::Duration(1.0 / max(freq, 1.0)), &EntryClass::mainLoop, this);

    fusion_cloud_pub_ = node_h.advertise<sensor_msgs::PointCloud2>("colored_point_cloud", 1);

    depth_laser_to_image_ = node_image.advertise<sensor_msgs::Image>("depth_laser_image", 1);

    intrinsics_sub_ = node_h.subscribe("/camera/rgb/camera_info", 1, &EntryClass::intrinsicValueCallback, this);

    laserScan_sub_ = node_h.subscribe("/scan", 1, &EntryClass::laserScanCallback, this);

    cameraImage_sub_ = node_h.subscribe("/camera/rgb/image_raw", 1, &EntryClass::cameraImageCallback, this);
}

EntryClass::~EntryClass()
{
    
}

//获取摄像机内参参数及畸变系数
void EntryClass::intrinsicValueCallback(const sensor_msgs::CameraInfo::ConstPtr &intrinsic_value_msg)
{
    image_frame_size.height = intrinsic_value_msg->height;
    image_frame_size.width = intrinsic_value_msg->width;

    // 相机内参
    camera_intrinsic_value = cv::Mat(3, 3, CV_64F);

    //相机内参变换矩阵3x3,把3D点投影到2D像素平面时使用
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            camera_intrinsic_value.at<double>(row, col) = intrinsic_value_msg->K[row * 3 + col];
        }
    }

    // 相机畸变参数. For "plumb_bob"模式, the 5 parameters are: (k1, k2, t1, t2, k3).
    distortion_coefficients = cv::Mat(1, 5, CV_64F);
    for (int col = 0; col < 5; col++)
    {
        distortion_coefficients.at<double>(col) = intrinsic_value_msg->D[col];
    }

    // 投影系数,获取投影矩阵3x4的数组的fx,fy,cx,cy元素
    fx = static_cast<float>(intrinsic_value_msg->P[0]);
    fy = static_cast<float>(intrinsic_value_msg->P[5]);
    cx = static_cast<float>(intrinsic_value_msg->P[2]);
    cy = static_cast<float>(intrinsic_value_msg->P[6]);

    intrinsics_sub_.shutdown(); //关闭intrinsics subscriber

    camera_info_ok_ = true;
    ROS_INFO("intrinsicValueCallback : camera instrinsics get");
}

void EntryClass::cameraImageCallback(const sensor_msgs::Image::ConstPtr &image_msg) //const sensor_msgs::Image &image_msg)
{
    //ROS_INFO("cameraImageCallback at time %f ", ros::Time::now().toSec());
    if ( !camera_info_ok_ ) {
        ROS_INFO("cameraImageCallback : waiting for intrinsics to be availiable");
        return;
    }

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_msg, "bgr8");
    cv::Mat image = cv_image->image;

    //TODO: 图像去畸变, 使用相机内参和畸变系数可以图像去畸变
    if( true )
        cv::undistort(image, current_image_frame, camera_intrinsic_value, distortion_coefficients);
    else
        current_image_frame = image;

    //最近image frame 信息
    image_frame_id = image_msg->header.frame_id;
    image_frame_size.height = current_image_frame.rows;
    image_frame_size.width = current_image_frame.cols;
}

void EntryClass::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr &laserScan_msg)
{
    //ROS_INFO("laserScanCallback at time %f ", ros::Time::now().toSec());
    
    if ( !camera_info_ok_ )
    {
        ROS_INFO("laserScanCallback : waiting for camera intrinsics.");
        return;
    }

    /*current_image_frame: 已经保存当前相机图像: cv::Mat*/
    if ( current_image_frame.empty() || image_frame_id == "" )
    {
        ROS_INFO("laserScanCallback : waiting for image frame ");
        return;
    }

    if ( !camera_lidar_tf_ok_ )
    {
        // 从tf树里面寻找变换关系
        camera_lidar_tf = findTransform(image_frame_id, laserScan_msg->header.frame_id);
    }

    if( !camera_lidar_tf_ok_ ){
        ROS_INFO("laserScanCallback : waiting for camera lidar tf.");
        return;
    }

    //单线激光雷达扫描数据转单线点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_msg->header.frame_id = laserScan_msg->header.frame_id;//"laser_link";
    cloud_msg->height = 1;
    //cloud_msg->width = image_frame_size.width;
    //cloud_msg->points.push_back(pcl::PointXYZ(1.0, 2.0, 3.0));
    transLaserScanToPointCloud(laserScan_msg,cloud_msg);//经过极坐标数据转笛卡尔坐标数据处理.

    cloud_msg->width = cloud_msg->points.size();

    //ROS_INFO("laserScanCallback 垂直插值前: cloud_msg->width=%d,height=%d,size=%d",cloud_msg->width,cloud_msg->height,cloud_msg->points.size());
#ifdef V_LIDAR_INTERPOLATION
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    point_cloud->header.frame_id = laserScan_msg->header.frame_id;//"laser_link";
    point_cloud->height = V_UP_LIDAR_INTERPOLATION_IN_PIXEL + V_DOWN_LIDAR_INTERPOLATION_IN_PIXEL + 1;
    point_cloud->width = cloud_msg->width;

    //向上插入
    for( int up = 1; up <= V_UP_LIDAR_INTERPOLATION_IN_PIXEL; up++ ){
        for(int i = 0; i < cloud_msg->points.size(); i++){
            point_cloud->points.push_back(pcl::PointXYZ(cloud_msg->points[i].x, cloud_msg->points[i].y, up/200.0));//这里z坐标应该为正数
        }
    }
    //插入原始行
    for(int i = 0; i < cloud_msg->points.size(); i++){
        point_cloud->points.push_back(pcl::PointXYZ(cloud_msg->points[i].x, cloud_msg->points[i].y, cloud_msg->points[i].z));//这里z坐标应该为0
    }
    //向下插入
    for( int down = 1; down <= V_DOWN_LIDAR_INTERPOLATION_IN_PIXEL; down++ ){
        for(int i = 0; i < cloud_msg->points.size(); i++){
            point_cloud->points.push_back(pcl::PointXYZ(cloud_msg->points[i].x, cloud_msg->points[i].y, -down/200.0));//这里z坐标开始为负数
        }
    }

    cloud_msg->height = point_cloud->height;
    cloud_msg->width = point_cloud->width;
    cloud_msg->points.clear();
    for(int i = 0; i < point_cloud->points.size(); i++) 
        cloud_msg->points.push_back(point_cloud->points[i]);
#endif
    //ROS_INFO("laserScanCallback 垂直插值后: cloud_msg->width=%d,height=%d,size=%d",cloud_msg->width,cloud_msg->height,cloud_msg->points.size());

    // 存储处理后的点云
    pcl::PointXYZRGB colored_3d_point;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outColorPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ROS_Cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    

    cv::Mat img8 = cv::Mat::zeros(image_frame_size.height, image_frame_size.width, CV_32FC1);

    std::vector<pcl::PointXYZ> cam_cloud(cloud_msg->points.size());
    int ans_row = 0;
    int ans_col = 0;

    for(int i = 0; i < cloud_msg->points.size(); i++) 
    {
        //把点云里面的3D坐标从激光雷达坐标系,变换到摄像机的坐标系
        cam_cloud[i] = transformPoint(cloud_msg->points[i], camera_lidar_tf);
        // 再使用相机内参将三维空间点投影到像素平面
        int col = int(cam_cloud[i].x * fx / cam_cloud[i].z + cx);
        int row = int(cam_cloud[i].y * fy / cam_cloud[i].z + cy);
        //ROS_INFO("laserScanCallback :  pixel =(%d,%d)",col,row);
        
        //根据映射后的坐标,从当前摄像机的当前图像帧current_image_frame上获取颜色值,并保存对应的3D位置信息
        if ((col >= 0) && (col < image_frame_size.width) && (row >= 0) && (row < image_frame_size.height) )
                //&& (img8.at<uchar>(row, col)==0) ) 
        {
            //生成的点云坐标
            colored_3d_point.x = cloud_msg->points[i].x;  
            colored_3d_point.y = cloud_msg->points[i].y; 
            colored_3d_point.z = cloud_msg->points[i].z;

            ans_row = max(ans_row,row);
            ans_col = max(ans_col,col);
            //ROS_INFO("rowmax=%d,colmax=%d",ans_row,ans_col);

            img8.at<uchar>(row, col) = 1;//pow(pow(cloud_msg->points[i].x,2)+pow(cloud_msg->points[i].y,2),0.5);
            

            cv::Vec3b rgb_pixel = current_image_frame.at<cv::Vec3b>(row, col);
            colored_3d_point.r = rgb_pixel[2];
            colored_3d_point.g = rgb_pixel[1];
            colored_3d_point.b = rgb_pixel[0];
            outColorPointCloud->points.push_back(colored_3d_point);
        }
    }
    // cout<<outColorPointCloud->points.size()<<endl;
    // ROS_INFO("shape0=%d,shape1=%d",img8.size[0],img8.size[1]);

    pcl::RandomSample<pcl::PointXYZRGB> rs;
    rs.setInputCloud(outColorPointCloud);
    rs.setSample(3000);
    rs.filter(*ROS_Cloud);

    // cout<<ROS_Cloud->points.size()<<endl;
    
    sensor_msgs::PointCloud2 out_colored_cloud_msg;
    pcl::toROSMsg(*ROS_Cloud, out_colored_cloud_msg);
    out_colored_cloud_msg.header = laserScan_msg->header;

    //发布"colored_point_cloud" Topic
    fusion_cloud_pub_.publish(out_colored_cloud_msg);
    ROS_Cloud->points.clear();
    outColorPointCloud->points.clear();

    // normalize(img8,img8,1.0,0.0,cv::NORM_MINMAX);
    // img8.convertTo(img9, CV_8UC1, 255, 0);
    // sensor_msgs::ImagePtr msg = cv_bridge::CvImage(laserScan_msg->header, "32FC1", img8).toImageMsg();
    // // cout<<img8<<endl;
    // depth_laser_to_image_.publish(msg);


}
//将激光雷达数据从极坐标系转为笛卡尔坐标系
void EntryClass::transLaserScanToPointCloud(const sensor_msgs::LaserScan::ConstPtr &laserScan_msg,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_msg)
{
    int pointCount1 = laserScan_msg->scan_time / laserScan_msg->time_increment;
    int pointCount = laserScan_msg->ranges.size();
    //ROS_INFO("transScanToPoints : laserScan_msg->ranges:  pointCount1=%d,pointCount=%d",pointCount1,pointCount);

    std::vector<pcl::PointXYZ>  pointVector; 
    float lastLeftRadian = -1;
    float lastRightRadian = -1;
    //range data [m] (Note: values < range_min or > range_max should be discarded)
    //对每一个激光雷达数据单独进行操作
    for(int i = 0; i < pointCount; i++)
    {   
        float distance = 3.5;
        // float distance = laserScan_msg->ranges[i];
        //过滤无效距离
        if(laserScan_msg->ranges[i] < laserScan_msg->range_max && laserScan_msg->ranges[i] > laserScan_msg->range_min)
        {
            distance = laserScan_msg->ranges[i];
        }
        if( laserScan_msg->ranges[i] < laserScan_msg->range_min)
            distance = laserScan_msg->range_min;


        /*
        小车载体坐标系是前向X轴正向,左侧Y轴正向;
        而雷达的极坐标系为前侧为0轴,后向180度.顺时针旋转,到360度
        所以,返回的雷达数据中,前方摄像头可视区域内,雷达数据存储顺序是[320,360] + [0,40],单位度.
        */
        float radian = laserScan_msg->angle_min+laserScan_msg->angle_increment*i;
        float degree = RAD2DEG(radian);
        //ROS_INFO("transScanToPoints : laserScan_msg->ranges:  degree=%f,distance=%f",degree,distance);
        /*
        雷达数据转换到和载体坐标系一致后,有效区域数据顺序为:[320,360]和[0,40]
	    */

        /*********这样计算的话，正前方是x的正方向，正右方是y的正方向************/

        //左前方雷达数据,存放为[320,360]
        if( degree > 320 and degree <= 360 ){
            float x = cos(radian) * distance; //已经做了坐标系变化同一位置,故直接计算x,y值
            float y = sin(radian) * distance;
            cloud_msg->points.insert(cloud_msg->points.begin(),pcl::PointXYZ(x, y, 0));//z坐标暂存变换后极坐标角度
            
            //水平插值
#ifdef H_LIDAR_INTERPOLATION
            float step = H_LIDAR_INTERPOLATION_IN_RADIAN;
            if( distance > 3.0f )
                step = H_LIDAR_INTERPOLATION_IN_RADIAN_3;
            if( lastLeftRadian >= 0 && (radian-lastLeftRadian) > step ){
                for(float rad = lastLeftRadian; rad < radian; rad+= step){
                    float x2 = cos(rad) * distance; //已经做了坐标系变化同一位置,故直接计算x,y值
                    float y2 = sin(rad) * distance;
                    cloud_msg->points.insert(cloud_msg->points.begin(),pcl::PointXYZ(x2, y2, 0));//z坐标暂存变换后极坐标角度
                }
            }
            
            lastLeftRadian = radian;
#endif
        }
       
        //右前方雷达数据，存放为[0，40]
        if( degree >= 0 && degree <= 40 ){
            float x = cos(radian) * distance; //已经做了坐标系变化同一位置,故直接计算x,y值
            float y = sin(radian) * distance;
            pointVector.insert(pointVector.begin(),pcl::PointXYZ(x, y, 0));//z坐标暂存变换后极坐标角度

            //水平插值
#ifdef H_LIDAR_INTERPOLATION
            if( lastRightRadian >= 0 && (radian-lastRightRadian) > H_LIDAR_INTERPOLATION_IN_RADIAN ){
                for(float rad = lastRightRadian; rad < radian; rad+= H_LIDAR_INTERPOLATION_IN_RADIAN){
                    float x2 = cos(rad) * distance; //已经做了坐标系变化同一位置,故直接计算x,y值
                    float y2 = sin(rad) * distance;
                    pointVector.insert(pointVector.begin(),pcl::PointXYZ(x2, y2, 0));//z坐标暂存变换后极坐标角度
                }
            }
            
            lastRightRadian = radian;
#endif
        }
    }

    for( int i = 0; i < pointVector.size(); i++ )
        cloud_msg->points.push_back(pcl::PointXYZ(pointVector.at(i).x, pointVector.at(i).y, pointVector.at(i).z));//z坐标暂存变换后极坐标角度
}

tf::StampedTransform EntryClass::findTransform(const std::string &target_frame, const std::string source_frame)
{
    tf::StampedTransform transform;

    camera_lidar_tf_ok_ = false;

    try
    {
        // ros::Time(0)指定了时间为0，即获得最新有效的变换。
        // 改变获取当前时间的变换，即改为ros::Time::now(),不过now的话因为监听器有缓存区的原因。一般会出错
        // 参考：https://www.ncnynl.com/archives/201702/1313.html
        transform_listener.lookupTransform(target_frame, source_frame, ros::Time(0), transform);
        camera_lidar_tf_ok_ = true;
        ROS_INFO("FindTransform : camera-lidar-tf obtained");
    }
    catch (tf::TransformException ex)
    {
        ROS_INFO("FindTransform : %s", ex.what());
    }

    return transform;
}

pcl::PointXYZ EntryClass::transformPoint(const pcl::PointXYZ& in_point, const tf::StampedTransform& in_transform)
{
    tf::Vector3 tf_point(in_point.x, in_point.y, in_point.z);
    tf::Vector3 tf_point_transformed = in_transform * tf_point;
    return pcl::PointXYZ(tf_point_transformed.x(), tf_point_transformed.y(), tf_point_transformed.z());
}

// main loop
void EntryClass::mainLoop(const ros::TimerEvent &e)
{
    //ROS_INFO("main Loop at time %f", ros::Time::now().toSec());
}
