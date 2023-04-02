#ifndef __ENTRY_CLASS_INCLUDE__
#define __ENTRY_CLASS_INCLUDE__

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
//#include <robot_pose_ekf/GetStatus.h>

// messages
#include "geometry_msgs/Twist.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/Image.h"
#include <sensor_msgs/CameraInfo.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/common/common.h>
#include <pcl/features/don.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/filters/random_sample.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

class EntryClass
{
private:
    ros::NodeHandle nh_private;
    ros::NodeHandle node_h;
    ros::NodeHandle node_image;
    ros::Timer timer_;
    ros::Subscriber laserScan_sub_;
    ros::Subscriber cameraImage_sub_;
    ros::Subscriber intrinsics_sub_;

    ros::Publisher fusion_cloud_pub_;
    ros::Publisher depth_laser_to_image_;

    cv::Mat current_image_frame;
    std::string image_frame_id;
    cv::Size image_frame_size;

    cv::Mat camera_intrinsic_value;  //相机内参
    cv::Mat distortion_coefficients; //畸变系数

    bool camera_lidar_tf_ok_;
    bool camera_info_ok_;
    bool usingObjs;

    float fx, fy, cx, cy;

    tf::StampedTransform camera_lidar_tf;
    tf::TransformListener transform_listener;

public:
    EntryClass();
    ~EntryClass();

private: 
    /// the mail loop that will be called periodically
    void mainLoop(const ros::TimerEvent &e);
    void intrinsicValueCallback(const sensor_msgs::CameraInfo::ConstPtr &intrinsic_value_msg);
    void cameraImageCallback(const sensor_msgs::Image::ConstPtr &image_msg);
    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr &laserScan_msg);
    void transLaserScanToPointCloud(const sensor_msgs::LaserScan::ConstPtr &laserScan_msg,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_msg);

    tf::StampedTransform findTransform(const std::string& target_frame, const std::string source_frame);
    pcl::PointXYZ transformPoint(const pcl::PointXYZ& in_point, const tf::StampedTransform& in_transform);
};
#endif