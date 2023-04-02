此代码资源是我的博文:十五. 单线激光雷达和视觉信息融合,配套的ROS实践功能包.
使用前请确认并修改:
1.你的单线雷达和相机发布的topic消息;
2.single_ladar_and_camera_fusion.launch为此功能包启动launch;
3.start_lidar_camera.launch为启动我机器上单线激光雷达和相机的launch. 使用时请按你的实际环境进行配置,或者干脆放弃此文件, 用你自己熟悉的方式启动你机器的相机和Lidar节点;
4.start_lidar_camera.launch文件中我还发布了相机和激光雷达的位姿信息(联合标定信息)到ROS的TF. 代码中会用到此数据进行相机到激光雷达的三维坐标系变换. 使用时请确认你的环境也有这样的TF;
