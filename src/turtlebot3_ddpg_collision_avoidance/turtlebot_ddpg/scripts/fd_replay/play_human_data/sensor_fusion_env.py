#! /usr/bin/env python


## version 2: 
## 1, navigate the robot using a constant heading angle
## 2, add the ddpg neural network
## 3, 24 laser data and just heading
## 4, added potential collisions



## Command:
## roslaunch turtlebot_iros turtlebot_world.launch world_file:='/home/hanlin/catkin_ws/src/turtlebot/turtlebot_iros/modified_world.world'
## source ~/iros_env/bin/activate
## rosrun turtlebot_iros ddpg_turtlebot.py

from cmath import nan
import rospy
import rospkg
import tf
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion

import cv2
import numpy as np
from cv_bridge import *
import ros_numpy

import threading
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState 

from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import *
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
#from kobuki_msgs.msg import BumperEvent
import time

import tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import gym
import numpy as np
import math
import random

from std_srvs.srv import Empty


bridge = CvBridge()

class InfoGetter(object):
    def __init__(self):
        #event that will block until the info is received
        self._event = threading.Event()
        #attribute for storing the rx'd message
        self._msg = None
        

    def __call__(self, msg):
        #Uses __call__ so the object itself acts as the callback
        #save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        self._event.wait(timeout)
        return self._msg




class GameState:

    def __init__(self):
        self.talker_node = rospy.init_node('talker', anonymous=True)
        self.pose_ig = InfoGetter()
        self.laser_ig = InfoGetter()
        self.image_ig = InfoGetter()
        self.depth_laser_ig = InfoGetter()
        self.sensor_fusion_ig = InfoGetter()
        self.collision_ig = InfoGetter()
        self.number_image= 0
        self.last_normalized_laser = [4 for i in range(24)]

        self.tra_num = 0
        

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.position = Point()
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.laser_info = rospy.Subscriber("/laserscan_filtered", LaserScan, self.laser_ig)
        self.image_info = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_ig)
        self.depth_laser_info = rospy.Subscriber("/camera/depth/scan",LaserScan,self.depth_laser_ig)
        self.sensor_fusion_info = rospy.Subscriber("/colored_point_cloud",PointCloud2,self.sensor_fusion_ig)
        # self.image_info = rospy.Subscriber("/camera/depth/image_raw", Image,DepthImageCallback)
        # self.bumper_info = rospy.Subscriber("/mobile_base/events/bumper", BumperEvent, self.processBump)


        # tf
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'
        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
        (self.position, self.rotation) = self.get_odom()

  
        self.rate = rospy.Rate(100) # 100hz

        # Create a Twist message and add linear x and angular z values
        self.move_cmd = Twist()
        self.move_cmd.linear.x = 0.6 #linear_x
        self.move_cmd.angular.z = 0.2 #angular_z

        self.sum_angular = 0
        self.sum_num = 0

        # crush default value
        self.crash_indicator = 0
        self.crash_num = 0

        # observation_space and action_space
        self.state_num = 3100 #32*32*3 +24 + 4                 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)
        # self.state_input1_space =  np.empty(1)
        # self.state_input2_space =  np.empty(1)

        self.laser_reward = 0

        self.nearest_distance = 10000000000000
        # set target position
        self.target_x = 10
        self.target_y = 10

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)




    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return (Point(),2)

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


    def print_odom(self):
        while True:
            (self.position, self.rotation) = self.get_odom()
            print("position is %s, %s, %s, ", self.position.x, self.position.y, self.position.z)
            print("rotation is %s, ", self.rotation)

    def target_point(self,x,y):
        mid = (x+y)/2
        span = abs(x-y)/2
        point = (np.random.random()-0.5)*2*span + mid
        return point

    def reset(self):

        targetx = [(np.random.random()-0.5)*18,8,(np.random.random()-0.5)*18,-8]
        targety = [8,(np.random.random()-0.5)*18,-8,(np.random.random()-0.5)*18]
        targetx = [(np.random.random()-0.5)*12,6,(np.random.random()-0.5)*12,-6]
        targety = [6,(np.random.random()-0.5)*12,-6,(np.random.random()-0.5)*12]
        aim = int(np.random.random()*4)
        self.target_x = targetx[aim]
        self.target_y = targety[aim]


        
        #左上
        # position_x = -2
        # position_y = 5.5
        #左下
        # position_x = -6
        # position_y = -7
        #dynamic
        position_x = -3
        position_y = -1


        
        
        # self.target_x = -8
        # self.target_y = 5
        # self.target_x = -4
        # self.target_y = -2
        self.target_x = -7
        self.target_y = -1


        self.crash_indicator = 0

        state_msg = ModelState()    
        state_msg.model_name = 'turtlebot3_waffle'   #turtlebot3_waffle_pi
        state_msg.pose.position.x = position_x
        state_msg.pose.position.y = position_y   #random_turtlebot_y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = -0.2
        state_msg.pose.orientation.w = 0

        state_target_msg = ModelState()    
        state_target_msg.model_name = 'unit_sphere' #'unit_sphere_0_0' #'unit_box_1' #'cube_20k_0'
        state_target_msg.pose.position.x = self.target_x
        state_target_msg.pose.position.y = self.target_y
        state_target_msg.pose.position.z = 0.0
        state_target_msg.pose.orientation.x = 0
        state_target_msg.pose.orientation.y = 0
        state_target_msg.pose.orientation.z = -0.2
        state_target_msg.pose.orientation.w = 0


        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")



        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
            resp_target = set_state( state_target_msg )

        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

        initial_state = np.ones(self.state_num)
        #initial_state[self.state_num-2] = 0
        initial_state[self.state_num-1] = 0
        initial_state[self.state_num-2] = 0
        initial_state[self.state_num-3] = 0
        initial_state[self.state_num-4] = 0

        # self.move_cmd.linear.x = 0
        # self.move_cmd.angular.z = 0
        # self.pub.publish(self.move_cmd)
        # time.sleep(1)
        # self.pub.publish(self.move_cmd)
        self.rate.sleep()


        return initial_state


    def turtlebot_is_crashed(self, laser_values, range_limit):
        self.laser_crashed_value = 0
        self.laser_crashed_reward = 0
        crash_flag = 0  #其中一个激光雷达信号检测到碰撞即可

        for i in range(len(laser_values)):
            if (laser_values[i] < 3*range_limit):
                self.laser_crashed_reward = -70
            if (laser_values[i] < 2*range_limit):
                self.laser_crashed_reward = -100
            if (laser_values[i] < range_limit):
                self.laser_crashed_value = 1
                self.laser_crashed_reward = -200
                self.reset()
                time.sleep(1)
                break
        return self.laser_crashed_reward


    def game_step(self, time_step=0.1, linear_x=0.8, angular_z=0.3):

        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        self.move_cmd.linear.x = linear_x*0.26
        self.move_cmd.angular.z = angular_z
        self.rate.sleep()


        (self.position, self.rotation) = self.get_odom()
        turtlebot_x_previous = self.position.x
        turtlebot_y_previous = self.position.y


        while (record_time_step < time_step) and (self.crash_indicator==0):
            self.pub.publish(self.move_cmd)
            self.rate.sleep()
            record_time = time.time()
            record_time_step = record_time - start_time

        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y




        angle_turtlebot = self.rotation

        # make input, angle between the turtlebot and the target
        angle_turtlebot_target = atan2(self.target_y - turtlebot_y, self.target_x- turtlebot_x)

        if angle_turtlebot < 0:
            angle_turtlebot = angle_turtlebot + 2*math.pi

        if angle_turtlebot_target < 0:
            angle_turtlebot_target = angle_turtlebot_target + 2*math.pi


        angle_diff = angle_turtlebot_target - angle_turtlebot
        if angle_diff < -math.pi:
            angle_diff = angle_diff + 2*math.pi
        if angle_diff > math.pi:
            angle_diff = angle_diff - 2*math.pi



        ############# subscibe the message of image #############
        cv_image = self.image_ig.get_msg()
        
        cv_image = bridge.imgmsg_to_cv2(cv_image,"rgb8")
        cv_image = np.array(cv_image)
        cv_image = cv_image.reshape(-1)
        # zero_cv_image = np.zeros((32*32*3))
        # zero_cv_image = cv_image.reshape(-1)


        ############# subscribe the message of pointcloud ##############
        point_cloud = self.sensor_fusion_ig.get_msg()
        gen = point_cloud2.read_points_list(point_cloud, skip_nans=True)
        if np.shape(gen)[0] >=512:
            k = random.sample(gen,512)
            container_fusion = list()
            for p in k:
                rgb = np.array([p.rgb])
                rgb.dtype = np.uint32
                r = np.asarray((rgb >> 16) & 0xFF, dtype=np.uint8)
                g = np.asarray((rgb >> 8) & 0xFF, dtype=np.uint8)
                b = np.asarray(rgb & 0xFF, dtype=np.uint8)
                # print(r[-1],g[-1],b[-1])
                # print(np.shape(np.array(p)))

                container_fusion.append([p[0],p[1],p[2],r[-1],g[-1],b[-1]])
            
        else:
            container_fusion = [[0]*6 for i in range(512)]
        container_fusion = np.array(container_fusion)
        container_fusion = container_fusion.reshape(-1)
        # print(np.shape(container_fusion))
        
        ############# subscibe the message of laser #############
        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        # print('turtlebot laser_msg.ranges is %s', laser_msg.ranges)
        # print('turtlebot laser data is %s', laser_values)

        # print('turtlebot laser data is %s', laser_values[0])

        #平均选取24个激光雷达距离信息
        forward_laser = laser_values[270:360] + laser_values[0:90]
        normalized_laser = [forward_laser[i*7+ int(i/2)]/3.5 for i in range(24)]
        zero_laser = np.zeros((1,24))
        zero_laser += 1
        #print('turtlebot normalized laser range is %s', normalized_laser)


        # prepare state

        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)

        state = np.append(container_fusion,normalized_laser)

        state = np.append(state, current_distance_turtlebot_target)
        #print("normalized_laser is %s", normalized_laser)
        #print("current_distance_turtlebot_target is %s", current_distance_turtlebot_target)

        state = np.append(state, angle_diff)
        #print("angle_diff is %s", angle_diff)

        state = np.append(state, linear_x)
        #print("linear_x is %s", linear_x)

        state = np.append(state, angular_z)
        #print("angular_z is %s", angular_z)

        # print("angle_turtlebot and angle_diff are %s %s", angle_turtlebot*180/math.pi, angle_diff*180/math.pi)
        # print("position x is %s position y is %s", turtlebot_x, turtlebot_y)
        # print("target position x is %s target position y is %s", self.target_x, self.target_y)
        # print("command angular is %s", angular_z*1.82)
        # print("command linear is %s", linear_x)
        #print("state is %s", state)

        state = state.reshape(1, self.state_num)



        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y


        np.save('trace_fusion/trace_fusion_x3/tracex_epoch_{}'.format(self.tra_num), turtlebot_x)
        np.save('trace_fusion/trace_fusion_y3/tracey_epoch_{}'.format(self.tra_num), turtlebot_y)
        self.tra_num += 1


        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target      ###########################################################
        distance_reward = distance_reward*(5/time_step)*10


        angle_reward = abs(angle_diff)
        if abs(angle_diff) > 1.57:
            angle_reward = 2*angle_reward
        angle_reward = -math.pow(1.5,angle_reward)
        


        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.2)
        self.laser_reward = sum(normalized_laser)-24
        self.laser_reward = 0
        self.collision_reward = self.laser_crashed_reward + self.laser_reward


        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        # if angular_z > 0.7:
        #     self.angular_punish_reward = -4
        # if angular_z < -0.7:
        #     self.angular_punish_reward = -4


        if linear_x < 0.7:
            self.linear_punish_reward = -4


        self.arrive_reward = 0
        if distance_turtlebot_target<0.3:
            self.arrive_reward = 300
            self.reset()
            time.sleep(1)


 
       
        reward  = distance_reward + angle_reward + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward
        # print(1111111111111111111111111111111111111111111)
        # print("laser_reward is %s", self.laser_reward)
        # print("laser_crashed_reward is %s", self.laser_crashed_reward)
        # print("arrive_reward is %s", self.arrive_reward)
        # print("distance reward is : %s", distance_reward)
        # print("angle reward is :%s",angle_reward)
        # print("reward is : %s", reward)
        # # print(1111111111111111111111111111111111111111111)

        return reward, state, self.laser_crashed_value




if __name__ == '__main__':
    try:
        sess = tensorflow.Session()
        K.set_session(sess)

        game_state = GameState()
        #game_state.reset()
        while True:
            linear_x = 0.8
            angular_z = 0.3
            game_state.print_odom()
            #game_state.game_step(0.1, 0.0, 0.0)
        
    except rospy.ROSInterruptException:
        pass


