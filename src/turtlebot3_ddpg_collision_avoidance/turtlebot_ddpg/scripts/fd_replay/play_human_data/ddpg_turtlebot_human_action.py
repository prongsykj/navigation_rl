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

import rospy
import rospkg
import tf
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, Quaternion
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion

import threading
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState 

from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
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
        self.collision_ig = InfoGetter()
        self.action_ig = InfoGetter()
        

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.position = Point()
        self.move_cmd = Twist()

        self.pose_info = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_ig)
        self.laser_info = rospy.Subscriber("/laserscan_filtered", LaserScan, self.laser_ig)
        self.action_info = rospy.Subscriber("/cmd_vel", Twist, self.action_ig)


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

        # crush default value
        self.crash_indicator = 0
        self.crash_num = 0

        # observation_space and action_space
        self.state_num = 28 #685                 # when you change this value, remember to change the reset default function as well
        self.action_num = 2
        self.observation_space = np.empty(self.state_num)
        self.action_space = np.empty(self.action_num)
        # self.state_input1_space =  np.empty(1)
        # self.state_input2_space =  np.empty(1)

        self.laser_reward = 0
        # set target position
        self.target_x = 10
        self.target_y = 10

        # set turtlebot index in gazebo world
        self.model_index = 10 #25

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        self.return_action = np.empty(self.action_num)
        self.return_action = self.return_action.reshape((1, self.action_space.shape[0]))

    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


    def print_odom(self):
        while True:
            (self.position, self.rotation) = self.get_odom()
            print("position is %s, %s, %s, ", self.position.x, self.position.y, self.position.z)
            print("rotation is %s, ", self.rotation)


    def reset(self):
        index_list = [-1, 1]
        index_x = random.choice(index_list)
        index_y = random.choice(index_list)
        index_turtlebot_y = random.choice(index_list)
        # self.target_x = (np.random.random()-0.5)*5 + 12*index_x
        # self.target_y = (np.random.random()-0.5)*5 + 12*index_y
        targetx = [(np.random.random()-0.5)*20,9,(np.random.random()-0.5)*20,-9]
        targety = [9,(np.random.random()-0.5)*20,-9,(np.random.random()-0.5)*20]
        aim = int(np.random.random()*4)

        self.target_x = targetx[aim]
        self.target_y = targety[aim]

        self.target_x = 1
        self.target_y = 2


        random_turtlebot_y = (np.random.random())*4 + index_turtlebot_y


        self.crash_indicator = 0

        state_msg = ModelState()    
        state_msg.model_name = 'turtlebot3_waffle'
        state_msg.pose.position.x = 2
        state_msg.pose.position.y = 2 #random_turtlebot_y
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

        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0
        self.pub.publish(self.move_cmd)
        time.sleep(1)
        self.pub.publish(self.move_cmd)
        self.rate.sleep()


        return initial_state


    def turtlebot_is_crashed(self, laser_values, range_limit):
        self.laser_crashed_value = 0
        self.laser_crashed_reward = 0
        crash_flag = 0  #其中一个激光雷达信号检测到碰撞即可

        for i in range(len(laser_values)):
            if (laser_values[i] < 1.6*range_limit):
                self.laser_crashed_reward = -100
            if (laser_values[i] < range_limit):
                if (crash_flag == 0):
                    self.crash_num += 1
                self.laser_crashed_value = 1
                self.laser_crashed_reward = -300
                crash_flag = 1
                #累积碰撞10次再停止
                if(self.crash_num >= 1):
                    self.crash_num = 0
                    #self.nearest_distance = 10000000000000
                    self.reset()
                    time.sleep(1)
                    break
        return self.laser_crashed_reward



    def read_action(self):
        action_value_total = self.action_ig.get_msg()
        print ("action_value linear x is %s", action_value_total.linear.x)
        print ("action_value angular z is %s", action_value_total.angular.z)
        self.return_action[0][0] = action_value_total.angular.z
        self.return_action[0][1] = action_value_total.linear.x
        return self.return_action


    def read_state(self):
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        angle_turtlebot = self.rotation


        action_value_total = self.action_ig.get_msg()
        linear_x = action_value_total.linear.x
        angular_z = action_value_total.angular.z



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


        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges

        #print(np.size(laser_values))

        #normalized_laser = [laser_values[i*7+ int(i/2)]/3.5 for i in range(24)]

        forward_laser = laser_values[270:360] + laser_values[0:90]
        normalized_laser = [forward_laser[i*7+ int(i/2)]/3.5 for i in range(24)]
        # print(normalized_laser)
        # laser_angles = laser_msg.angle_max
        # print(123456789)
        # print(laser_angles)
        # laser_angles = np.rad2deg(laser_angles)
        # print(laser_angles[270:360])
        # print(laser_angles[0:90])

        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x)
        state = np.append(state, angular_z)

        state = state.reshape(1, self.state_num)

        return state



    def read_game_step(self, time_step=0.1, linear_x=0.8, angular_z=0.3):
        start_time = time.time()
        record_time = start_time
        record_time_step = 0
        # self.move_cmd.linear.x = linear_x
        # self.move_cmd.angular.z = angular_z
        self.rate.sleep()


        (self.position, self.rotation) = self.get_odom()
        turtlebot_x_previous = self.position.x
        turtlebot_y_previous = self.position.y


        while (record_time_step < time_step) and (self.crash_indicator==0):
            #self.pub.publish(self.move_cmd)
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



        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        normalized_laser = [laser_values[i*7+ int(i/2)]/3.5 for i in range(24)]
        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x)
        state = np.append(state, angular_z)
        # print("angle_turtlebot and angle_diff are %s %s", angle_turtlebot*180/math.pi, angle_diff*180/math.pi)
        # print("position x is %s position y is %s", turtlebot_x, turtlebot_y)
        # print("target position x is %s target position y is %s", self.target_x, self.target_y)
        # print("command angular is %s", angular_z*1.82)
        # print("command linear is %s", linear_x*0.26)
        #print("state is %s", state)

        state = state.reshape(1, self.state_num)


        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target

        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        self.laser_reward = sum(normalized_laser)-24
        self.collision_reward = self.laser_crashed_reward + self.laser_reward


        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if angular_z > 0.8:
            self.angular_punish_reward = -3
        if angular_z < -0.8:
            self.angular_punish_reward = -3

        if linear_x < 0.2:
            self.linear_punish_reward = -4


        self.arrive_reward = 0
        if distance_turtlebot_target<0.7:
            self.arrive_reward = 400
            self.reset()
            time.sleep(1)


 
        reward  = distance_reward*(5/time_step)*10 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward-0.01
        # print("laser_reward is %s", self.laser_reward)
        # print("laser_crashed_reward is %s", self.laser_crashed_reward)
        # print("arrive_reward is %s", self.arrive_reward)
        # print("distance reward is : %s", distance_reward*(5/time_step)*1.2*7)


        return reward, state, self.laser_crashed_value      


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



        # prepare the normalized laser value and check if it is crash
        laser_msg = self.laser_ig.get_msg()
        laser_values = laser_msg.ranges
        laser_angles = laser_msg.angle_increment
        laser_angles = np.rad2deg(laser_angles)
        print(laser_angles[270:360])
        print(laser_angles[0:90])
        #print('turtlebot laser_msg.ranges is %s', laser_msg.ranges)
        #print('turtlebot laser data is %s', laser_values)

        forward_laser = laser_values[270:360] + laser_values[0:90]
        
        normalized_laser = [forward_laser[i*7+ int(i/2)]/3.5 for i in range(24)]
        # print('turtlebot normalized laser range is %s', normalized_laser)


        # prepare state
        #state = np.append(normalized_laser, angle_diff)
        #state = np.append(normalized_laser,self.target_x- turtlebot_x)
        #state = np.append(state, self.target_y - turtlebot_y)
        current_distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)

        state = np.append(normalized_laser, current_distance_turtlebot_target)
        state = np.append(state, angle_diff)
        state = np.append(state, linear_x*0.26)
        state = np.append(state, angular_z)
        # print("angle_turtlebot and angle_diff are %s %s", angle_turtlebot*180/math.pi, angle_diff*180/math.pi)
        # print("position x is %s position y is %s", turtlebot_x, turtlebot_y)
        # print("target position x is %s target position y is %s", self.target_x, self.target_y)
        # print("command angular is %s", angular_z*1.82)
        # print("command linear is %s", linear_x*0.26)
        #print("state is %s", state)

        state = state.reshape(1, self.state_num)


        # make distance reward
        (self.position, self.rotation) = self.get_odom()
        turtlebot_x = self.position.x
        turtlebot_y = self.position.y
        distance_turtlebot_target_previous = math.sqrt((self.target_x - turtlebot_x_previous)**2 + (self.target_y - turtlebot_y_previous)**2)
        distance_turtlebot_target = math.sqrt((self.target_x - turtlebot_x)**2 + (self.target_y - turtlebot_y)**2)
        distance_reward = distance_turtlebot_target_previous - distance_turtlebot_target

        self.laser_crashed_reward = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
        self.laser_reward = sum(normalized_laser)-24
        self.collision_reward = self.laser_crashed_reward + self.laser_reward


        self.angular_punish_reward = 0
        self.linear_punish_reward = 0

        if angular_z > 0.8:
            self.angular_punish_reward = -3
        if angular_z < -0.8:
            self.angular_punish_reward = -3

        if linear_x < 0.3:
            self.linear_punish_reward = -4


        self.arrive_reward = 0
        if distance_turtlebot_target<0.5:
            self.arrive_reward = 400
            self.reset()
            time.sleep(1)


 

        reward  = distance_reward*(5/time_step)*10 + self.arrive_reward + self.collision_reward + self.angular_punish_reward + self.linear_punish_reward
        # print("laser_reward is %s", self.laser_reward)
        # print("laser_crashed_reward is %s", self.laser_crashed_reward)
        # print("arrive_reward is %s", self.arrive_reward)
        # print("distance reward is : %s", distance_reward*(5/time_step)*1.2*7)


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


