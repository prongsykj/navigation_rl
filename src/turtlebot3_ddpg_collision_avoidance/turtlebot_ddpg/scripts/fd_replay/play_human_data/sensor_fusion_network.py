#! /usr/bin/env python

# this network can change both heading and velocity

from operator import concat
from cv2 import imwrite
import sensor_fusion_env
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense,LSTM, Dropout, Input, merge,Conv2D,MaxPooling2D,BatchNormalization
from keras.layers.merge import Add, Concatenate
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow.compat.v1 as tf
import random
from collections import deque
import os.path
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from priortized_replay_buffer import PrioritizedReplayBuffer
import cv2

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


picture_shape = (32,32,3)
picture_listnum = picture_shape[0]*picture_shape[1]*picture_shape[2]
trace_len = 16

point_cloud_shape = (512,6)
point_cloud_listnum = point_cloud_shape[0]*point_cloud_shape[1]
	

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .90
		self.tau   = .001


		self.buffer_size = 1000000
		self.batch_size = 64

		self.hyper_parameters_lambda3 = 0.2
		self.hyper_parameters_eps = 0.2
		self.hyper_parameters_eps_d = 0.4

		self.demo_size = 1000

		self.obs_dim = self.env.observation_space.shape[0]
		self.act_dim = self.env.action_space.shape[0]

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = RNNReplayBuffer(self.obs_dim,self.act_dim)
		#create actor and target actor
		self.actor_obs_history,self.actor_action_history,self.actor_obs_t, self.actor_model = self.create_actor_model()
		_,_,_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32,
			[None,self.act_dim]) # where we will feed de/dC (from critic)
		# the trainable of parameters in actor model(just like w and b)
		actor_model_weights = self.actor_model.trainable_weights
		# return the derivatives of ys with respect to xs(a list)
		# add grad_ys(-self.actor_critic_grad),weight,the dimension is the same as ys
		self.actor_grads = tf.gradients(self.actor_model.output[:,:],
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		# zip every weight and the gradent of weight
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_obs_history,self.critic_action_history,self.critic_obs_t,self.critic_action_t, self.critic_model = self.create_critic_model()
		_, _, _, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output[:,:],
			self.critic_action_t) # where we calcaulte de/dC for feeding above

		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		"""
        Input:
        obs_history - (batch_size, trace_len-1, obs_dim)       o_0, ..., o_(T-1)
        action_history - (batch_size, trace_len-1, action_dim) a_0, ..., a_(T-1)
        obs_t - (batch_size,obs_dim)                           o_T
        Output:
        action - (batch_size,action_dim)                       a*_T
        """
		obs_history = Input(shape=(trace_len-1,self.obs_dim))
		action_history = Input(shape=(trace_len-1,self.act_dim))
		obs_t = Input(shape=self.obs_dim)

        ##########################  the section of memory  #############################
		# 3 dimension to 2 dimension
		cloud_point1 = tf.reshape(obs_history[:,:,:],[-1,self.obs_dim])
        # 2 dimension to (batch*trace_len,512,6)
		cloud_point1 = tf.reshape(cloud_point1[:,:point_cloud_listnum],[-1,point_cloud_shape[0], 
                                                                point_cloud_shape[1]])
		cloud_point1 = tf.expand_dims(cloud_point1,-1)
		c11 = Conv2D(32,[1,point_cloud_shape[1]],activation='relu',strides=(1,1))(cloud_point1)
		c12 = Conv2D(64,[1,1],activation='tanh',strides=(1,1))(c11)
		m11 = MaxPooling2D(pool_size=(point_cloud_shape[0],1))(c12)
		m11 = tf.reshape(m11,[-1,obs_history.shape[1],m11.shape[1]*m11.shape[2]*m11.shape[3]])
		concat_obs = Concatenate()([m11,obs_history[:,:,picture_listnum:],action_history])
		lstm_out = LSTM(512)(concat_obs)

		#########################  the section of now state  #########################
        # 3 dimension to 2 dimension
		cloud_point2 = tf.reshape(obs_t[:,:],[-1,self.obs_dim])
        # 2 dimension to (batch,512,6)
		cloud_point2 = tf.reshape(cloud_point2[:,:picture_listnum],[-1,point_cloud_shape[0], 
                                                                point_cloud_shape[1]])
		cloud_point2 = tf.expand_dims(cloud_point2,-1)
		c21 = Conv2D(32,[1,point_cloud_shape[1]],activation='relu',strides=(1, 1))(cloud_point2)
		c22 = Conv2D(64,[1,1],activation='tanh',strides=(1,1))(c21)
		m12 = MaxPooling2D(pool_size=(point_cloud_shape[0],1))(c22)
		m12 = tf.reshape(m12,[-1,m12.shape[1]*m12.shape[2]*m12.shape[3]])
		concat_now = Concatenate()([m12,obs_t[:,picture_listnum:]])
		d21 = Dense(512, activation='relu')(concat_now)

        ########################  the section of concat all  #########################
		concat_all = Concatenate()([lstm_out,d21])
		# print(lstm_out.shape)
		# print(d21.shape)
		h1 = Dense(512, activation='relu')(concat_all)
		h2 = Dense(512, activation='relu')(h1)
		h3 = Dense(512, activation='relu')(h2)
		delta_theta = Dense(1, activation='tanh')(h3) 
		speed = Dense(1, activation='sigmoid')(h3) # sigmoid makes the output to be range [0, 1]
		output = Concatenate()([delta_theta, speed])
		model = Model([obs_history,action_history,obs_t], output)
		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return obs_history,action_history,obs_t, model

	def create_critic_model(self):
		"""
		Input:
		obs_history - (batch_size, trace_len-1, obs_dim)  o_0, ..., o_(T-1)
		action_history - (batch, trace_len-1, action_dim) a_0, ..., a_(T-1)
		obs_t - (batch_size,obs_dim)                      o_T
		action_t - (batch_size,obs_dim)                   a_T
		Output:
		value_sequence - (batch,1)                        v_T
		"""
		obs_history = Input(shape=(trace_len-1,self.obs_dim))
		action_history = Input(shape=(trace_len-1,self.act_dim))
		obs_t = Input(shape=self.obs_dim)
		action_t = Input(shape=self.act_dim)

		##########################  the section of memory  #############################
		# 3 dimension to 2 dimension
		cloud_point1 = tf.reshape(obs_history[:,:,:],[-1,self.obs_dim])
        # 2 dimension to (batch*trace_len,512,6)
		cloud_point1 = tf.reshape(cloud_point1[:,:point_cloud_listnum],[-1,point_cloud_shape[0], 
                                                                point_cloud_shape[1]])
		cloud_point1 = tf.expand_dims(cloud_point1,-1)
		c11 = Conv2D(32,[1,point_cloud_shape[1]],activation='relu',strides=(1,1))(cloud_point1)
		c12 = Conv2D(64,[1,1],activation='tanh',strides=(1,1))(c11)
		m11 = MaxPooling2D(pool_size=(point_cloud_shape[0],1))(c12)
		m11 = tf.reshape(m11,[-1,obs_history.shape[1],m11.shape[1]*m11.shape[2]*m11.shape[3]])
		concat_history = Concatenate()([m11,obs_history[:,:,picture_listnum:],action_history])
		lstm_out = LSTM(512)(concat_history)

		#########################  the section of now state  #########################
		# 3 dimension to 2 dimension
		cloud_point2 = tf.reshape(obs_t[:,:],[-1,self.obs_dim])
        # 2 dimension to (batch,512,6)
		cloud_point2 = tf.reshape(cloud_point2[:,:picture_listnum],[-1,point_cloud_shape[0], 
                                                                point_cloud_shape[1]])
		cloud_point2 = tf.expand_dims(cloud_point2,-1)
		c21 = Conv2D(32,[1,point_cloud_shape[1]],activation='relu',strides=(1, 1))(cloud_point2)
		c22 = Conv2D(64,[1,1],activation='tanh',strides=(1,1))(c21)
		m12 = MaxPooling2D(pool_size=(point_cloud_shape[0],1))(c22)
		m12 = tf.reshape(m12,[-1,m12.shape[1]*m12.shape[2]*m12.shape[3]])
		concat_now = Concatenate()([m12,obs_t[:,picture_listnum:],action_t])
		d21 = Dense(512, activation='relu')(concat_now)

		########################  the section of concat all  #########################
		concat_all = Concatenate()([lstm_out,d21])
		merged_h1 = Dense(512, activation='relu')(concat_all)
		merged_h2 = Dense(512, activation='relu')(merged_h1)
		output = Dense(1, activation='linear')(merged_h2)
		model  = Model([obs_history,action_history,obs_t,action_t], output)
		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return obs_history,action_history,obs_t,action_t, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #
	def remember(self,obs,action,reward,done):
		self.memory.put(obs,action,reward,done)

	def _train_critic_actor(self, samples):
		"""
		current state -- 0,1,...,T-2 + T-1(-2)
		new state     -- 1,2,...,T-1 + T(-1)
		"""
		obs_history, action_history, reward_history, done_history = samples

		print(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999)

		target_actions = self.target_actor_model.predict([obs_history[:, 1:-1, :],
                                        					action_history[:, 1:-1, :],
                                        					obs_history[:,-1, :]])
		future_rewards = self.target_critic_model.predict([obs_history[:, 1:-1, :],
                                                			action_history[:, 1:-1, :],
                                                			obs_history[:,-1, :],
                                                			target_actions])
		ONE = np.ones((self.batch_size,trace_len,1))
		rewards = reward_history[:,-1, :] + self.gamma* future_rewards * (1-done_history[:,-1, :])


		# train critic based on weights
		# print("_sample_weight is %s", _sample_weight)
		evaluation = self.critic_model.fit([obs_history[:, :-2, :], 
                                				action_history[:, :-2, :],
                                				obs_history[:,-2,:],
                                				action_history[:,-2,:]], 
											rewards, verbose=0)
		# print('\nhistory dict:', evaluation.history)


		# train actor based on weights
		predicted_actions = self.actor_model.predict([obs_history[:, :-2, :], 
                                    					action_history[:, :-2, :],
                                    					obs_history[:,-2,:]])


		grads = self.sess.run(self.critic_grads, feed_dict={
			self.critic_obs_history: obs_history[:, :-2, :],
			self.critic_action_history: action_history[:, :-2, :],
			self.critic_obs_t: obs_history[:,-2,:],
			self.critic_action_t: predicted_actions
		})[0]


		self.sess.run(self.optimize, feed_dict={
			self.actor_obs_history: obs_history[:, :-2, :],
			self.actor_action_history: action_history[:, :-2, :],
			self.actor_obs_t: obs_history[:,-2,:],
			self.actor_critic_grad:grads
		})

	def read_Q_values(self, obs_history,action_history,obs_t,action_t):
		critic_values = self.critic_model.predict([obs_history,action_history,obs_t,action_t])
		return critic_values

	def train(self):
		batch_size = self.batch_size
		if self.memory.size() < batch_size: #batch_size:
			return
		self._train_critic_actor(self.memory.sample(batch_size))


	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
		self.target_critic_model.set_weights(critic_target_weights)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, obs_history,action_history,obs_t):  # this function returns action, which is predicted by the model. parameter is epsilon
		eps = 0.9
		action = self.actor_model.predict([obs_history,action_history,obs_t])
		if np.random.random() < self.epsilon:
			action[0][0] = action[0][0] + (np.random.random()-0.5)*0.4
			action[0][1] = action[0][1] + np.random.random()*0.4
			return action
		else:
			action[0][0] = (np.random.random()-0.5)*2   # angular velocity
			action[0][1] = np.random.random()   # linear velocity
			return action

		

	# ========================================================================= #
	#                              save weights                            #
	# ========================================================================= #

	def save_weight(self, num_trials, trial_len):
		self.actor_model.save_weights('actormodel' + '-' +  str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)
		self.critic_model.save_weights('criticmodel' + '-' + str(num_trials) + '-' + str(trial_len) + '.h5', overwrite=True)#("criticmodel.h5", overwrite=True)

	def play(self, cur_state):
		return self.actor_model.predict(cur_state)



class History:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.obs_hist = np.zeros((0, obs_dim))

        self.action_dim = action_dim
        self.action_hist = np.zeros((0, action_dim))

        self.reward_hist = np.zeros((0, 1))
        self.done_hist = np.zeros((0, 1))

    @staticmethod
    def _insert(hist, new_value):
        # hist is in tf.float32, hence cast new_value to tf.float32
        new_value = np.expand_dims(new_value, 0)
        return np.concatenate((hist, new_value), 0)

    def insert_obs(self, obs):
        self.obs_hist = self._insert(self.obs_hist, obs)

    def insert_action(self, action):
        self.action_hist = self._insert(self.action_hist, action)

    def insert_reward(self, reward):
        self.reward_hist = self._insert(self.reward_hist, [reward])

    def insert_done(self, done):
        self.done_hist = self._insert(self.done_hist, [done])

    def get_action_history(self):
        return self.action_hist

    def get_obs_history(self):
        return self.obs_hist

    def get_reward_history(self):
        return self.reward_hist

    def get_done_history(self):
        return self.done_hist


class RNNReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=1000000, seed=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.seed = seed
        self.buffer = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(self.seed)

    def put(self, obs_history, action_history, reward_history,done_history):
        """
        obs_history: Tensor, sequence_length (T) * obs_dim
        action_history: Tensor, sequence_length (T) * action_dim
        reward_history: Tensor, sequence_length (T) * 1
        """
        self.buffer.append(
            np.concatenate((obs_history, action_history, reward_history,done_history), 1))

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, replacement=True):
        idx = self.rng.choice(self.size(),
                              size=batch_size,
                              replace=replacement)
        buffer_arr = np.array(self.buffer, dtype=object)
        # print(buffer_arr.shape)
        samples = buffer_arr[idx]

        
        # print(np.shape(samples))
        # print(idx)

        batch_obs_history = samples[:, :, :self.obs_dim]
        batch_action_history = samples[:, :,self.obs_dim:(self.obs_dim +self.action_dim)]
        batch_reward_history = samples[:, :, (self.obs_dim + self.action_dim):(self.obs_dim + self.action_dim +1)]
        batch_done_history = samples[:,:,(self.obs_dim + self.action_dim +1):(self.obs_dim + self.action_dim +2)]
        return batch_obs_history, batch_action_history, batch_reward_history,batch_done_history


def main():

	sess = tf.compat.v1.Session()
	K.set_session(sess)

	########################################################
	game_state= sensor_fusion_env.GameState()   # game_state has frame_step(action) function
	obs_dim = game_state.observation_space.shape[0]
	action_dim = game_state.action_space.shape[0]

	actor_critic = ActorCritic(game_state, sess)

	########################################################
	num_trials = 10000
	trial_len  = 1200

	current_state = game_state.reset()

	#actor_critic.read_human_data()
	
	step_reward = [0,0]
	step_Q = [0,0]
	step = 0
	


	actor_critic.actor_model.load_weights("actormodel-28-1200.h5")
	actor_critic.critic_model.load_weights("criticmodel-28-1200.h5")
	for i in range(num_trials):
		# history save the obs,action,reward of the whole num_steps  
		history = History(obs_dim, action_dim)
		# history_tracesave the obs,action,reward of the trace_len
		history_trace = History(obs_dim, action_dim)

		current_state = game_state.reset()
		##############################################################################################
		total_reward = 0
		
		for j in range(trial_len):
			print("trial:" + str(i),"trial_len:" + str(j))
			###########################################################################################
			#print('wanted value is %s:', game_state.observation_space.shape[0])
			current_state = current_state.reshape((1, obs_dim))

			if j < trace_len :
				action = np.array([[0,0]])
			else:
				obs_history = np.expand_dims(history.get_obs_history(), axis=0)
				action_history = np.expand_dims(history.get_action_history(),axis=0)
				action = actor_critic.act(obs_history[:,j-trace_len+1:j,:],
											action_history[:,j-trace_len+1:j,:],
											current_state)

				Q_values = actor_critic.read_Q_values(obs_history[:,j-trace_len+1:j, :], 
													action_history[:,j-trace_len+1:j, :],
													current_state,
													action)
				step_Q = np.append(step_Q,[step,Q_values[0][0]])
				print("Q_values is %s", Q_values[0][0])
				sio.savemat('step_Q.mat',{'data':step_Q},True,'5', False, False,'row')

			action = action.reshape((1, action_dim))
			print("action is speed: ",action[0][1]," angular: ",action[0][0])
			reward, new_state, done = game_state.game_step(0.1, action[0][1], action[0][0]) 
			total_reward = total_reward + reward
			###########################################################################################
			if j == (trial_len - 1):
				done = 1
				print("this is reward:", total_reward)

			history.insert_obs(current_state.reshape(-1))
			history.insert_action(action.reshape(-1))
			history.insert_reward(reward)
			history.insert_done(done)

			if j != 0 and j % trace_len == 0:
				history_trace.insert_obs(current_state.reshape(-1))
				history_trace.insert_action(action.reshape(-1))
				history_trace.insert_reward(reward)
				history_trace.insert_done(done)
				# history, o_0, ...,o_(T-1), o_T; a_0, ..., a_(T-1), a_T; r_0, ..., r_(T-1), r_T
				# history has a value when init
				actor_critic.remember(history_trace.get_obs_history()[:,:], history_trace.get_action_history()[:,:],
								history_trace.get_reward_history()[:,:],history_trace.get_done_history()[:,:])
				history_trace = History(obs_dim, action_dim)

			history_trace.insert_obs(current_state.reshape(-1))
			history_trace.insert_action(action.reshape(-1))
			history_trace.insert_reward(reward)
			history_trace.insert_done(done)

			step = step + 1
			#plot_reward(step,reward,ax,fig)
			step_reward = np.append(step_reward,[step,reward])
			sio.savemat('step_reward.mat',{'data':step_reward},True,'5', False, False,'row')
			print("step is %s", step)

			start_time = time.time()
			# if (j % 5 == 0):
			# 	game_state.game_step(0.001, action[0][1], 0)
			# 	actor_critic.train()
			# 	actor_critic.update_target() 
			# 	game_state.game_step(0.001, action[0][1], action[0][0])
			end_time = time.time()
			print("train time is %s", (end_time - start_time))
			
			new_state = new_state.reshape((1, obs_dim))
			current_state = new_state


			##########################################################################################
		np.save('reward_fusion/reward_epoch_{}'.format(i), total_reward)


		if (i % 2==0):
			actor_critic.save_weight(i, trial_len)

	


if __name__ == "__main__":
	main()
