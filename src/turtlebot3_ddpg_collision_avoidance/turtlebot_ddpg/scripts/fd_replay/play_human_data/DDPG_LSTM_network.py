#! /usr/bin/env python

from operator import concat

from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import gym
import cv2
from tensorflow.keras.layers import Dense, LSTM ,Conv2D,MaxPooling2D,Input

from tensorflow import keras
from tensorflow.keras.layers import concatenate 
from tensorflow.keras import Model
from collections import deque
#from rl_tf2.agents.utils import soft_update_weights, print_env_step_info
import ddpg_LSTM_camera

import tensorflow.compat.v1 as tf1
#tf.enable_eager_execution()
from tensorflow.keras.utils import plot_model


num_episodes = 500  
num_steps = 300
trace_len = 2
# TODO: initialize env
game_state = ddpg_LSTM_camera.GameState()

picture_shape = [64,64,3]
picture_listnum = picture_shape[0]*picture_shape[1]*picture_shape[2]

obs_dim = game_state.observation_space.shape[0]
action_dim = game_state.action_space.shape[0]
action_ub = np.array([1,1.5])
action_lb = np.array([-1,0])

buffer_size = 10000  # replay_buffer size
lstm_size = 256
dense_size = 256
seed = 1
batch_size = 50
critic_lr = 0.0001
actor_lr = 0.0001
target_network_update_rate = 0.005
discount = 0.99
noise_std = 0.1

@tf.function
def soft_update_weights(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def print_env_step_info(step, obs, action, reward):
    print(
        f'Step {step} - Observation: {obs}, Action: {action}, Reward: {reward}'
    )

class RNNCritic(Model):
    """
    Input:
    obs_history - (batch_size, trace_len-1, obs_dim)  o_0, ..., o_(T-1)
    action_history - (batch, trace_len-1, action_dim) a_0, ..., a_(T-1)
    obs_t - (batch_size,1,obs_dim)                    o_T
    action_t - (batch_size,1,obs_dim)                 a_T
    Output:
    value_sequence - (batch, trace_len, 1)            v_0, ...,  v_T
    """
    def __init__(self, lstm_size, dense_size, name='RNNCritic'):
        super(RNNCritic, self).__init__(name=name)

        self.cnn11 = Conv2D(16, (3, 3), activation='relu',name="HiddenCNN11" ,input_shape=(picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]))
        self.cnn12 = Conv2D(16, (3, 3), activation='relu',name="HiddenCNN12" ,input_shape=(picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]))
        self.maxpooling11 = MaxPooling2D((2, 2))
        self.maxpooling12 = MaxPooling2D((2, 2))
        self.cnn21 = Conv2D(32, (3, 3), activation='relu',name="HiddenCNN21")
        self.cnn22 = Conv2D(32, (3, 3), activation='relu',name="HiddenCNN22")
        self.maxpooling21 = MaxPooling2D((2, 2))
        self.maxpooling22 = MaxPooling2D((2, 2))
        self.dense11 = Dense(dense_size, activation='relu',name="HiddenDense11")
        self.dense12 = Dense(dense_size, activation='relu',name="HiddenDense12")
        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense2 = Dense(dense_size, activation='relu',name="HiddenDense2")
        self.dense3 = Dense(1, activation='linear',name="OutputDense")
        

    def call(self, obs_history, action_history,obs_t,action_t):

        # the section of memory
        picture1 = tf.reshape(obs_history[:,:,:],[obs_history.shape[0]*obs_history.shape[1],obs_dim])
        picture1 = tf.reshape(picture1[:,:picture_listnum],[obs_history.shape[0]*obs_history.shape[1],picture_shape[0], 
                                                                                                    picture_shape[1], 
                                                                                                    picture_shape[2]])
        cnn11_out = self.cnn11(picture1)
        maxpooling11_out = self.maxpooling11(cnn11_out)
        cnn21_out = self.cnn21(maxpooling11_out)
        maxpooling21_out = self.maxpooling21(cnn21_out)
        obs_feature1 = tf.reshape(maxpooling21_out,[obs_history.shape[0]*obs_history.shape[1],-1])
        obs_feature1 = tf.reshape(obs_feature1,[obs_history.shape[0],obs_history.shape[1],obs_feature1.shape[1]])
        dense11_in = concatenate([obs_feature1,obs_history[:,:,picture_listnum:obs_dim],action_history])
        dense11_out = self.dense11(dense11_in)
        lstm_out = self.lstm1(dense11_out)

        # the section of now state
        picture2 = tf.reshape(obs_t[:,:,:],[obs_t.shape[0]*obs_t.shape[1],obs_dim])
        picture2 = tf.reshape(picture2[:,:picture_listnum],[obs_t.shape[0]*obs_t.shape[1],picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]])
        cnn12_out = self.cnn12(picture2)
        maxpooling12_out = self.maxpooling12(cnn12_out)
        cnn22_out = self.cnn22(maxpooling12_out)
        maxpooling22_out = self.maxpooling22(cnn22_out)
        obs_feature2 = tf.reshape(maxpooling22_out,[obs_t.shape[0]*obs_t.shape[1],-1])
        obs_feature2 = tf.reshape(obs_feature2,[obs_t.shape[0],obs_t.shape[1],obs_feature2.shape[1]])
        dense12_in = concatenate([obs_feature2,obs_t[:,:,picture_listnum:obs_dim],action_t])
        dense12_out = self.dense12(dense12_in)

        # concat two sections
        concat_all = concatenate([lstm_out,dense12_out],axis=1)
        x = self.dense2(concat_all)
        value = self.dense3(x)
        
        return value
        # squeeze the second dimension so that the output shape will be (batch, )
        #  return tf.squeeze(x, axis=2)


class RNNActor(Model):
    """
    Input:
    obs_history - (batch_size, trace_len-1, obs_dim)       o_0, ..., o_(T-1)
    action_history - (batch_size, trace_len-1, action_dim) a_0, ..., a_(T-1)
    obs_t - (batch_size,1,obs_dim)                         o_T
    Output:
    action - (batch_size, trace_len, action_dim)           a*_0, ..., a*_T
    """
    def __init__(self,
                 action_dim,
                 lstm_size,
                 dense_size,
                 action_lb=None,
                 action_ub=None,
                 name='RNNActor'):
        super(RNNActor, self).__init__(name=name)
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.action_dim = action_dim
        self.cnn11 = Conv2D(16, (3, 3), activation='relu',name="HiddenCNN11" ,input_shape=(picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]))
        self.cnn12 = Conv2D(16, (3, 3), activation='relu',name="HiddenCNN12" ,input_shape=(picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]))
        self.maxpooling11 = MaxPooling2D((2, 2))
        self.maxpooling12 = MaxPooling2D((2, 2))
        self.cnn21 = Conv2D(32, (3, 3), activation='relu',name="HiddenCNN21")
        self.cnn22 = Conv2D(32, (3, 3), activation='relu',name="HiddenCNN22")
        self.maxpooling21 = MaxPooling2D((2, 2))
        self.maxpooling22 = MaxPooling2D((2, 2))
        self.dense11 = Dense(dense_size, activation='relu',name="HiddenDense11")
        self.dense12 = Dense(dense_size, activation='relu',name="HiddenDense12")
        self.lstm1 = LSTM(lstm_size, name="LSTM1", return_sequences=True)
        self.dense2 = Dense(dense_size, activation='relu',name="HiddenDense2")
        self.dense3 = Dense(1, activation='tanh',name="OutputDense")

    def call(self, obs_history, action_history,obs_t):
        # the section of memory
        picture1 = tf.reshape(obs_history[:,:,:],[obs_history.shape[0]*obs_history.shape[1],obs_dim])
        picture1 = tf.reshape(picture1[:,:picture_listnum],[obs_history.shape[0]*obs_history.shape[1],picture_shape[0], 
                                                                                                    picture_shape[1], 
                                                                                                    picture_shape[2]])
        cnn11_out = self.cnn11(picture1)
        maxpooling11_out = self.maxpooling11(cnn11_out)
        cnn21_out = self.cnn21(maxpooling11_out)
        maxpooling21_out = self.maxpooling21(cnn21_out)
        obs_feature1 = tf.reshape(maxpooling21_out,[obs_history.shape[0]*obs_history.shape[1],-1])
        obs_feature1 = tf.reshape(obs_feature1,[obs_history.shape[0],obs_history.shape[1],obs_feature1.shape[1]])
        dense11_in = concatenate([obs_feature1,obs_history[:,:,picture_listnum:obs_dim],action_history])
        dense11_out = self.dense11(dense11_in)
        lstm_out = self.lstm1(dense11_out)

        # the section of now state
        picture2 = tf.reshape(obs_t[:,:,:],[obs_t.shape[0]*obs_t.shape[1],obs_dim])
        picture2 = tf.reshape(picture2[:,:picture_listnum],[obs_t.shape[0]*obs_t.shape[1],picture_shape[0], 
                                                                                        picture_shape[1], 
                                                                                        picture_shape[2]])
        cnn12_out = self.cnn12(picture2)
        maxpooling12_out = self.maxpooling12(cnn12_out)
        cnn22_out = self.cnn22(maxpooling12_out)
        maxpooling22_out = self.maxpooling22(cnn22_out)
        obs_feature2 = tf.reshape(maxpooling22_out,[obs_t.shape[0]*obs_t.shape[1],-1])
        obs_feature2 = tf.reshape(obs_feature2,[obs_t.shape[0],obs_t.shape[1],obs_feature2.shape[1]])
        dense12_in = concatenate([obs_feature2,obs_t[:,:,picture_listnum:obs_dim]])
        dense12_out = self.dense12(dense12_in)

        # concat two sections
        concat_all = concatenate([lstm_out,dense12_out],axis=1)
        x = self.dense2(concat_all)
        action = self.dense3(x)

        if self.action_lb is not None and self.action_ub is not None:
            mid = (self.action_lb + self.action_ub) / 2
            span = (self.action_ub - self.action_lb) / 2
            action = span * action + mid
            
        return action


class History:
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.obs_hist = tf.zeros([0, obs_dim])

        self.action_dim = action_dim
        self.action_hist = tf.zeros([0, action_dim])

        self.reward_hist = tf.zeros([0, 1])
        self.done_hist = tf.zeros([0, 1])

    @staticmethod
    def _insert(hist, new_value):
        # hist is in tf.float32, hence cast new_value to tf.float32
        new_value = tf.cast(tf.expand_dims(new_value, 0), dtype=tf.float32)
        return tf.concat([hist, new_value], 0)

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
    def __init__(self, obs_dim, action_dim, capacity=10000, seed=None):
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
            tf.concat([obs_history, action_history, reward_history,done_history], 1))

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
        samples_tensor = tf.convert_to_tensor(samples, dtype=tf.float32)

        batch_obs_history = samples_tensor[:, :, :self.obs_dim]
        batch_action_history = samples_tensor[:, :,self.obs_dim:(self.obs_dim +self.action_dim)]
        batch_reward_history = samples_tensor[:, :, (self.obs_dim + self.action_dim):(self.obs_dim + self.action_dim +1)]
        batch_done_history = samples_tensor[:,:,(self.obs_dim + self.action_dim +1):(self.obs_dim + self.action_dim +2)]
        return batch_obs_history, batch_action_history, batch_reward_history,batch_done_history

    


def generate_action_noise(action_dim,
                          action_upper_bound,
                          action_lower_bound,
                          ):
    noise = np.zeros(action_dim)
    span = (action_upper_bound - action_lower_bound)/2
    mid = (action_upper_bound + action_lower_bound)/2
    noise = (np.random.random()-0.5)*2*span + mid
    
    return tf.convert_to_tensor(noise, dtype=tf.float32)

def act(noise,action):
    epsilon = 0.8
    if np.random.random() < epsilon:
        return action
    else:
        return noise



def main():

    # set train_continue is 1 , then load weights
    train_continue = 0


    replay_buffer = RNNReplayBuffer(obs_dim,
                                    action_dim,
                                    capacity=buffer_size,
                                    seed=seed)
    critic_loss_fun = tf.keras.losses.MeanSquaredError()
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

    critic = RNNCritic(lstm_size, dense_size)
    actor = RNNActor(action_dim,
                    lstm_size,
                    dense_size,
                    action_lb=action_lb,
                    action_ub=action_ub)
    target_critic = RNNCritic(lstm_size, dense_size)
    target_actor = RNNActor(action_dim,
                            lstm_size,
                            dense_size,
                            action_lb=action_lb,
                            action_ub=action_ub)

    # if train_continue == 1:
    #     # actor.build(tf.convert_to_tensor(np.zeros(32,32,3)))
    #     # critic.build(tf.convert_to_tensor(np.zeros(32,32,3)))

    #     actor.load_weights("LSTM_actormodel-16-1000.h5")
    #     critic.load_weights("LSTM_criticmodel-16-1000.h5")

    # Making the weights equal
    target_actor.set_weights(actor.get_weights())
    target_critic.set_weights(critic.get_weights())

    for episode in range(num_episodes):
        # history save the obs,action,reward of the whole num_steps  
        history = History(obs_dim, action_dim)
        # history_tracesave the obs,action,reward of the trace_len
        history_trace = History(obs_dim, action_dim)
        curr_obs = game_state.reset()

        # insert curr_obs and dummy action and reward, so the obs, action, reward
        # history have the same sequence length.

        # history.insert_obs(curr_obs)
        # history.insert_action(tf.zeros([action_dim]))
        # history.insert_reward(0)

        # history_trace.insert_obs(curr_obs)
        # history_trace.insert_action(tf.zeros([action_dim]))
        # history_trace.insert_reward(0)

        sum_reward = 0
        for t in range(num_steps):
            # TODO: noise generation
            noise = generate_action_noise(action_dim,
                                        action_ub,
                                        action_lb,
                                        )
            if t == 0 or t == 1:
                action = noise
            else:
                # add dummy batch dimension.
                obs_history = tf.expand_dims(history.get_obs_history(), axis=0)
                action_history = tf.expand_dims(history.get_action_history(),
                                                axis=0)

                # o1, ..., oT; a0, ..., a_(T-1)
                if t > trace_len:
                    action_seq = actor(obs_history[:, t-trace_len-1:t-1, :],
                                    action_history[:, t-trace_len-1:t-1, :],
                                    tf.expand_dims(obs_history[:,-1,:],axis=1))
                else:
                    action_seq = actor(obs_history[:, :-1, :],
                                    action_history[:, :-1, :],
                                    tf.expand_dims(obs_history[:,-1,:],axis=1))

                # Only the last action is needed.
                action = action_seq[:, -1, :] 

                # need to squeeze the dummy batch dimension
                action = tf.squeeze(action, axis=0)

                Qpredicts = critic(obs_history[:, :-1, :], 
                                action_history[:, :-1, :],
                                tf.expand_dims(obs_history[:,-1,:],axis=1),
                                tf.expand_dims(action_seq[:,-1,:],axis=1))
                print("Qpredict is ",Qpredicts[:,-1,:].numpy())

            




            action = act(noise,action)
                
            #tensor transform to numpy 
            action_np = action.numpy()
            
            #print(action_np)
            action_np = action_np.reshape((1, game_state.action_space.shape[0]))
            action_np[0][0] += (np.random.random()-0.5)*0.2
            action_np[0][1] += np.random.random()*0.2
            #print("action is speed: %s, angular: %s", action_np[0][1], action_np[0][0])


            reward,curr_obs,done = game_state.game_step(0.1,action_np[0][1],action_np[0][0])
            sum_reward += reward
            #  print_env_step_info(t, curr_obs, action, reward)

            curr_obs = curr_obs.reshape(-1)  #reshape a matrix to a list   [1*28] to [28]
            # print(222222222222222222222222222222222222222222222222)
            # print(curr_obs.shape)

            # picture = curr_obs[:3072].reshape(32,32,3)
            # cv2.imwrite("111.jpg",picture)

            history.insert_obs(curr_obs)
            history.insert_action(action)
            history.insert_reward(reward)
            history.insert_done(done)

            if t != 0 and t % (trace_len-1) == 0:
                history_trace.insert_obs(curr_obs)
                history_trace.insert_action(action)
                history_trace.insert_reward(reward)
                history_trace.insert_done(done)
                # history, o_0, ...,o_(T-1), o_T; a_0, ..., a_(T-1), a_T; r_0, ..., r_(T-1), r_T
                # history has a value when init
                replay_buffer.put(history_trace.get_obs_history()[:,:], history_trace.get_action_history()[:,:],
                                history_trace.get_reward_history()[:,:],history_trace.get_done_history()[:,:])
                history_trace = History(obs_dim, action_dim)

            history_trace.insert_obs(curr_obs)
            history_trace.insert_action(action)
            history_trace.insert_reward(reward)
            history_trace.insert_done(done)

   
            soft_update_weights(target_critic.variables, critic.variables,
                        target_network_update_rate)
            soft_update_weights(target_actor.variables, actor.variables,
                        target_network_update_rate)

            # actor.summary()
            # critic.summary()
            # target_actor.summary()
            # target_critic.summary()

            # plot_model(actor, to_file='actor_model.png', show_shapes=True)
            # plot_model(critic, to_file='critic_model.png', show_shapes=True)
            # plot_model(target_actor, to_file='target_actor_model.png', show_shapes=True)
            # plot_model(target_critic, to_file='target_critic_model.png', show_shapes=True)
            if t == 100:
                actor.save_weights('LSTM_actormodel' + '-' +  str(episode) + '-' + str(num_steps) + '.h5', overwrite=True)
                critic.save_weights('LSTM_criticmodel' + '-' +  str(episode) + '-' + str(num_steps) + '.h5', overwrite=True)
                
            #print("episode is ",episode,"  ","step is ",t)


        
        obs_history, action_history, reward_history, done_history= replay_buffer.sample(batch_size)

        with tf.GradientTape() as tape:
            # obs_history: 1, ..., T; action_history: 0, 1, ..., T-1
            # target_actions: 1, ..., T;
            target_actions = target_actor(obs_history[:, :-1, :],
                                        action_history[:, :-1, :],
                                        tf.expand_dims(obs_history[:,-1,:],axis=1))

            # y1*, ..., yT*; o1, ..., oT; a*_1, ..., a*_T
            target_critic_output = target_critic(obs_history[:, :-1, :],
                                                target_actions[:, :-1, :],
                                                tf.expand_dims(obs_history[:,-1,:],axis=1),
                                                tf.expand_dims(target_actions[:,-1,:],axis=1))

            # reward_history 0, 1, ..., T - 1, target_critic_output 1, ... T,
            # target_values 0, ..., T - 1
            ONE = tf.ones([batch_size,trace_len,1], dtype=tf.float32)
            target_values = reward_history + discount *(ONE-done_history)* target_critic_output

            # yhat 0, ..., T - 1, obs_history 0, ... T - 1, aciton_history 0, ... T - 1
            Qpredicts = critic(obs_history[:, :-1, :], 
                                action_history[:, :-1, :],
                                tf.expand_dims(obs_history[:,-1,:],axis=1),
                                tf.expand_dims(action_history[:,-1,:],axis=1))

            critic_loss = critic_loss_fun(tf.stop_gradient(target_values),
                                        Qpredicts)

        critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_gradients, critic.trainable_variables))

        with tf.GradientTape() as tape:
            # obs_history: 1, ..., T, action_history: 0, 1, ..., T-1,
            # actor_actions: 1, ..., T
            actor_actions = actor(obs_history[:, :-1, :], 
                                    action_history[:, :-1, :],
                                    tf.expand_dims(obs_history[:,-1,:],axis=1)
                                    )
            actor_loss = -tf.math.reduce_mean(
                critic(obs_history[:, :-1, :], 
                        actor_actions[:, :-1, :],
                        tf.expand_dims(obs_history[:,-1,:],axis=1),
                        tf.expand_dims(actor_actions[:,-1,:],axis=1)))

        actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_gradients, actor.trainable_variables))

        if train_continue == 1:
                actor.load_weights("LSTM_actormodel-33-300.h5")
                critic.load_weights("LSTM_criticmodel-33-300.h5")
                train_continue = 0

        print(f'Episode {episode}: reward sum = {sum_reward}')

        

        

if __name__ == "__main__":
    main()
