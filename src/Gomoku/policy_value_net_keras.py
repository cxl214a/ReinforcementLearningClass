# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet with Keras
Tested under Keras 2.0.5 with tensorflow-gpu 1.2.1 as backend

@author: Mingxu Zhang
""" 

import tensorflow as tf
import keras.backend as K
import numpy as np
import pickle


class PolicyValueNet():
    """policy-value network """
    def __init__(self, width, height, model_file=None):
        self.width = width
        self.height = height 
        self.create_policy_value_net()   
        self._loss_train_op()
        self.regularizer = tf.keras.regularizers.l2(l=0.0001)
        
        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)
        
    def create_policy_value_net(self):
        """create the policy value network """   
        net_input = base_net = tf.keras.layers.Input(shape=(5, self.width, self.height))

        # conv layers
        base_net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", 
                                         data_format="channels_first", activation="relu", 
                                         kernel_regularizer=self.regularizer)(base_net)
        base_net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", 
                                         data_format="channels_first", activation="relu", 
                                         kernel_regularizer=self.regularizer)(base_net)
        base_net = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", 
                                         data_format="channels_first", activation="relu", 
                                         kernel_regularizer=self.regularizer)(base_net)
        # action policy layers
        policy = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), 
                                        data_format="channels_first", activation="relu",
                                        kernel_regularizer=self.regularizer)(base_net)
        policy = tf.keras.layers.Flatten()(policy)
        self.policy = tf.keras.layers.Dense(self.width*self.height, activation="softmax", 
                                            kernel_regularizer=self.regularizer)(policy)
        # state value layers
        value = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), 
                                       data_format="channels_first", activation="relu", 
                                       kernel_regularizer=self.regularizer)(base_net)
        value = tf.keras.layers.Flatten()(value)
        value = tf.keras.layers.Dense(64, kernel_regularizer=self.regularizer)(value)
        self.value = tf.keras.layers.Dense(1, activation="tanh",
                                           kernel_regularizer=self.regularizer)(value)

        self.model = tf.keras.models.Model(net_input, [self.policy, self.value])
        
        def policy_value(state):
            return self.model.predict(state)
        
        self.policy_value = policy_value
        
    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 5, self.width, self.height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """
        self.model.compile(optimizer=tf.train.AdamOptimizer, loss=['categorical_crossentropy', 'mean_squared_error'])

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy
        
        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()        
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
