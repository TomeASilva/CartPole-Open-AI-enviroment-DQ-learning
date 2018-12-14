import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from network import create_features_tensor, create_variables, forward_propagation, target_network_variables_placeholders

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
env = gym.make('CartPole-v0')
observation = env.reset()


class ReplayBuffer:

    def __init__(self, size):
        self.size = size
        self.memory = []

    def save_sample(self, experience_sample):
        "experience_sample: list with of [s, a, s', r]"
        self.memory.append(experience_sample)

        if len(self.memory) > self.size:
            self.memory.pop(0)

    def draw_sample_batch(self, size):
        "Sample batch: array dim [batch size, len[s, a, s', r]"

        indices = np.random.randint(0, len(self.memory) - 1, size)
        memory = np.array(self.memory)
        sample_batch = memory[indices, :]

        return sample_batch


def create_state_action_pair_features(current_state, sample_batch):
    """If current state = True  --> return a numpy array of size [state features + action features, sample batch]
       If current state = False --> return a tuple of arrays of size [state features + action features, sample batch]
       for a1, and a2

       Current state being false means that we are evaluating the best action at state s' hence the 2 state action pairs

    """

    if current_state == True:
        s_a = sample_batch[:, 0:5].T

        return s_a
    if current_state == False:

        s_prime_a1 = np.c_[sample_batch[:, 5:-1],
                           np.zeros((sample_batch.shape[0], 1))].T
        s_prime_a2 = np.c_[sample_batch[:, 5:-1],
                           np.ones((sample_batch.shape[0], 1))].T

        return (s_prime_a1, s_prime_a2)


def value_function_approximator(input_features, _weights, _biases, graph):
    """Return:
        batch_value functions array [size_output_layer, batch size] """

    with graph.as_default():
        with tf.Session(graph=graph) as sess:

            batch_value_functions = sess.run(
                output_target, feed_dict={features_target: input_features, weights_target: _weights, biases_target: _biases})


        return batch_value_functions


def run_policy(epsilon, target_weights, target_biases, replaybuffer, graph, full_episode=True, render=False):
    done = False
    initial_state = env.reset()
    initial_state = initial_state.reshape(initial_state.shape[0], 1)
    counter = 0
  
    while not done:
        if render:
            env.render()

        random_number = np.random.uniform(0, 1)

        if random_number < (1-epsilon):

            s_a1_input = np.concatenate(
                (initial_state, np.zeros((1, 1))), axis=0)
            s_a2_input = np.concatenate(
                (initial_state, np.ones((1, 1))), axis=0)

            Q_s_a1 = value_function_approximator(
                s_a1_input, target_weights, target_biases, graph)
            Q_s_a2 = value_function_approximator(
                s_a2_input, target_weights, target_biases, graph)
            a_max = np.argmax(np.append(Q_s_a1, Q_s_a2, axis=0), axis=0)

            action = a_max

        else:
            action = env.action_space.sample()

        s_prime, reward, done, info = env.step(int(action))

        if done == False:
            reward = 0
        if done == True:
            reward = -1


        sample = [initial_state, [action], s_prime, [reward]]
        sample = list(itertools.chain(*sample))
        replaybuffer.save_sample(sample)
        counter += 1
        if not full_episode:
            break        

        initial_state = s_prime.reshape(s_prime.shape[0], 1)

    if full_episode: 
        print("number of steps: ", counter)
    else: pass


##### Create Replay Buffer #####
model_replay_buffer = ReplayBuffer(1000)


#####Create the 2 graphs: For the target network and learning networking
g_principal_net = tf.Graph()
g_target_net = tf.Graph()

# Create target network
with g_target_net.as_default():

    #####Create placeholders for your features and parameters in the target Network
    features_target = create_features_tensor(5, g_target_net)
    weights_target, biases_target = target_network_variables_placeholders(
        4, [5, 25, 12, 1], g_target_net)
    
    #####Create the operation for the forward propagation in the target network
    output_target = forward_propagation(features_target, weights_target, biases_target, [
        tf.nn.relu, tf.nn.relu, None], g_target_net)



# Create the learning network
with g_principal_net.as_default():

    ##### Create placeholders for your features
    features = create_features_tensor(5, g_principal_net)
    rewards = tf.placeholder(dtype=tf.float32, shape=None, name="Rewards")
    ##### Create variables that will undergo learning
    weights, biases = create_variables(4, [5, 25, 12, 1], g_principal_net)
    ##### Create constansts that will represent your hyperparameters you can use placeholder as well
    gamma = tf.constant(0.99, dtype=tf.float32, shape=None, name='gama')

    ##### Create forward prop operation
    output_pred = forward_propagation(features, weights, biases, [
        tf.nn.relu, tf.nn.relu, None], g_principal_net)

   
    ##### Create the placeholders needed for computing the bellman operator for Q(s, a) = r(s, a) + max a Q(s', a)
    s_prime_max_a_value_function = tf.placeholder(
        dtype=tf.float32, shape=None, name="s_a_value")
    
    ##### Create the op for computing the bellman operator
    bellman_values = rewards + gamma * s_prime_max_a_value_function

    cost_function = tf.losses.mean_squared_error(output_pred, bellman_values)

    optimizer = tf.train.AdamOptimizer().minimize(cost_function)

    init = tf.global_variables_initializer()
    ##### end of graph



##### Training loop
with tf.Session(graph=g_principal_net) as sess:

    sess.run(init)
    #### collect the parameters of the learning neural network to be passed later onto the target network
    target_network_weights = sess.run(weights)
    target_network_biases = sess.run(biases)

    #### Fill the buffer with some initial examples

    for i in range(4):
        run_policy(0.05, target_network_weights,
                   target_network_biases, model_replay_buffer, g_target_net)

    ####Book keeping
    costs = []

    ####Main DQN loop
    #### You can interrupt the main loop with ctrl c, the expception will be handled and you can see the behaviour of your
    #### policy, if you think the network is already performing well.
    try:
        for num_iterations in range(10000):

            ####At each loop collect more data and add it to your replaybuffer
            run_policy(0.05, target_network_weights,
                    target_network_biases, model_replay_buffer, g_target_net, False)

            #### Draw minibatch from your replay buffer
            my_sample_batch = model_replay_buffer.draw_sample_batch(32)

            #### Tranform a data sample of the type [s, a , r, s'] into a data sample of [s, a]
            s_a = create_state_action_pair_features(True, my_sample_batch)
            #############################
            #### Tranform a data sample of the type [s, a , r, s'] into [r]
            s_a_rewards = my_sample_batch[:, -1]

            # Evaluating best action at action at s' with target network
            ###Tranform a data sample of the type [s, a , r, s'] into a data sample of [s', a1] and [s', a2]
            s_prime_a1, s_prime_a2 = create_state_action_pair_features(
                False, my_sample_batch)
            ### Using the target network comput Q(s', a1) and Q(s', a2)
            Q_s_prime_a1 = value_function_approximator(
                s_prime_a1, target_network_weights, target_network_biases, g_target_net)  # add the weights and biases
            Q_s_prime_a2 = value_function_approximator(
                s_prime_a2, target_network_weights, target_network_biases, g_target_net)  # add the weights and biases
            
            ### Choose the max value between Q(s', a1) and Q(s', a2)
            Q_s_prime_a_max = np.max(
                np.append(Q_s_prime_a1, Q_s_prime_a2, axis=0), axis=0)

            ## run batch gradient descent  on the sampled minibatch
            #### feed into you learning network
                                        #### Features : (s,a)
                                        #### Rewards  : r(s,a)
                                        #### max Q(s',a')
                                        #### mse = (bellman operator - output from your learning network)

            _, cost_at_step = sess.run([optimizer, cost_function], feed_dict={features: s_a, rewards: s_a_rewards,
                                                                            s_prime_max_a_value_function: Q_s_prime_a_max})

            #### Store the cost function value
            if num_iterations % 100 == 0:
                print("Cost at step {} is : {}".format(
                    num_iterations, cost_at_step))

                costs.append(cost_at_step)
                run_policy(0.05,target_network_weights, target_network_biases, model_replay_buffer, g_target_net)
            ##### Update target network parameters asyncronously 
            if num_iterations % 40 == 0:
                # print("Update to the target network")
                target_network_weights = sess.run(weights)
                target_network_biases = sess.run(biases)
        
        #### Run the environment with the learnt policy 
        for i in range (10):
                run_policy(0, sess.run(weights), sess.run(biases), model_replay_buffer, g_target_net, True, True) 

    except KeyboardInterrupt:
            print("Test you learnt Policy: \n")
            for i in range (10):
                run_policy(0, sess.run(weights), sess.run(biases), model_replay_buffer, g_target_net, True, True)
    
   
            