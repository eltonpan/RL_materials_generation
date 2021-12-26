import numpy as np
from one_hot import feature_calculators, featurize_target, onehot_target, element_set, comp_set, one_hot_to_element, element_to_one_hot, one_hot_to_comp, comp_to_one_hot, step_to_one_hot, one_hot_to_step
from CVAE import TempTimeGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pickle
import time

# Load scaler
compressed_inputs = np.load("data/ss_sg_inputs_impute_precs_onehot_targets_all_targets_1.npz") 
compressed_outputs = np.load("data/ss_sg_outputs_impute_precs_onehot_targets_all_targets_1.npz")
x_temp = []
for temp_time, _, route, _, _ in zip(compressed_inputs["x_temp_time_rxn"], compressed_inputs["c_material"], compressed_inputs["c_route"], compressed_inputs["c_precursors"], compressed_outputs["x_temp_time_rxn"]):
    if route == 1:
        x_temp.append(temp_time)
scaler = StandardScaler()
x_temp = np.array(x_temp)
x_temp = np.reshape(scaler.fit_transform(x_temp), (-1, 8, 1))

# Load prediction model
temp_gen = TempTimeGenerator()
temp_gen.build_nn_model()
temp_gen.load_models(model_variant="epochs_40", load_path="cvae_models/")

class MaterialEnvironment():
    """
    Defines the Markov decision process of generating an inorganic material.
    """
    def __init__(self, 
                element_set,
                comp_set,
                init_mat = '',
                max_steps = 5,
                ):
        '''
        Initializes the parameters for the MDP.
        
        Args:
        element_set: List. Set of elements (strings) for constructing an inorganic material
        init_mat: String. Initial material
        max_steps: Int. Max number of steps per episode
        '''
        self.element_set = element_set
        self.comp_set    = comp_set
        self.init_mat    = init_mat
        self.max_steps   = max_steps
        self.state = ''
        self.counter = 0

        self.path = []

    @property
    def num_steps_taken(self):
        return self.counter

    def get_path(self):
        return self.path

    def initialize(self):
        """Resets the MDP to its initial state."""
        self.state      = self.init_mat
        self.path       = []
        self.counter    = 0
        self.terminated = False

    def reward(self):
        if self.counter == self.max_steps:
            # Predict sintering temperature of material using CVAE
            op_arr = temp_gen.generate_samples(
            onehot_target(self.state).reshape(1, 40, 115),
            n_samples=100)
            sinter_T = [] # List of generated sintering T
            for conds in op_arr:
                conds = np.reshape(conds, (8,))
                temp_time = scaler.inverse_transform(conds.reshape(1, -1)).flatten()
                if temp_time[1] > 0 and temp_time[5] > 0:
                    sinter_T.append(round(temp_time[1], 1))
            sinter_T_pred = np.mean(sinter_T)
            reward = -sinter_T_pred
        else:
            reward = 0

        return reward
    
    def step(self, action):
        """
        Takes a step forward according to the action.

        Args:
          action: List of 2 tuples. 1st element is tuple of shape (1, num_elements), 2nd element is np.array of shape (1,10) 

        Returns:
            NA
        """

        # Take action
        element, comp = action
        element = one_hot_to_element([element])[0]
        comp = one_hot_to_comp([comp])[0]

        old_state = self.state

        if comp != '0': # Add element only if composition is non-zero
            add = True
        else:
            add = False

        if self.counter == 0: # If empty compound, initialize state
            if add: # Add element only if composition is non-zero
                self.state = element + comp
        else: # Else not initial state, so add element to exisiting state
            if add: # Add element only if composition is non-zero
                try: 
                    self.state += element + comp
                except: # Might still be an empty compound for non-initial states
                    self.state = element + comp
        self.counter += 1

        reward = self.reward()

        # Record state and action
        # s_a_r = ([onehot_target(old_state), step_to_one_hot([self.counter])[0]], action, reward) # One-hot states for storing into path, actions are already one-hot
        if old_state == '': # if empty string (starting state), don't featurize with Magpie, but with a vector of zeroes instead
            s_a_r = ([featurize_target(old_state), step_to_one_hot([self.counter])[0]], action, reward) # One-hot states for storing into path, actions are already one-hot
        else:
            s_a_r = ([featurize_target(old_state), step_to_one_hot([self.counter])[0]], action, reward) # One-hot states for storing into path, actions are already one-hot

        self.path.append(s_a_r) # append (state, action, reward) - [material, step], [element, composition], reward
    
env = MaterialEnvironment(element_set = element_set,
                          comp_set =  comp_set,)

# print('step:',env.num_steps_taken)
# print('state:',env.state)
# print('')

# env.step(
#      [(1., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
#       (0., 1., 0., 0., 0., 0., 0., 0., 0., 0.)
#      ])

# print('step:',env.num_steps_taken)
# print('state:',env.state)
# print('reward:',env.reward())
# print('')


# env.step(
#      [(0., 0., 1., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
#       (0., 0., 1., 0., 0., 0., 0., 0., 0., 0.)
#      ])

# print('step:',env.num_steps_taken)
# print('state:',env.state)
# print('reward:',env.reward())
# print('')

def generate_random_act():
    '''
    Generates random action.
    Returns: List. [tuple(element), tuple(composition)]
    '''
    # Sample random action
    element = random.sample(element_set, 1)
    element = tuple(element_to_one_hot(element)[0])
    comp = random.sample(comp_set, 1)
    comp = tuple(comp_to_one_hot(comp)[0])
    action = [element, comp]
    return action

def generate_random_ep(max_steps = 5):
    '''
    Generates an episode with random policy
    
    Args:
    max_steps: Int

    Returns: 
    env.path (an episode): List of SAR data in the form of [[material, step], [element, composition], reward]
    
    '''
    env.initialize()

    for i in range(max_steps):
        # Sample random action
        action = generate_random_act()

        # Take step with action
        env.step(action)
        print('step:', env.counter)
        print('state:',env.state)
        print('reward:',env.reward())
        # print(env.num_steps_taken)
        print('')
    return env.path


def extract_data_from_ep(episode, disc_factor =  0.9):
    """
    Extracts from each episode Q targets.
    
    Args:
    episode: List of SAR data in the form of [[material, step], [element, composition], reward]
    
    Returns:
    Q_data: List of [state, action, Q] datapoints
    """
    Q_data = []
    for step in reversed(range(env.max_steps)):
        state, action, reward = episode[step]

        if step == env.max_steps - 1: # If terminal state
            G = reward # Return = reward
        else:
            G = reward + disc_factor*G # Return = reward + discounted return of the PREVIOUS state

        data_point = [state, action, G]
        Q_data.append(data_point)
    return Q_data
 
# ========= FOR RANDOM POLICY ===========
if __name__ == "__main__":
    start = time.time()
    # Generate random episodes
    num_eps = 1000
    episodes = []
    for j in range(num_eps):
        episode = generate_random_ep()
        episodes.append(episode)

    Q_data_random = []
    # Extract Q_data from episodes
    for episode in episodes:
        Q_data = extract_data_from_ep(episode)
        Q_data_random.append(Q_data)
    end = time.time()
    print('time taken:', end - start)


# Save Q_data
if __name__ == "__main__":
    with open('./data/Q_data_random.pkl', 'wb') as f:
        pickle.dump(Q_data_random, f, pickle.HIGHEST_PROTOCOL)





    
        