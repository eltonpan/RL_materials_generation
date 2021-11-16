import numpy as np
from one_hot import onehot_target, element_set, comp_set, one_hot_to_element, element_to_one_hot, one_hot_to_comp, comp_to_one_hot
from CVAE import TempTimeGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# # Make single prediction
# op_arr = temp_gen.generate_samples(
#         onehot_target('Te3').reshape(1, 40, 115),
#         n_samples=100
#         )
# sinter_T = [] # List of generated sintering T
# for conds in op_arr:
#     conds = np.reshape(conds, (8,))
#     temp_time = scaler.inverse_transform(conds.reshape(1, -1)).flatten()
#     if temp_time[1] > 0 and temp_time[5] > 0:
#         sinter_T.append(round(temp_time[1], 1))
# sinter_T_pred = np.mean(sinter_T)
# # print(sinter_T_pred)

class MaterialEnvironment():
    """
    Defines the Markov decision process of generating an inorganic material.
    """
    def __init__(self, 
                element_set,
                comp_set,
                init_mat = None,
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
        self.state = None
        self.counter = 0

        self.path = []

    # @property
    # def state(self):
    #     return self.state

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

        return sinter_T_pred
    
    def step(self, action):
        """
        Takes a step forward according to the action.

        Args:
          action: List of np.array. 1st element is tuple of shape (1, num_elements), 2nd element is np.array of shape (1,10) 

        Returns:
          results: Namedtuple containing the following fields:
            * state: The molecule reached after taking the action.
            * reward: The reward get after taking the action.
            * terminated: Whether this episode is terminated.

        Raises:
          ValueError: If the number of steps taken exceeds the preset max_steps, or
            the action is not in the set of valid_actions.
        """
        # Record state and action
        state_action = (self.state, self.counter, action)
        self.path.append(state_action)

        # Take action
        element, comp = action
        element = one_hot_to_element([element])[0]
        comp = one_hot_to_comp([comp])[0]

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

        print(self.state)

        self.counter += 1
        result = (self.state, self.reward()) # result is a tuple of new state and reward from taking the action
    
env = MaterialEnvironment(element_set = element_set,
                          comp_set =  comp_set,)

print('step:',env.num_steps_taken)
print('state:',env.state)
print('')

env.step(
     [(1., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
      (0., 1., 0., 0., 0., 0., 0., 0., 0., 0.)
     ])

print('step:',env.num_steps_taken)
print('state:',env.state)
print('reward:',env.reward())
print('')

env.step(
     [(0., 0., 1., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
      (0., 0., 1., 0., 0., 0., 0., 0., 0., 0.)
     ])

print('step:',env.num_steps_taken)
print('state:',env.state)
print('reward:',env.reward())
print('')

    
# print('counter:', env.counter)
# print('state:',env.state)
# print(env.state)
# print(env.num_steps_taken)


