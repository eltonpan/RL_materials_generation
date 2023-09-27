from asyncio import tasks
import numpy as np
from one_hot import feature_calculators, featurize_target, onehot_target, element_set, comp_set, one_hot_to_element, element_to_one_hot, one_hot_to_comp, comp_to_one_hot, step_to_one_hot, one_hot_to_step
# from CVAE import TempTimeGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import pickle
import time
from matminer.featurizers.base import MultipleFeaturizer
import matminer.featurizers.composition as cf
from pymatgen.core.composition import Composition
from sklearn.ensemble import RandomForestRegressor
import joblib
from constraints.checkers import check_electronegativity
from roost_models.roost_model import predict_formation_energy, predict_bulk_mod, predict_shear_mod, predict_band_gap
from configs import configs

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

# # Load CVAE prediction model
# temp_gen = TempTimeGenerator()
# temp_gen.build_nn_model()
# temp_gen.load_models(model_variant="epochs_40", load_path="cvae_models/")

# Load RF prediction model
rf_regr_sinter  = RandomForestRegressor()
rf_regr_calcine = RandomForestRegressor()
# rf_regr_sinter = joblib.load("rf_models/rf_sinter_predict_no_imputation_no_precursors.joblib") # original RF model
rf_regr_sinter = joblib.load("rf_models/optimal_sinter_RF.joblib") # final RF sintering model
rf_regr_calcine = joblib.load("rf_models/optimal_calcine_RF.joblib") # final RF sintering model

# Featurization for RF model
feature_calculators = MultipleFeaturizer([
    cf.element.Stoichiometry(),
    cf.composite.ElementProperty.from_preset("magpie"),
    cf.orbital.ValenceOrbital(props=["avg"]),
    cf.ion.IonProperty(fast=True)
])

def predict_sinter(chemical):
    '''
    Predicts the sintering temperature of a material

    Args:
    chemical: Str.
    
    Returns
    sinter_T: float. Predicted sintering temperature
    '''
    try:
        chemical = Composition(chemical)
        features = feature_calculators.featurize(chemical)
        features = np.array(features).reshape(1, -1)
        # print(features)
        sinter_T = rf_regr_sinter.predict(features)[0]
    except IndexError: # Ad-hoc fix for featurization problem (chemical = Composition(self.state))
        sinter_T = 1000.0

    return sinter_T

def predict_calcine(chemical):
    '''
    Predicts the calcination temperature of a material

    Args:
    chemical: Str.
    
    Returns
    calcine_T: float. Predicted sintering temperature
    '''
    try:
        chemical = Composition(chemical)
        features = feature_calculators.featurize(chemical)
        features = np.array(features).reshape(1, -1)
        # print(features)
        calcine_T = rf_regr_calcine.predict(features)[0]
    except IndexError: # Ad-hoc fix for featurization problem (chemical = Composition(self.state))
        calcine_T = 1000.0

    return calcine_T

class ConstrainedMaterialEnvironment():
    """
    Defines the Markov decision process of generating an inorganic material.

    """
    def __init__(self, 
                element_set,
                comp_set,
                tasks,
                init_mat = '',
                max_steps = 5,
                ):
        '''
        Initializes the parameters for the MDP.
        
        Args:
        element_set: List. Set of elements (strings) for constructing an inorganic material
        comp_set: List. Set of compositions (strings) for constructing an inorganic material
        tasks: List. Rewards to optimize with respect to 
            e.g. 'sinter', 'form_e', 'bulk_mod', 'shear_mod', 'band_gap'
        init_mat: String. Initial material
        max_steps: Int. Max number of steps per episode
        '''
        self.element_set = element_set
        self.comp_set    = comp_set
        self.init_mat    = init_mat
        self.max_steps   = max_steps
        self.state = ''
        self.counter = 0
        self.tasks = tasks

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
            
            reward = 0 # Initialize reward to 0 and accumulate reward according to tasks

            # 1A) Sintering temperature of material using RF
            if 'sinter' in self.tasks:
                sinter_T = predict_sinter(self.state)
            else:
                sinter_T = 0
            reward -= sinter_T

            # 1B) Calcination temperature of material using RF
            if 'calcine' in self.tasks:
                calcine_T = predict_calcine(self.state)
            else:
                calcine_T = 0
            reward -= calcine_T

            # 2) Formation energy of material using ROOST
            if 'form_e' in self.tasks: # Positive formation energy taken as good
                form_e = predict_formation_energy(self.state)
            else:
                form_e =  0
            reward -= form_e

            # 3) Bulk modulus of material using ROOST
            if 'bulk_mod' in self.tasks:
                bulk_mod = predict_bulk_mod(self.state)
            else:
                bulk_mod = 0
            reward += 500*bulk_mod

            # 4) Shear modulus of material using ROOST
            if 'shear_mod' in self.tasks:
                shear_mod = predict_shear_mod(self.state)
            else:
                shear_mod = 0
            reward += shear_mod

            # 5) Band gap of material using ROOST
            if 'band_gap' in self.tasks:
                band_gap = predict_band_gap(self.state)
                target_band_gap = 1.0
                # reward_bg = -np.e**(10*max(0, np.abs(band_gap - target_band_gap)))
                reward_bg = np.e**(10*band_gap)
            else:
                reward_bg = 0
            reward += reward_bg

        else:
            reward = 0

        return reward

    def en(self): # Electronegativity
        if self.counter == self.max_steps:
            # Constraint: electronegativity
            try:
                chemical = Composition(self.state)
                if check_electronegativity(chemical):
                    en = 1.0 # 1.0 = en OK, 0.0 = en not OK
                else:
                    en = 0.0
            except Exception:
                en = 0.0
                print('Compound not valid')
        else:
            en = None

        return en
    
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
        self.counter += 1 # Counter increased before since we start with step 1 instead of 0 (Just convention)

        reward = self.reward()
        en     = self.en() # electronegativity: 1.0 for OK, 0.0 for not OK

        # Record state and action
        # s_a_r = ([onehot_target(old_state), step_to_one_hot([self.counter])[0]], action, reward) # One-hot states for storing into path, actions are already one-hot
        if old_state == '': # if empty string (starting state), don't featurize with Magpie, but with a vector of zeroes instead
            s_a_r_c = ([featurize_target(old_state), step_to_one_hot([self.counter])[0]], action, reward, en) # One-hot states for storing into path, actions are already one-hot
        else:
            s_a_r_c = ([featurize_target(old_state), step_to_one_hot([self.counter])[0]], action, reward, en) # One-hot states for storing into path, actions are already one-hot

        self.path.append(s_a_r_c) # append (state, action, reward, constraint) - [material, step], [element, composition], reward

tasks = configs['tasks']
env = ConstrainedMaterialEnvironment(element_set = element_set,
                          comp_set =  comp_set,
                          tasks = tasks)

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

def generate_random_act(oxide = False):
    '''
    Generates random action.
    Returns: List. [tuple(element), tuple(composition)]
    '''
    # Sample random action
    element = random.sample(element_set, 1)
    if oxide:
        element = tuple(element_to_one_hot('O')[0])
        comp = random.sample(comp_set[1:], 1) # consider only non-zero compositions
    else:
        element = tuple(element_to_one_hot(element)[0])
        comp = random.sample(comp_set, 1)
    comp = tuple(comp_to_one_hot(comp)[0])
    action = [element, comp]
    return action

def generate_random_ep(max_steps = 5, oxide = False):
    '''
    Generates an episode with random policy
    
    Args:
    max_steps: Int

    Returns: 
    env.path (an episode): List of SAR data in the form of [[material, step], [element, composition], reward]
    
    '''
    env.initialize()

    for i in range(max_steps):
        
        # if i == max_steps-1: # if first step, choose oxygen only
        #     element, comp = action
        #     element = tuple(element_to_one_hot('O')[0])
        #     action  = [element, comp]
        if i == max_steps-1: # if last step, choose oxygen only
            action = generate_random_act(oxide = oxide)
        else:
            action = generate_random_act(oxide = False)

        # Take step with action
        env.step(action)
        print('step:', env.counter)
        print('state:',env.state)
        print('reward:',env.reward())
        print('en:', env.en())
        # print(env.num_steps_taken)
        print('')
    return env.path
    # return env.state
# ep = generate_random_ep(oxide = True)

# # Generating random oxides - make sure to change env.path to env.state in generate_random_ep()
# if __name__ == "__main__":
#     random_oxides = []
#     for i in range(1000):
#         compound = generate_random_ep(oxide = True)
#         random_oxides.append(compound)               
#     with open('./training_data/random_oxides-3.pkl', 'wb') as f:
#         pickle.dump(random_oxides, f, pickle.HIGHEST_PROTOCOL)
#     print(random_oxides)

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
        state, action, reward, constraint = episode[step]

        if step == env.max_steps - 1: # If terminal state
            G = reward # Return = reward
        else:
            G = reward + disc_factor*G # Return = reward + discounted return of the PREVIOUS state
        
        if step == env.max_steps - 1:  # If terminal state
            en = constraint # 1.0 for en OK, 0.0 for en not OK
        else:
            pass # no need to update en, let the previous states be assigned the same en as final compound

        data_point = [state, action, G, en]
        Q_data.append(data_point)
    return Q_data
# print(extract_data_from_ep(episode = ep))




# # ========= FOR RANDOM POLICY ===========
# if __name__ == "__main__":
#     start = time.time()
#     # Generate random episodes
#     num_eps = 10000
#     episodes = []
#     for j in range(num_eps):
#         episode = generate_random_ep(oxide = True)
#         episodes.append(episode)

#     Q_data_random = []
#     # Extract Q_data from episodes
#     for episode in episodes:
#         Q_data = extract_data_from_ep(episode)
#         Q_data_random.append(Q_data)
#     end = time.time()
#     print('time taken:', end - start)\
    
#     # Save Q_data
#     with open('./data/oxides_sinter/Q_data_random.pkl', 'wb') as f:
#         pickle.dump(Q_data_random, f, pickle.HIGHEST_PROTOCOL)