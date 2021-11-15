import numpy as np
from one_hot import element_set, comp_set, one_hot_to_element, element_to_one_hot

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
        self.comp_set  = comp_set
        self.init_mat    = init_mat
        self.max_steps   = max_steps
        self.terminated  = False
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

    def reward(self): # TO-DO
        ####### run temperature CVAE model on material
        a = 1
        return 1
    
    def step(self, action): # TO-DO
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
        # if self.counter >= self.max_steps:
        #     self.terminated = True

        # Record state and action
        state_action = (self.state, self.counter, action)
        self.path.append(state_action)

        # Take action
        element = [one_hot_to_element([act]) for act in action][0][0] # String form of element
        if self.counter == 0: # If empty compound, initialize state
            self.state = element
        else: # Else not initial state, so add element to exisiting state
            self.state += element
        # print(element)
        # self.state = 

        self.counter += 1
        result = (self.state, self.reward()) # result is a tuple of new state and reward from taking the action
    
env = MaterialEnvironment(element_set = element_set,
                          comp_set =  comp_set,)

# print(env.state)
# print(env.num_steps_taken)

env.step([(1., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)])
env.step([(0., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.)])
env.step([(0., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.)])       
print('counter:', env.counter)
print('state:',env.state)
# print(env.state)
# print(env.num_steps_taken)


