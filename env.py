import numpy as np

class MaterialEnvironment():
    """
    Defines the Markov decision process of generating an inorganic material.
    """
    def __init__(self, 
                element_set,
                number_set,
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
        self.number_set  = number_set
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
          action: List of np.array. 1st element is np.array of shape (1, num_elements), 2nd element is np.array of shape (1,10) 

        Returns:
          results: Namedtuple containing the following fields:
            * state: The molecule reached after taking the action.
            * reward: The reward get after taking the action.
            * terminated: Whether this episode is terminated.

        Raises:
          ValueError: If the number of steps taken exceeds the preset max_steps, or
            the action is not in the set of valid_actions.
        """
        if self.counter >= self.max_steps:
            self.terminated = True

        # Record state and action
        state_action = (self.state, action)
        self.path.append(state_action)

        # Take action # TO-DO
        # self.state = 

        self.counter += 1
        result = (self.state, self.reward()) # result is a tuple of new state and reward from taking the action
    

element_set = ['Te', 'Sc', 'C', 'Hg', 'Ru', 'Na', 'Co', 'Mo', 'I', 'Tm', 'F', 'Al', 'Pd', 'Fe', 'Th', 'Cs', 'Gd', 'W', 'Ta', 'Dy', 'Pb', 'Rb', 'Ba', 'Ce', 'Ga', 'Tl', 'Mn', 'B', 'Ni', 'Tb', 'Hf', 'Ge', 'V', 'Ho', 'In', 'Cd', 'Yb', 'Pt', 'Nd', 'Mg', 'Zr', 'Re', 'P', 'Sb', 'O', 'N', 'Zn', 'Au', 'Lu', 'Be', 'Cr', 'Ag', 'Pu', 'Si', 'Cu', 'Os', 'Li', 'Am', 'Pr', 'S', 'As', 'Ti', 'Nb', 'Eu', 'H', 'Br', 'La', 'Er', 'Sm', 'Cl', 'Sn', 'K', 'Sr', 'Rh', 'Se', 'U', 'Y', 'Bi', 'Ca', 'Ir']
number_set  = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
env = MaterialEnvironment(element_set = element_set,
                          number_set =  number_set,)
print(env.state)
print(env.num_steps_taken)

env.step('a')

print(env.state)
print(env.num_steps_taken)
