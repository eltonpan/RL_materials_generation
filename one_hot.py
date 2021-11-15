import numpy as np
import re
from pymatgen import Composition, Element
from env import element_set, number_set

def _get_target_char_sequence(compound_string):
    comp = Composition(compound_string).formula.replace(" ", "")
    char_seq = re.findall(r"[A-Z][a-z]?|[0-9]|.", comp)

    return char_seq
def onehot_target(target):
    all_elements = [Element.from_Z(i).symbol for i in range(1, 104)]
    all_digits = [str(i) for i in range(0, 10)]
    target_charset = ["<NULL>"] + all_elements + all_digits + ["."]
    charset = ["<NULL>"] + all_elements + all_digits
    charset_size = len(charset)
    target_charset_size = len(target_charset)
    max_material_length = 14
    max_target_length = 40
    min_operation_length = 3
    max_operation_length = 20
    paper_batch_size = 50
    max_num_precs = 5
    try:
        mat_char_seq = _get_target_char_sequence(target)
        filtered_char_seq = [str(c) for c in mat_char_seq if str(c) in target_charset]
        char_seq_vec = np.zeros(shape=(max_target_length, target_charset_size))
        for j, char in enumerate(filtered_char_seq):
            char_vec = np.zeros(shape=(target_charset_size,))
            char_vec[target_charset.index(char)] = 1.0
            char_seq_vec[j] = char_vec
        for i in range(len(char_seq_vec), max_target_length):
            char_vec = np.zeros(shape=(target_charset_size,))
            char_vec[0] = 1.0
            char_seq_vec[i] = char_vec
    except Exception as e:
            print(target) 
    return char_seq_vec

# Find element to one-hot dictionary
element_set = ['Te', 'Sc', 'C', 'Hg', 'Ru', 'Na', 'Co', 'Mo', 'I', 'Tm', 'F', 'Al', 'Pd', 'Fe', 'Th', 'Cs', 'Gd', 'W', 'Ta', 'Dy', 'Pb', 'Rb', 'Ba', 'Ce', 'Ga', 'Tl', 'Mn', 'B', 'Ni', 'Tb', 'Hf', 'Ge', 'V', 'Ho', 'In', 'Cd', 'Yb', 'Pt', 'Nd', 'Mg', 'Zr', 'Re', 'P', 'Sb', 'O', 'N', 'Zn', 'Au', 'Lu', 'Be', 'Cr', 'Ag', 'Pu', 'Si', 'Cu', 'Os', 'Li', 'Am', 'Pr', 'S', 'As', 'Ti', 'Nb', 'Eu', 'H', 'Br', 'La', 'Er', 'Sm', 'Cl', 'Sn', 'K', 'Sr', 'Rh', 'Se', 'U', 'Y', 'Bi', 'Ca', 'Ir']
element_to_one_hot_dict = {}
for element_idx in range(len(element_set)):
    element = element_set[element_idx]
    enc = np.zeros(len(element_set))
    enc[element_idx] = 1
    element_to_one_hot_dict[element] = enc

# Find one-hot dictionary to element
one_hot_to_element_dict = {}
for element in element_to_one_hot_dict.keys():
    one_hot_to_element_dict[tuple(element_to_one_hot_dict[element])] = element # find the inverse mapping

def element_to_one_hot(elements):
    """
    converts a single element, or a list of multiple elements into their one-hot form

    Args:
    elements: List. list of elements

    Returns:
    element_to_one_hot: np.array() of shape (no. of elements in input of this function, no. of elements in element_set)
    '''

    """
    element_to_one_hot = []
    for element in elements:
        enc = element_to_one_hot_dict[element]
        element_to_one_hot.append(enc)
    return element_to_one_hot

def one_hot_to_element(one_hot_encs):
    """
    converts a single element, or a list of multiple elements in one-hot form into their string form

    Args:
    elements: List. list of elements in one-hot form 
    i.e. [ tuple(1,0,...,0,0),
           ...
           tuple(0,1,...,0,0)]

    Returns:
    one_hot_to_element: List of elements in string form
    """
    one_hot_to_element = []
    for enc in one_hot_encs:
        element = one_hot_to_element_dict[enc]
        one_hot_to_element.append(element)
    return one_hot_to_element


# print(element_to_one_hot(['Te', 'C', 'Ru']))
print(one_hot_to_element(
[(0., 0., 0., 0, 0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.)]

))
# print(onehot_target('BaTiO3').reshape(1, 40, 115).shape)

