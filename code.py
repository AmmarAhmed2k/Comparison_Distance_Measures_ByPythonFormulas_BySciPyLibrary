# -*- coding: utf-8 -*-
"""
@author: Ammar Ahmed Siddiqui
Credits: https://machinelearningmastery.com/distance-measures-for-machine-learning/

"""
#_____________________________________________________
# calculating euclidean distance between vectors

from math import sqrt
import numpy as np

# all distance related imports
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial import minkowski_distance
from scipy.spatial.distance import hamming
from scipy.spatial import distance
from scipy.spatial.distance import jaccard
from sklearn.metrics.pairwise import cosine_similarity


#_____________________________________________________
# Additional Formulas for validation

# calculate euclidean distance
def euclidean_byformula(a, b):
	return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
# calculate manhattan distance
def manhattan_byformula(a, b):
	return sum(abs(e1-e2) for e1, e2 in zip(a,b))
# calculate minkowski distance
def minkowski_byformula(a, b, p):
 return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)
# calculate hamming distance
def hamming_byformula(a, b):
 return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)

def chebyshev_byformula(point1, point2):
    """
    Calculate the Chebyshev distance between two points using the formula.
    
    Args:
        point1 (list or tuple): The coordinates of the first point.
        point2 (list or tuple): The coordinates of the second point.
        
    Returns:
        float: The Chebyshev distance between the two points.
    """
    distance = max(abs(point1[i] - point2[i]) for i in range(len(point1)))
    return distance

def dot_product(vector1, vector2):
    """
    Compute the dot product of two vectors.
    
    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.
        
    Returns:
        float: The dot product of the two vectors.
    """
    return sum(x * y for x, y in zip(vector1, vector2))

def magnitude(vector):
    """
    Compute the magnitude of a vector.
    
    Args:
        vector (list): The vector.
        
    Returns:
        float: The magnitude of the vector.
    """
    return sqrt(sum(x**2 for x in vector))

def cosine_similarity_byformula(vector1, vector2):
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.
        
    Returns:
        float: The cosine similarity between the two vectors.
    """
    dot_prod = dot_product(vector1, vector2)
    magnitude_prod = magnitude(vector1) * magnitude(vector2)
    return dot_prod / magnitude_prod


def jaccard_byformula(set1, set2):
    """
    Compute the Jaccard distance between two sets.
    
    Args:
        set1 (set or list): The first set.
        set2 (set or list): The second set.
        
    Returns:
        float: The Jaccard distance between the two sets.
    """
    set1 = set(set1)
    set2 = set(set2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    jaccard_distance = 1.0 - (intersection / union)
    return jaccard_distance

#_____________________________________________________
# define data for work
row1 = [10, 20, 15, 10, 5]
row2 = [12, 24, 18, 8, 7]

# hamming requirement [Binary data only]
row1h = [1, 0, 0, 0, 0, 0]
row2h = [0, 0, 0, 1, 0, 0]

# Vector requirement for Cosine Similarity Distance Measure
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

# Sets for use in Jaccard Distance Example
set1 = [1, 2, 3, 4]
set2 = [3, 4, 5, 6]


# calculate various distances
dist_eu_f = euclidean_byformula(row1, row2)
dist_eu_L = euclidean(row1, row2)
dist_mn_f = manhattan_byformula(row1, row2)
dist_mn_L = cityblock(row1, row2)

dist_mink_f_p1 = minkowski_byformula(row1, row2, 1)
dist_mink_f_p2 = minkowski_byformula(row1, row2, 2)
dist_mink_f_p3 = minkowski_byformula(row1, row2, 3)

dist_mink_L_p1 = minkowski_distance(row1, row2, 1)
dist_mink_L_p2 = minkowski_distance(row1, row2, 2)
dist_mink_L_p3 = minkowski_distance(row1, row2, 3)

dist_chb_f     = chebyshev_byformula(row1, row2)
dist_chb_L     = distance.chebyshev(row1, row2)


dist_ham_f     = hamming_byformula(row1h, row2h)
dist_ham_L     = hamming(row1h, row2h)


dist_cos_f     = cosine_similarity_byformula(vector1, vector2)

# some conversions required

v1 = np.array(vector1)
v2 = np.array(vector2)

v1 = v1.reshape(1,-1)
v2 = v2.reshape(1,-1)

dist_cos_L     = cosine_similarity(v1, v2)

dist_jac_f     = jaccard_byformula(set1,set2)
dist_jac_L     = jaccard(set1,set2)


# __________________________________ Printing output______


print(f"Euclidean by Formula..........= {dist_eu_f}")
print(f"Euclidean by Scipy............= {dist_eu_L}")

print(f"Manhattan by Formula.........= {dist_mn_f}")
print(f"Manhattan by Scipy...........= {dist_mn_L}")

print(f"Minkowski by Formula, p is 1 = {dist_mink_f_p1}")
print(f"Minkowski by Scipy ,  p is 1 = {dist_mink_L_p1}")

print(f"Minkowski by Formula, p is 2 = {dist_mink_f_p2}")
print(f"Minkowski by Scipy   ,p is 2 = {dist_mink_L_p2}")

print(f"Minkowski by Formula, p is 3 = {dist_mink_f_p3}")
print(f"Minkowski by Scipy   ,p is 3 = {dist_mink_L_p3}")


print(f"chebyshev by Formula........ = {dist_chb_f}")
print(f"chebyshev by Scipy.......... = {dist_chb_L}")

print("____ Note dataset for hamming is changed___")

print(f"Haming by Formula............= {dist_ham_f}")
print(f"Haming by Scipy..............= {dist_ham_L}")

print("____ Note dataset for cosine similarity is changed___")

print(f"Cosine by Formula............= {dist_cos_f}")
print(f"Cosine by Scipy..............= {dist_cos_L}")

print("____ Note dataset for Jaccard Distance is changed___")

print(f"Jaccard by Formula...........= {dist_jac_f}")
print(f"Jaccard by Scipy.............= {dist_jac_L}")

