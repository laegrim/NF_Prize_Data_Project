import numpy as np
import random
import pickle
from scipy import sparse
#import scipy
from scipy.sparse.linalg import svds

with open('../data/name_one_hot_dict.pickle', 'rb') as handle:
    name_one_hot = pickle.load(handle)
    
with open('../data/mean_rating.pickle', 'rb') as handle:
    mean_rating = pickle.load(handle)
    
with open('../data/median_rating.pickle', 'rb') as handle:
    median_rating = pickle.load(handle)

with open('../data/movie_var.pickle', 'rb') as handle:
    movie_var = pickle.load(handle)
    
min_mean_movie = 1.2878787878787878
max_mean_movie = 4.723269925683507
max_median_movie = 5
min_median_movie = 1
min_var_movie = 0.33648393194707
max_var_movie = 2.692905509387504
year_one = 1999
year_end = 2005
year_range = year_end - year_one

def user_vector(user_matrix, user_ids, user):
    return user_matrix[user_ids[user]]

def create_user_ids(dset):
    #make user id dictionary
    user_ids = list(set(dset[:,1]))
    user_ids = {user:i for i, user in enumerate(user_ids)}
    return user_ids

def create_user_maxtrix(training_data, user_ids, k_n = 50):
    
    movie_ids = list(range(0, 17771))
    
    # sort training data to speed up building sparse matrix
    training_data = training_data[training_data[:,1].argsort()]
    
    # make empty sparse matrix
    user_rating_matrix = sparse.lil_matrix( (len(user_ids), 17771), dtype=np.int8 )
    
    # iterate through data adding rating to movie matrix
    for i in training_data:
        user_rating_matrix[user_ids[i[1]], i[0]] = i[2]
    
    # convert to condensed matrix for faster multiplication
    user_rating_matrix = user_rating_matrix.tocsr()
    
    # call Singular Value Decomposition
    U, sigma, Vt = svds(user_rating_matrix.asfptype(), k_n)
    
    # return reduced dim matrix
    return U * sigma

def norm(max_val, min_val, val):
    return (val - min_val) / (max_val - min_val)



naive_input_size = 497965


def user_name_layer(original_name, name_dict):
    one_hot_layer = np.zeros(len(name_dict),)
    one_hot_layer[name_dict[original_name]] = 1
    return one_hot_layer

def movie_name_layer(movie_name):
    one_hot_layer = np.zeros(17770,)
    one_hot_layer[movie_name - 1] = 1
    return one_hot_layer



def review_date(date, normalized = False):
    date_array = np.zeros(3,)
    if(not normalized):
        date_array[0] = date // 10000 - year_one
        date_array[1] = date % 10000 // 100
        date_array[2] = date % 100
    else:
        date_array[0] = norm(year_range, year_one, date // 10000 - year_one)
        date_array[1] = norm(12, 1, date % 10000 // 100)
        date_array[2] = norm(31, 1, date % 100)
        
    return date_array

def latent_user_input_layer(data, user_matrix, user_ids, normalized = False):
    if(not normalized):
        layer = np.concatenate((user_vector(user_matrix, user_ids, data[1]),
                               movie_name_layer(data[0])), axis=0)
        return layer
    else:

        layer = np.concatenate((user_vector(user_matrix, user_ids, data[1]), movie_name_layer(data[0]) ), axis=0)
        
    
    return layer


def input_layer(data, normalized = False):
    statistics = np.zeros(3,)
    if(not normalized):
        statistics[0] = mean_rating[data[0]]
        statistics[1] = median_rating[data[0]]
        statistics[2] = movie_var[data[0]]
        layer = np.concatenate((user_name_layer(data[1], name_one_hot),
                               movie_name_layer(data[0]),
                               review_date(data[3]),
                               statistics), axis=0)
        return layer
    else:

        statistics[0] = norm(max_mean_movie,min_mean_movie, mean_rating[data[0]])
        statistics[1] = norm(max_median_movie,min_median_movie, median_rating[data[0]])
        statistics[2] = norm(max_var_movie, min_var_movie, movie_var[data[0]])
        layer = np.concatenate((user_name_layer(data[1], name_one_hot),
                               movie_name_layer(data[0]),
                               review_date(data[3], normalized = True),
                               statistics), axis=0)
        
    
    return layer

def Generate_User_Latent_Feedforward(dataset, 
                                     batch_size, 
                                     feature_shape, 
                                     label_shape, 
                                     feature_transformation_function, 
                                     label_transformation_function, 
                                     user_matrix, 
                                     user_ids, 
                                     normalized = False):
    
    features = np.zeros((batch_size, *feature_shape))
    labels = np.zeros((batch_size, *label_shape))
    index_set = set(range(len(dataset)))
    
    while True:
            
        batch_indicies = []    
        if (len(index_set) < batch_size):
            index_set = set(range(len(dataset)))
        #import pdb; pdb.set_trace()
        batch_indicies = random.sample(index_set, batch_size)
        index_set = index_set.difference(batch_indicies)
        
        for i, batch_index in enumerate(batch_indicies):
            features[i] = feature_transformation_function(dataset[batch_index], user_matrix, user_ids, normalized)
            labels[i] = label_transformation_function(dataset[batch_index])
            
        yield (features, labels)

def Generate_Naive_Feedforward(dataset, batch_size, feature_shape, label_shape, feature_transformation_function, label_transformation_function, normalized = False):
    
    features = np.zeros((batch_size, *feature_shape))
    labels = np.zeros((batch_size, *label_shape))
    index_set = set(range(len(dataset)))
    
    while True:
            
        batch_indicies = []    
        if (len(index_set) < batch_size):
            index_set = set(range(len(dataset)))
            
        batch_indicies = random.sample(index_set, batch_size)
        index_set = index_set.difference(batch_indicies)
        
        for i, batch_index in enumerate(batch_indicies):
            features[i] = feature_transformation_function(dataset[batch_index], normalized)
            labels[i] = label_transformation_function(dataset[batch_index])
            
        yield (features, labels)
        
        

        
def Generate_Naive_Feedforward_Contig(dataset, batch_size, feature_shape, label_shape, feature_transformation_function, label_transformation_function, normalized = False):
    
    features = np.zeros((batch_size, *feature_shape))
    labels = np.zeros((batch_size, *label_shape))
    index_list = list(range(len(dataset)))
    
    while True:
            
        batch_indicies = []    
        if (len(index_list) < batch_size):
            index_list = list(range(len(dataset)))
        
        #find the index of the first sample in the batch randomly    
        batch_first_index = random.randrange(len(index_list) - batch_size - 1)
        batch_indicies = list(range(batch_first_index, batch_first_index + batch_size))
        index_list = [i for i in index_list if i not in batch_indicies]
        
        for i, batch_index in enumerate(batch_indicies):
            features[i] = feature_transformation_function(dataset[batch_index], normalized)
            labels[i] = label_transformation_function(dataset[batch_index])
            
        yield (features, labels)