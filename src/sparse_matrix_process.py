import numpy as np
import scipy.stats as ss

# 1) One good data structure for sparse vectors would be by using
#    a hashmap.  In the case of Python, this could be implemented with
#    a dictionary.  The keys of this dictionary is the location in the
#    original vector where the non-zero value is located. The value
#    would be the non-zero value

def get_hashmap(vec):
    """
    input:  a python list representation of a vector
    output: a sparse representation (given zeros) of the given vector using a python
            dictionary
    """
    hashmap = dict()
    for i, value in enumerate(vec):
        if value!=0:
            hashmap[i] = value
    return hashmap


def get_user_hashmap(user_data):
    '''make a list of user sparse hashman'''
    hash_list = []
    for user in user_data:
        hashmap = get_hashmap(user)
        hash_list.append(hashmap)
    return hash_list

def symmetrize(a):
    '''make a upper or lower triagnle matrix symmetric'''
    return a + a.T - np.diag(a.diagonal())

# 2) Implementation: See below

def sparse_dot_product(a, b):
    """
    input: a, b are both representations of the vectors in a dictionary
           hashmap.  Only the non-zero values are stored.  The key is the
           position of the non-zero value in the vector

    output:Dot product of the two vectors
    """
    dot_product = 0
    for key in a.viewkeys() & b.viewkeys():
        dot_product += a[key] * b[key]
    return dot_product



def pearson_sparse_sim(user1,user2):
    '''
    input: user1, user2: dictionary representation of user's artist column
    index to their listen counts

    a function to compute pearson similarity in sparse matrix using representations
    of hashmap for faster speed

    '''
    user1_mu = np.mean(user1.values())
    user2_mu = np.mean(user2.values())
    common_keys = user1.viewkeys() & user2.viewkeys()
    user1_ratings = np.array([user1[key] for key in common_keys])
    user2_ratings = np.array([user2[key] for key in common_keys])
    user1_demean = user1_ratings - user1_mu
    user2_demean = user2_ratings - user2_mu
    nom=np.sum(user1_demean*user2_demean)

    denom = np.sqrt(np.sum(user1_demean**2)*np.sum(user2_demean**2))
    pear = nom/denom
    return pear

def adj_pearson_sim(user1,user2,common=5):

    '''
    adjusted pearson correlation to penalize simialrity scores that are
    based on small number of overlapping items, reflecting lack of confidence
    input: user 1, user 2: arrays of user's rating/latent factors of music_pref
    common: int: the number of minimum overlapping artists before punishment
    is alloted
    output: float: modified pearson's similarity between user1 and user 2
    '''
    p = pearson_sparse_sim(user1,user2)
    common_keys = user1.viewkeys() & user2.viewkeys()
    penality= min(len(common_keys), common)/float(common)
    adj_p = penality*p
    return adj_p

def spearman_rank(user1,user2):
    '''
    input: user1, user2: dictionary representation of user's artist column
    index to their listen counts

    a function to compute spearman rank in sparse matrix using representations
    of hashmap for faster speed

    '''
    common_keys = list(user1.viewkeys() & user2.viewkeys())
    common = len(common_keys)
    if not common:
        return 0
    user1_c = [user1[k] for k in common_keys]
    user2_c = [user2[k] for k in common_keys]
    user1_r = np.array(ss.rankdata(user1_c))
    user2_r = np.array(ss.rankdata(user2_c))
    nom = 6*np.sum( (user1_r-user2_r)**2)
    denom = common*(common**2-1)
    spear = 1- nom/float(denom)
    return spear



def mod_rating_distance(user_data):
    hash_list = get_user_hashmap(user_data)
    n_users = len(hash_list)
    matrix=np.zeros((n_users,n_users))
    for i in range(n_users):
        user1 = hash_list[i]
        print i
        for j in range(i,n_users):
            user2 = hash_list[j]
            pearson = pearson_sparse_sim(user1,user2)
            matrix[i,j] = pearson
    return matrix



if __name__ == '__main__':

    # example usage:

    a = [1,0,4,0,0,1,1,0,4,0,0,1,1,0,4,0,0,1,1,0,4,0,0,1]
    b = [0,0,2,0,0,1,1,0,4,0,0,1,0,0,2,0,0,1,0,0,2,0,0,1]

    # let's fill it with a lot of zeros.
    a.extend([0] * 100000)
    b.extend([0] * 100000)

    a_hash = get_hashmap(a)
    b_hash = get_hashmap(b)

    dot_product = sparse_dot_product(a_hash, b_hash)
    # should print 45
    print(dot_product)




def _take_n_out(userid,hash_ratings,leave_out):
    test_set = {}
    for i in range(len(hash_ratings)):
        if len(hash_ratings[i]) <= leave_out:
            test_set[userid[i]] = []

        elif leave_out<len(hash_ratings[i])<(leave_out+1)*2:
            keys_rm = choice(hash_ratings[i].keys(),leave_out,replace=False)
            for k in keys_rm:
                test_set[userid[i]] = k
                hash_ratings[i].pop(k)
        else:
            perc = np.median(hash_ratings[i].values())
            keys = [k for k,v in hash_ratings[i].items() if v>=perc]
            keys_rm = choice(keys,leave_out,replace=False)
            for k in keys_rm:
                test_set[userid[i]] = k
                hash_ratings[i].pop(k)


def recommendation_accuracy(artist_rec_index,test_set):
    match_count = 0.0
    all_matches = {}
    for key in artist_rec_index:
        print key
        matches = set(test_set[key]).intersection(set(artist_rec_index[key]))
        if matches:
            all_matches[key] = list(matches)
            match_count+=1
    return match_count/len(artist_rec_index), all_matches
