import numpy as np
from itertools import izip
import pandas as pd
from sparse_matrix_process import get_user_hashmap,symmetrize
from numbers import Number
import scipy.stats as ss
from numpy.random import choice

class Personality_Rating_Hybrid(object):

    def __init__(self,sim_option='adj_pearson'):

        '''
        input:
            sim_option: { "pearson_sim": user rating similarity by pearson correlation,
            "adj_pearson": user rating similarity adjusted for the amount of overlapping artists}
            user's personality similarity in user similarity matrix
            weight: float, the weight placed on user's personality in hybrid model
            nd rating
        '''
        sim_options= {'pearson_sim': self._pearson_sparse_sim,'adj_pearson': self._adj_pearson_sim,'spearman_sim': self._spearman_rank}
        self.method =None
        self.sim_function= sim_options[sim_option]
        self.rating_dist = None
        self.personality_dist = None
        self.user_sim = None
        self.user_id_index = None
        self.artist_id_index = None
        self.ratings = None
        self.artist_recommendations = {}
        self.artist_rec_index = {}
        self.train_hash_list = {}
        self.weight = 0.5
        self.test_set = {}
        self.matches = {}
        self.baseline_artist = {}
        self.baseline_index = {}

    def train(self, ratings, user_column = 'user_id', personality = None, method='rating', weight=0.5):

        '''input:

        user_id: string: the name of the column that contains user id
        ratings: a pandas dataframe with columns representing the user's rating
        for an artist, one column is the user_id
        personality: a pandas dataframe with columns representing each attribute
        in big 5s
        Note the rows of ratings must match the rows of personality for the user

        method:
        str: the method to calculate user similarity
        'rating': using user rating of artists
        'personality': using user's big 5 personality
        'hybrid': use a combined rating and personality similarity weighted by weighted

        weight: float, user_sim = weight*user_rating + (1-weight)*user_personality
        '''
        ratings = ratings.copy()
        user_id = ratings[user_column]
        ratings.drop(user_column,axis=1,inplace=True)
        self.weight = weight
        self.method = method
        #create user to row index dictionary'''
        self.user_id_index = { ind:uid for ind,uid in enumerate(user_id)}
        #create artist to column index dictionary'''
        artists = ratings.columns.values
        self.artist_id_index = {ind:aid for ind,aid in enumerate(artists)}
        ratings = np.array(ratings)
        #store ratings'''
        self.ratings = ratings
        if method != 'rating':
            # store personality
            personality = np.array(personality)
            self.personality = personality

        if method == 'hybrid':
            self.personality_dist = self.personality_sim(personality)
            self.train_hash_list = get_user_hashmap(ratings)
            self.rating_dist = self.mod_rating_distance()
            self.user_sim = weight*self.personality_dist + (1-weight)*self.rating_dist
        elif method == 'rating':
            self.train_hash_list = get_user_hashmap(ratings)
            self.user_sim = self.mod_rating_distance()
        elif method == 'personality':
            self.user_sim = self.personality_sim(personality)
            self.train_hash_list = get_user_hashmap(ratings)

    def personality_sim(self,personality,otherpersonality=None):
        '''input: np.array, list of list: user's big 5 personality
        output: np.array: user's personality similarity matrix
        save personality similiarity
         '''
        personality_dist = np.corrcoef(personality,otherpersonality)
        return personality_dist
    # def pearson_sim(self,user1,user2):
    #
    #     '''
    #     calculate pearson coefficient between two users based on the common
    #     items between two users
    #     input: user 1, user 2: arrays of user's rating/latent factors of
    #            music_pref
    #     output: float: pearson's similarity between user1 and user 2 based on
    #             common items
    #      '''
    #     user1_mu = np.mean(user1)
    #     user2_mu = np.mean(user2)
    #     common= (user1!=0) & (user2!=0)
    #     user1_demean = user1[common] - user1_mu
    #     user2_demean = user2[common] - user2_mu
    #     nom=np.sum(user1_demean*user2_demean)
    #     denom = np.sqrt(np.sum(user1_demean**2)*np.sum(user2_demean**2))
    #     pear = nom/denom
    #     return pear
    def _adj_pearson_sim(self,user1,user2,common=5):

        '''
        adjusted pearson correlation to penalize simialrity scores that are
        based on small number of overlapping items, reflecting lack of confidence
        input: user 1, user 2: arrays of user's rating/latent factors of music_pref
        common: int: the number of minimum overlapping artists before punishment
        is alloted
        output: float: modified pearson's similarity between user1 and user 2
        '''
        p = self._pearson_sparse_sim(user1,user2)
        common_keys = user1.viewkeys() & user2.viewkeys()
        penality= min(len(common_keys), common)/float(common)
        adj_p = penality*p
        return adj_p

    def _pearson_sparse_sim(self,user1,user2):
        '''
        input: user1, user2: dictionary representation of user's artist column
        index to their listen counts

        a function to compute pearson similarity in sparse matrix using representations
        of hashmap for faster speed

        '''
        user1_mu = np.mean(user1.values())
        user2_mu = np.mean(user2.values())
        common_keys = list(user1.viewkeys() & user2.viewkeys())
        if common_keys:
            user1_ratings = np.array([user1[k] for k in common_keys])
            user2_ratings = np.array([user2[k] for k in common_keys])
            user1_demean = user1_ratings - user1_mu
            user2_demean = user2_ratings - user2_mu
            nom=np.sum(user1_demean*user2_demean)
            denom = np.sqrt(np.sum(user1_demean**2)*np.sum(user2_demean**2))
            if not denom:
                return 0
            else:
                pear = nom/denom
                return pear
        else:
            return 0

    def _spearman_rank(self,user1,user2):
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
        user1_r = ss.rankdata(user1_c)
        user2_r = ss.rankdata(user2_c)
        nom = 6*np.sum(user1_r-user2_r)
        denom = common*(common**2-1)
        spear = 1-nom/float(denom)
        return spear

    def mod_rating_distance(self):
        '''
        iterate through each users and calculate user distance using the choosen
        distance metrics

        returns: user distance matrix
         '''
        #finding similarity between user to every other users
        n_users = len(self.train_hash_list)
        matrix=np.zeros((n_users,n_users))
        for i in range(n_users):
            user1 = self.train_hash_list[i]
            for j in range(i,n_users):
                user2 = self.train_hash_list[j]
                pearson = self.sim_function(user1,user2)
                matrix[i,j] = pearson
        #fill in nan values with zeroes
        matrix = symmetrize(matrix)
        matrix=np.nan_to_num(matrix)

        return matrix

    # def rating_distance_metrics(self,user_data):
    #
    #     n_users = len(user_data)
    #     matrix=np.zeros((n_users,n_users))
    #     for i in range(n_users):
    #         print i
    #         for j in range(i,n_users):
    #             pearson = self.sim_function(user_data[i,],user_data[j,])
    #             matrix[i,j] = pearson
    #     '''create a full, symmetric distance matrix'''
    #     inds = np.triu_indices_from(matrix,k=1)
    #     matrix[(inds[1], inds[0])] = matrix[inds]
    #
    #     return matrix

    def recommend(self,id_col='user_id',ratings=None,personality=None,top=5,n_artist=5):
        '''input:
        user_id: string: the name of the column that contains users' ids
        ratings (optional): a pandas dataframe with columns representing the
        user's rating for an artist.
        personality: a pandas dataframe with columns representing each attribute
         in big 5, row index representing the user
        top (int): the number of most similar users to choose recommendations from
        n_artist (int): the number of top artists from each most similar users
        to recommend

        Note the rows of ratings must match the rows of personality for the user
        , userid must be present in personality matrix if user rating matrix is
        not present
        '''
        if (isinstance(ratings, pd.DataFrame) and isinstance(personality, pd.DataFrame)):
            userid = ratings[id_col]
            ratings = np.array(ratings.drop(id_col,axis=1))
            personality = np.array(personality)
            hash_list = get_user_hashmap(ratings)
            test_rating_distance = self._get_test_rating_distance(hash_list)
            personality_dist = self._personality_dist(personality)
            overall_dist = self.weight*personality_dist + self.weight*test_rating_distance
            self._distance_based_rec(userid,overall_dist,top,n_artist)

        elif isinstance(ratings, pd.DataFrame):
            userid = ratings[id_col]
            ratings = np.array(ratings.drop(id_col,axis=1))
            hash_list = get_user_hashmap(ratings)
            test_rating_distance = self._get_test_rating_distance(hash_list)
            self._distance_based_rec(userid,test_rating_distance,top,n_artist)

        elif isinstance(personality, pd.DataFrame):
            userid = personality[id_col]
            personality = np.array(personality.drop(id_col,axis=1))
            personality_dist = self._personality_dist(personality)
            self._distance_based_rec(userid,personality_dist,top,n_artist)
        else:
            print "need either personality or rating"

    def _recommend_top_n_artist(self,user,top,n_artist):
        '''input: one row of user's similarity matrix
        output:  the row index of n most similar users to that user
        '''
        top_similar_users=np.argsort(-user)[:top]
        #find their ratings based on the row index
        top_user_ratings = self.ratings.take(top_similar_users,axis=0)
        recommended_artists = set()
        recommended_artists_ind = set()
        #for each user rating
        for rate in top_user_ratings:
            #find top n artist index
            top_artists_ind=np.argsort(-rate)[:n_artist]
            #match to the artist name in a dictionary
            top_rated_artists = [self.artist_id_index[ind] for ind in top_artists_ind]
            # add artist name to the unique set
            recommended_artists.update(top_rated_artists)
            #add artist index to the unique list
            recommended_artists_ind.update(top_artists_ind)
        return recommended_artists,recommended_artists_ind

    def _personality_dist(self,personality):
        '''
        input: numpy matrix of user personality

        calls on function to recommend user artists based on personality
        similarity only'''
        length = len(personality)
        #distance of the user to each of the other users
        personality_dist = self.personality_sim(personality,self.personality)[:length,length:]
        return personality_dist

    def _get_test_rating_distance(self,hash_rating):
        '''
        input:
        hash_rating: (dictionary) hash map representation of user rating matrix
        output : numpy matrix of user similarity

        get similarity for each test user to every other training users
        based on the similarity function

        '''
        n_test_users = len(hash_rating)
        n_train_users = len(self.train_hash_list)
        matrix=np.zeros((n_test_users,n_train_users))
        for i in range(n_test_users):
            user1 = hash_rating[i]
            for j in range(n_train_users):
                user2 = self.train_hash_list[j]
                pearson = self.sim_function(user1,user2)
                matrix[i,j] = pearson
        return matrix

    def _distance_based_rec(self,userid,user_dis,top,n_artist):
        '''
        input:
        userid: numpy array,list:  users' ids
        user_dis: numpy array: user similarity matrix
        top: the top n most similar user
        n_artist: top n artist from each of the similar users

        output:
        artist_rec_index: dictionary: user id being the key and the index of
        recommended artist for that user being the values
        '''
        for uid, userdist in zip(userid,user_dis):
            print uid
            self.artist_recommendations[uid],self.artist_rec_index[uid] = self._recommend_top_n_artist(userdist,top,n_artist)
        return self.artist_rec_index
    # def _recommend_top_n_songs()

    def _take_n_out(self,userid,hash_ratings,leave_out):
        '''
        take n artists out from each test user's top 50 percentile most listened
        artist to evaluation model performance
        input:
        userid: numpy array,list:  users' ids
        hash_rating: (dictionary) hash map representation of user rating matrix
        leave_out: int: the number of artists to leave out from each test user
        output:
        the hashmap of user's rating with n artists and rating removed

        '''
        for i in range(len(hash_ratings)):
            if len(hash_ratings[i]) <= leave_out:
                self.test_set[userid[i]] = []

            elif leave_out<len(hash_ratings[i])<(leave_out+1)*2:
                keys_rm = choice(hash_ratings[i].keys(),leave_out,replace=False)
                for k in keys_rm:
                    if userid[i] not in self.test_set.keys():
                        self.test_set[userid[i]] = [k]
                    else:
                        self.test_set[userid[i]].append(k)
            else:
                perc = np.median(hash_ratings[i].values())
                keys = [k for k,v in hash_ratings[i].items() if v>=perc]
                keys_rm = choice(keys,leave_out,replace=False)
                for k in keys_rm:
                    if userid[i] not in self.test_set.keys():
                        self.test_set[userid[i]] = [k]
                    else:
                        self.test_set[userid[i]].append(k)
                    hash_ratings[i].pop(k)
        return hash_ratings

    def recommendation_accuracy(self,artist_index):
        '''
        input: a dictionary of recommended user to recommended artist index
        output: the percentages of users whose top 50 percentile artist is within
        the recommended artists
        '''
        matches = {}
        match_count = 0.0
        for key in artist_index:
            print key
            key_matches = list(set(self.test_set[key]) & set(artist_index[key]))
            if key_matches:
                matches[key] = key_matches
                match_count+=1
        return match_count/len(artist_index), matches

    def recommend_random(self,user_id,n_artist,top):

        '''
        sample with replacement (n_artist * top) random artist from the
        artist list to each test users
        '''
        for uid in user_id:
            n = n_artist*top
            random_key = list(set(choice(self.artist_id_index.keys(),n)))
            random_artist = [self.artist_id_index[key] for key in random_key]
            self.baseline_artist[uid] , self.baseline_index[uid] = random_artist,random_key
        return self.baseline_index

    def score(self,id_col='user_id',ratings=None,personality=None,top=5,n_artist=5,leave_out=4):
        '''
        Compute the probability of recommending user's top artist by taking away
        n artist in the user's top 50 percentile rating before making recommendation.
        Counts as success if any of the left out artists exist in the recommended set

        input:
        user_id: string: the name of the column that contains users' ids
        ratings: a pandas dataframe with columns representing the
        user's rating for an artist.
        personality: (optional) a pandas dataframe with columns representing each attribute in big 5, row index representing the user
        top: (int) the number of most similar users to choose recommendations from
        n_artist (int): the number of top artists from each most similar users
        to recommend

        output:
        p: the overall probability of recommending a user's favored artist
        '''

        userid = ratings[id_col]
        ratings = np.array(ratings.drop(id_col,axis=1))
        test_hash_ratings = get_user_hashmap(ratings)
        leave_n_out_ratings = self._take_n_out(userid,test_hash_ratings,leave_out)

        if self.method in ['personality','hybrid']:
            personality = np.array(personality)
            if self.method == 'hybrid':
                personality_dist = self._personality_dist(personality)
                test_rating_distance=self._get_test_rating_distance(leave_n_out_ratings)
                overall_dist = self.weight*personality_dist + (1-self.weight)*test_rating_distance
            if self.method == 'personality':
                overall_dist = self._personality_dist(personality)
        elif self.method == 'rating':
            overall_dist = self._get_test_rating_distance(leave_n_out_ratings)

        recommended_artists_ind=self._distance_based_rec(userid,overall_dist,top,n_artist)

        random_artist_ind=self.recommend_random(userid,top,n_artist)

        self.score, self.model_matches = self.recommendation_accuracy(recommended_artists_ind)
        self.basescore, self.basematches = self.recommendation_accuracy(random_artist_ind)

        print 'the baseline model scored {}, the {} recommender scored {}'.format(self.basescore,self.method,self.score)
