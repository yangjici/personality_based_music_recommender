import numpy as np
import pandas as pd
import scipy.sparse
from itertools import izip
from personality_music_recommender import Personality_Rating_Hybrid

'''
how to run the model
'''


train_ratings = pd.read_csv('../data/train_ratings.csv')
train_ratings.drop(['Unnamed: 0'],axis=1,inplace=True)

train_person = pd.read_csv('../data/train_person.csv')
train_person.drop(['Unnamed: 0'],axis=1,inplace=True)

test_rating = pd.read_csv('../data/test_ratings.csv')
test_rating.drop(['Unnamed: 0'],axis=1,inplace=True)

test_person = pd.read_csv('../data/test_person.csv')
test_person.drop(['Unnamed: 0'],axis=1,inplace=True)

big5 = ['ope','con','ext','agr','neu']

#testing personality based recommendation
rec = Personality_Rating_Hybrid()
rec.train(ratings=train_ratings,personality=train_person[big5],method='personality')

#rec.recommend(personality=test_person,id_col='userid')
rec.score(ratings=test_rating,personality=test_person[big5],id_col='user_id')


#testing rating based recommendation
rec1 = Personality_Rating_Hybrid(sim_option='spearman_sim')
rec1.train(ratings=train_ratings,method='rating')
# rec1.recommend(ratings=test_rating,id_col='user_id')
rec1.score(ratings=test_rating,id_col='user_id')

#testing hybrid recommendation model

rec2 = Personality_Rating_Hybrid()
rec2.train(ratings=train_ratings,personality =train_person[big5] , method='hybrid')
# rec2.recommend(ratings=test_rating,personality=test_person.drop('userid',axis=1) , id_col='user_id')
rec2.score(ratings=test_rating,personality=test_person.drop('userid',axis=1) , id_col='user_id')
