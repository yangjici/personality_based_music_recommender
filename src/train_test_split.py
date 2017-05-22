import numpy as np
import pandas as pd
import scipy.sparse
from itertools import izip
from personality_music_recommender import Personality_Rating_Hybrid
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/user_artist_rating.csv')

df.drop(['Unnamed: 0'],axis=1,inplace=True)



df.to_csv('../data/user_artist_rating.csv',encoding='utf-8-sig')


train_pivot=df.pivot(index='user_id',columns='artist',values='rating')

train_pivot.fillna(0,inplace=True)

train_pivot.reset_index(inplace=True)



''' train user personality similarity matrix, matching each row'''

user_big5 = pd.read_csv('../data/user_big5.csv')

user_big5= user_big5[user_big5['userid'].isin (train_pivot['user_id'].unique())]

user_big5=user_big5.sort('userid')

user_big5 = user_big5.reset_index()

'''all match'''

(user_big5.userid == train_pivot.user_id).all()

all_dat = pd.concat([user_big5,train_pivot],axis=1)


train,test = train_test_split(all_dat,test_size=0.2)

train.drop(['index','Unnamed: 0'],axis=1,inplace=True)

test.drop(['index','Unnamed: 0'],axis=1,inplace=True)


'''persist data'''


train_person = train[['userid','ope','con','ext','agr','neu']]

train_person.to_csv('train_person.csv',encoding='utf-8-sig')

train_ratings = train.drop(['userid','ope','con','ext','agr','neu'],axis=1)

train_ratings.to_csv('train_ratings.csv',encoding='utf-8-sig')


test_person= test[['userid','ope','con','ext','agr','neu']]

test_person.to_csv('test_person.csv',encoding='utf-8-sig')


test_ratings = test.drop(['userid','ope','con','ext','agr','neu'],axis=1)

test_ratings.to_csv('test_ratings.csv',encoding='utf-8-sig')
