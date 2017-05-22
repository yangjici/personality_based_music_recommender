import numpy as np
import pandas as pd
import uuid
from itertools import izip
import scipy.sparse


'''
user listen count data
'''

listen_count = pd.read_csv('../data/user_top_songs.csv')
# generate a uuid for each song
unique_song_artist = pd.read_csv('data/unique_song_artist.csv')
unique_song_artist['track_id']=[uuid.uuid4() for i in xrange(len(unique_song_artist))]
unique_song_artist.to_csv('data/unique_song_artist.csv',encoding='utf-8-sig')

#generate a dictionary of song's unique_id with it's track - artist pair for indentification purpose'''

track_to_unique_id = { track_artist: uid    for track_artist,uid in izip(izip(unique_song_artist.name,unique_song_artist.artist),unique_song_artist.track_id)}
'''to access the unique id
track_to_unique_id[('Strange Vine','Delta Spirit')]'''

listen_count['track_id'] = [track_to_unique_id[(name,artist)] for name,artist in izip(listen_count.name,listen_count.artist)]
listen_count.to_csv('data/user_top_songs.csv',encoding='utf-8-sig')
#create an aggregate user to artist listening count
listen_count_by_artist = listen_count.groupby(['user_id','artist'],as_index=False).agg({'playcount': 'sum'})
''' sparse df with all artists in the columns and all users as index  and play count as value'''
all_artist_count_per_user=listen_count_by_artist.pivot(index='user_id',columns='artist',values='playcount')
''' fill the nas with zero'''
all_artist_count_per_user.fillna(0,inplace=True)
'''convert to sparse matrix'''
sparse_listen_count_mat = scipy.sparse.coo_matrix(all_artist_count_per_user.as_matrix())
'''get number of zeroes'''
sparse_listen_count_mat.getnnz()
''' 0.2 percent populated
too sparse, lets eliminate some obscure and not listened to artist'''
'''find artist play total count '''
artist_total_count=listen_count_by_artist.groupby('artist',as_index=False).agg({'playcount':'sum'})
'''check artists with more than 20 total playcounts from our users: from 146562 to 53913 artists'''
artist_20_more=artist_total_count[artist_total_count['playcount']>20]
'''get these play count per user for these artists only, bonus: artist total listen counts and listen counts by each user'''
user_artist_count_abridged=listen_count_by_artist.merge(artist_20_more,on='artist')
user_artist_count_abridged.columns=['user_id', 'artist', 'playcount', 'playcount_total']
user_artist_count_abridged.to_csv('data/artists_per_user_reduced.csv',encoding='utf-8-sig')
''' check matrix sparsity of the new user-artist matrix '''
all_artist_count_per_user_reduced=user_artist_count_abridged.pivot(index='user_id',columns='artist',values='playcount')
all_artist_count_per_user_reduced.fillna(0,inplace=True)
sparse_listen_count_mat_reduced = scipy.sparse.coo_matrix(all_artist_count_per_user_reduced.as_matrix())
sparse_listen_count_mat_reduced.getnnz()

''' 0.7 percent populated '''
'''lets get user's implicit rating'''
'''import the listen count of artists per user reduced version from data_wrangling'''
user_artist_count_abridged = pd.read_csv('data/artists_per_user_reduced.csv')
'''normalize user's playcount per artists by the user's total plays'''
user_song_counts=user_artist_count_abridged.groupby(['user_id','artist']).agg({'playcount': 'sum'})
weighted_implicit_rating=user_song_counts.groupby(level=0).apply(lambda x: x/float(x.sum()))
weighted_implicit_rating.reset_index(inplace=True)
'''create a implicit rating system
1. rank each artist per user

'''
weighted_implicit_rating['rank']=weighted_implicit_rating.groupby('user_id')['playcount'].rank(method='first',ascending=False)
weighted_implicit_rating=weighted_implicit_rating.groupby('user_id').apply(pd.DataFrame.sort_values, 'rank')
'''
find rating on sorted rank dataframe per each user based on percentile of listen counts
'''
def from_rank_to_rating(df):
    rank = df['rank']
    perc_freq = df['playcount']
    ratings = []
    for i in range(len(rank)):
        rating = 4*(1- sum(perc_freq[:i]))
        ratings.append(rating)
    return ratings
reshape_this =weighted_implicit_rating.groupby('user_id').apply(lambda x: from_rank_to_rating(x))
reshape_this=reshape_this.reset_index()
row_as_rating_for_top_song = pd.DataFrame(reshape_this[0].tolist(),).T.reset_index()
rating_as_list= pd.melt(row_as_rating_for_top_song,id_vars='index',value_name='rating')['rating']
'''finally, implicit rating'''
weighted_implicit_rating['rating']= [x for x in rating_as_list if x==x]
weighted_implicit_rating.to_csv('data/user_artist_rating.csv',encoding='utf-8-sig')
''' match users to their personality type'''
big5 = pd.read_csv('data/source_data/big5.csv')
user_big5= big5[big5['userid'].isin (weighted_implicit_rating['user_id'].unique())]  [['userid','ope','con','ext','agr','neu']]
user_big5.to_csv('data/user_big5.csv')
