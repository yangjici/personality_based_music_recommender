import numpy as np
import pandas as pd


'''
user listen count data
'''

listen_count = pd.read_csv('../data/user_top_songs.csv')

#a user may listen to a song with same name, must use mbid or combo of name and artist

#save it for tmr

user_song_counts=listen_count.groupby(['user_id','name','artist']).agg({'playcount': 'sum'})


weighted=user_song_counts.groupby(level=0).apply(lambda x: x/float(x.sum()))

weighted.reset_index(level=0,inplace=True)
