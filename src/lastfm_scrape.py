import requests
import time
import os
import pandas as pd
import math
import json
import re
from collections import defaultdict
from itertools import izip
import threading
import numpy as np
from mult_thread_scaper import scraping

df = pd.read_csv('personality_music_attribute.csv')
#get a of list of non-nan user names
user_names=[re.sub('http://www.last.fm/user/','',x)  for x in df.last_fm_username if x==x]
user_dict = {name:userid for name,userid in zip(user_names,df.userid) if name==name}
api_key = os.environ['LASTFM_API_KEY']
url_root ='http://ws.audioscrobbler.com/2.0/'
#get top track

'''build a data frame with the following columns
user_id: user id from personality_dataset
song_name: top songs most listened to by the user
artist: artist of the song
playcount: the number of plays per each song
weightedcount: weighted counts per each song 0<count<=1
mbid: musicbrainid of the song '''

def single_user_crawler(user,userid):
    empty= pd.DataFrame([])
    url_root ='http://ws.audioscrobbler.com/2.0/'
    api_key = 'b287e973174474be25ecf66adaf53c5b'
    limit = '3000'
    method = 'user.gettoptracks'
    pay_load = {'api_key': api_key,'user':user,'method':method,'format':'json','limit':limit}
    response = requests.get(url_root,pay_load)
    j = response.json()
    #try here
    if 'error' in j:
        return empty
    attempt=json.dumps(j['toptracks']['track'])
    try:
        single_user = pd.read_json(attempt)[['name','playcount','artist','mbid']]
    except KeyError:
        return empty
    single_user['artist'] = [defaultdict(str,x)['name'] for x in single_user['artist']]
    user_id = user_dict[user]
    single_user['user_id']=user_id

    #weighted counts?
    return single_user

def lastfm_song_crawler(user_dict):
    all_user_counts= pd.DataFrame([])
    for name,userid in user_dict.iteritems():
        print name
        single_user = single_user_crawler(name,userid)
        all_user_counts = pd.concat([all_user_counts,single_user],ignore_index=True,axis=0)
    return all_user_counts

all_result=lastfm_song_crawler(user_dict)
all_result.to_csv('user_top_songs.csv',encoding='utf-8-sig')

#crawling for song tags

def single_song_tag_crawler(song,artist):
    url_root ='http://ws.audioscrobbler.com/2.0/'
    empty=pd.DataFrame({'name':song,'artist':artist,"tags":[[]]})
    api_key = 'b287e973174474be25ecf66adaf53c5b'
    method = 'track.gettoptags'
    pay_load = {'api_key': api_key,'artist':artist,'track':song,'method':method,'format':'json','autocorrect':'1'}
    response = requests.get(url_root,pay_load)
    j = response.json()
    if 'error' in j:
        return empty
    attempt=json.dumps(j['toptags']['tag'])
    df_tag = pd.read_json(attempt).head(15)
    try:
        tags =df_tag['name']
    except KeyError:
        return empty
    load=pd.DataFrame({'name':song,'artist':artist,"tags":[list(tags)]})
    return load

def lastfm_tag_crawler(songs,artists):
    all_tags= pd.DataFrame([])
    count = 0
    for song,artist in izip(np.array(songs),np.array(artists)):
        count += 1
        if count %100 ==0:
            print '.'
        single_song = single_song_tag_crawler(song,artist)
        all_tags = pd.concat([all_tags,single_song],ignore_index=True,axis=0)
    return all_tags

all_songs_tags = lastfm_tag_crawler(all_result.name,all_result.artist)
all_songs_tags.to_csv('all_song_tags.csv')
#testing multithread scaper'''
small_subset = pd.read_csv('../data/unique_song_artist.csv').head()
scaper = scraping()
scaper.lastfm_tag_crawler_multi_threaded(small_subset.name,small_subset.artist)
#start scraping ~1 million song tags '''
all_unique_songs = pd.read_csv('../data/unique_song_artist.csv')
scaper = scraping()
scaper.lastfm_tag_crawler_multi_threaded(all_unique_songs.name,all_unique_songs.artist)
