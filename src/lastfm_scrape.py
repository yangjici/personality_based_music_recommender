from bs4 import BeautifulSoup
from PIL import Image
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


df = pd.read_csv('personality_music_attribute.csv')

'''get a of list of non-nan user names

The usual way to test for a NaN is to see if it's equal to itself

'''



user_names=[re.sub('http://www.last.fm/user/','',x)  for x in df.last_fm_username if x==x]


user_dict = {name:userid for name,userid in zip(user_names,df.userid) if name==name}




api_key = os.environ['LASTFM_API_KEY']

url_root ='http://ws.audioscrobbler.com/2.0/'


#get top track



user = 'zoot_money'

method = 'user.gettoptracks'

limit = '420'


pay_load = {'api_key': api_key,'user':user,'method':method,'format':'json','limit':limit}

response = requests.get(url_root,pay_load)

j = response.json()

#cleaning up json file

attempt=json.dumps(j['toptracks']['track'])


js_df = pd.read_json(attempt)[['name','playcount','artist','mbid']]

#get user_id match

user_id = user_dict[user]

#concatentate the user_id to the DataFrame

js_df['user_id']=user_id

js_df['artist'] = [d['name'] for d in js_df['artist']  ]











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
    api_key = os.environ['LASTFM_API_KEY']
    limit = '2000'
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

all_result['artist']=[d['name'] for d in all_result['artist']]

all_result.to_csv('user_top_songs.csv',encoding='utf-8-sig')




#crawling for song tags




artist = "Richard O'Brien"

track = 'Over At The Frankenstein Place'

method = 'track.gettoptags'

pay_load = {'api_key': api_key,'artist':artist,'track':track,'method':method,'format':'json'}

response = requests.get(url_root,pay_load)

j = response.json()

attempt=json.dumps(j['toptags']['tag'])

df_tag = pd.read_json(attempt).head(15)

load=pd.DataFrame({'name':track,'artist':artist,"tags":[list(df_tag['name'])]})








def single_song_tag_crawler(song,artist):
    url_root ='http://ws.audioscrobbler.com/2.0/'
    empty=pd.DataFrame({'name':song,'artist':artist,"tags":[[]]})
    api_key = os.environ['LASTFM_API_KEY']
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



# MAX_THREADS = 10
# all_tags= pd.DataFrame([])
#
# def single_song_tag_crawler_multi_threaded(song, artist):
#     url_root ='http://ws.audioscrobbler.com/2.0/'
#     empty=pd.DataFrame({'name':song,'artist':artist,"tags":[[]]})
#     api_key = 'b287e973174474be25ecf66adaf53c5b'
#     method = 'track.gettoptags'
#     pay_load = {'api_key': api_key,'artist':artist,'track':song,'method':method,'format':'json','autocorrect':'1'}
#     response = requests.get(url_root,pay_load)
#     j = response.json()
#     if 'error' in j:
#         return empty
#     attempt=json.dumps(j['toptags']['tag'])
#     df_tag = pd.read_json(attempt).head(15)
#     try:
#         tags =df_tag['name']
#     except KeyError:
#         return empty
#     load=pd.DataFrame({'name':song,'artist':artist,"tags":[list(tags)]})
#
#     all_tags = pd.concat([all_tages,load],ignore_index=True,axis=0)
#
#
#
# def lastfm_tag_crawler_multi_threaded(songs,artists):
#     jobs = []
#
#     for song,artist in izip(np.array(songs),np.array(artists)):
#         if len(jobs) >= MAX_THREADS:    # wait for threads to complete and join them back into the main thread
#             t = jobs.pop(0)
#             t.join()
#         t = threading.Thread(target=single_song_tag_crawler_multi_threaded, args=(song, artist,))
#         jobs.append(t)
#         t.start()
#
#     for t in jobs:
#         t.join()
#
#     return all_tags
#
#
# lastfm_tag_crawler_multi_threaded(all_result.name,all_result.artist)


# #get loved toptracks
#
#
# method1 = 'user.getlovedtracks'
#
# pay_load1 = {'api_key': api_key,'user':user,'method':method1,'format':'json','limit':limit}
#
# response1 = requests.get(url_root,pay_load1)
#
# j1 = response1.json()
#
# attempt1=json.dumps(j1['lovedtracks']['track'])
#
#
# js_df1 = pd.read_json(attempt1)
