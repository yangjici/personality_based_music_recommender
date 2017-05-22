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

class scraping(object):

    def __init__(self,MAX_THREADS=10,method='track.gettoptags'):
        self.url_root = 'http://ws.audioscrobbler.com/2.0/'
        self.api_key = os.envrion['LASTFM_API_KEY']
        self.MAX_THREADS = 10
        self.method = method
        self.all_tags = pd.DataFrame([])
        self.all_user_top_artists = pd.DataFrame([])

    def single_song_tag_crawler_multi_threaded(self,song, artist):
        empty=pd.DataFrame({'name':song,'artist':artist,"tags":[[]]})
        method = 'track.gettoptags'
        pay_load = {'api_key': self.api_key,'artist':artist,'track':song,'method':self.method,'format':'json','autocorrect':'1'}
        response = requests.get(self.url_root,pay_load)
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

        self.all_tags = pd.concat([self.all_tags,load],ignore_index=True,axis=0)

    def lastfm_tag_crawler_multi_threaded(self,songs,artists):
        jobs = []
        index = 0
        for song,artist in izip(np.array(songs),np.array(artists)):
            index+=1
            if index %500 ==0:
                print index
            if len(jobs) >= self.MAX_THREADS:    # wait for threads to complete and join them back into the main thread
                t = jobs.pop(0)
                t.join()
            t = threading.Thread(target=self.single_song_tag_crawler_multi_threaded, args=(song, artist,))
            jobs.append(t)
            t.start()

        for t in jobs:
            t.join()

        return self.all_tags
