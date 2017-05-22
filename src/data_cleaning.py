import pandas as pd

personality = pd.read_csv('data/source_data/big5.csv')
music_pref = pd.read_csv('data/source_data/mpq.csv')
music_attr = pd.read_csv('data/source_data/attribute_rating.csv')
main_data = pd.read_csv('data/source_data/data-mypersonality_main.csv')
#
# merger = pd.read_csv('data/source_data/merge_this.csv')
#
# main_data.drop(['Valence_loading_mixed_fb','Arousal_loading_mixed_fb','Depth_loading_mixed_fb'],axis=1,inplace=True)
#
#
# main_data = pd.merge(main_data,merger,on='userid')
#
#TBD whether to use this

#dropping the second test taken by the same same user who takes the test more than once'''
music_pref.drop_duplicates(subset='userid',keep='first',inplace=True)
main_data.drop_duplicates(subset='userid',keep='first',inplace=True)
#merge personality info with music preferences inner join on userid'''
personality_music = pd.merge(personality,music_pref,on='userid')
#find the index of the attributes
start1=int(np.where(main_data.columns.values=='INSTRU_mixed')[0])
end1=int(np.where(main_data.columns.values=='VOC_jazz')[0])
attributes_to_include = main_data.columns.values[start1:end1]
start2=int(np.where(main_data.columns.values=='zMellow_loading_mixed_fb')[0])
end2 = start2+5
more_attributes=np.append(attributes_to_include, main_data.columns.values[start2:end2])
all_attributes = np.append(more_attributes,['Arousal_38_mixed','Valence_38','Depth_38','userid'])
personality_music_attr = pd.merge(personality_music,main_data[all_attributes],on='userid')
personality_music_attr.to_csv('personality_music_attribute.csv')
