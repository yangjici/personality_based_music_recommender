## Summary

For the capstone project at Galvanize Data Science Program, I built an artist recommender from scratch that incoporates user's personality as part of its collaborative filtering (CF) algorithm. I also implemented artist rating based CF as well as a hybrid algorithm that uses both user's ratings and personality.

## Motivation

Have you ever encountered a song that you identify with so much that, in time, it has become part of your identity? 

I have. 

I have noticed that we choose to listen to music that reflect our quintenssence, guide our decisions, and shape us to be who we are.

With this idealistic notion in mind, I have wondered whether similar minded person would enjoy similar artists and songs. And, more importantly, whether knowing a person's personality would help us make better music and artist recommendations.

All recommender engine feel the pain of dealing with the problem of coldstart. In the case of user rating based CF,it is difficult to match a new user with other similar users with good confidence. However, knowing the personality of the new user we can match him/her to other users quickly and start making informed recommendations as soon as user personality information is obtained.

Obtaining user personality might sound labor intensive but in fact it is as quick as [applying magic sauce](https://applymagicsauce.com/) -  a profile based personality mining interface that allows for quick inference. Imagine quickly plug a new user in a community of like-minded person and make good recommendation! No more cold start! 

How did my personality based recommendation perform? keep reading and find out!

## My Data

I reached out to the admins at a mypersonality.org, a well-known psychometrics databased used by numerous psychology researchers. From there I obtained 2000 users with both big 5 personality test results and user name to lastfm.

Big 5 is currently the most well known and accepted personality factor model. The scale of the underlying traits: openness, conscientiousness, extroversion, agreeableness, neuroticism, ranges from 1 to 7.
More information on big 5 can be found [here](https://www.123test.com/big-five-personality-theory/) 

Lastfm compiles and record user's listen history from numerous popular sources such as spotify, and itunes. From [Lastfm API](http://www.last.fm/api), I was able to retrieve the entire listening history for each of the users. The listening history include the play counts of songs and their artists. Overall, there are approximately 1.5 million unique songs and 40 thousand unique artists for the users I retrieved. 

The user-rating matrix for individual songs is ultra sparse (~0.05%) The user-rating for artistis is less sparse but is still only 0.7% populated. 20% of user-pairs have no common artists and 36% of them have 5 or less common artists.

Sparsity of ratings is mainly a result of lack of users in our dataset, as it is uncommon to gather joint information about user's listening history and their personality. However, with a challenge comes an opportunity: the issue at hand very much resembles a case of cold start. Will personality of the users come to rescue? Stay and find out!

## Methodology

We need to estimate the implicit rating of user to a given artist based on the play count information. First we normalize the play count for each artist by total play count. Then we represented a user's rating to an artist on a percentile function from 1 to 4 by each user. Lastly, we subtract each user's rating by their own mean to establish relative baseline for each user. 

In my recommender, I implemented user-based neighborhood model of collaborative filtering. User based CF predicts an unrated item user A based on ratings of other users who have also rated that item. Specifically, it checksthe rating of top n most similar users to user A for that item.

Conventionally, user similarity is determined before recommendation by calculating a correlation coefficient between users using the user's ratings on common items. In my approach, I implemented Pearson's product-moment correlation coefficient, a custom, adjusted pearson's correlation (to be discussed later), Spearman's rank correlation.

Pearson's correlation is calculated on the common set of items ratings between the users. However, it lack confidence when the sample is small. In order to penalize similarity scores that are based on small number of overlapping items, which reflect a lack of confidence and prior disposition of users not sharing similar music taste, adjusted pearson's coefficient sets a control requirement of items two users must have in common before penality is applied. Additionally, the Spearman's rank correlation is implemented to be more robust against outliers, account for lack of confidence from small samples in rating data. 

Since the rating data is sparse and user similiarity results may not be reliable, we can also determine user's similarity based on their big 5 personality score. The similarity between two users is calculated by pearson's correlation. The hope is that knowing user's personality similarity we can make up for the lack of accuracy from user's ratings. Lastly, I generated a hybrid similarity matrix combining user personality and rating based similarity with an weighted average. 

## Model Evaluation

Two baseline models have been created to compare with my models. The random model simply uniformly generate a random float number between 1 and 4 for each of the artists. The Beyonce model fills the rating matrix with the average rating of the most listened artist in my dataset. (Beyonce)

The user is split into ~1400 samples training set and ~300 samples testing set. Cross validation is done using leave 5 out method: in each run, 5 known artist ratings are taken out at random for users in the testing. The model will generates a prediction for these unknown ratings and calculate the prediction error in RMSE.

The random model yielded RMSE of 1.8 and Beyonce model has RMSE of 1.1. In my CF model, user similarity performed the best using Spearman's rank correlation for rating, with a RMSE of 0.9, and Personality based similarity model yielded a RMSE of 0.85. Lastly, the hybrid model generated a RMSE of 1.1. 

![alt text](https://github.com/yangjici/personality_based_music_recommender/blob/master/graphs/figure_1.png)

## Discussion

From result shown above, it seems that personality based user similarity performed just as well as rating based similarity under the condition of sparsity. However, knowing the user's personality have added benefit of immediately matching new users with other similar users.

What was surprising is that hybrid model performed worse than rating or personality based similarity alone. What I suspect is that both personality and rating has some good signals for different user pairs but combining it together obfuscated the true signal.

For the next part of the project, I will create a pipeline to fully automate the process of music recommendation by simply entering user's personality and their lastfm username.

Thanks for reading 
 



