# spotify-skip-prediction

This is my ML analysis on the "Spotify Sequential Skip Prediction Challenge" by Spotify from [AICrowd](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge). 

**Challenge: Predict if users will skip or listen to the music they're streamed**

Datasets: "tf" and "log", two datasets that contain:
- 49 features (combined) about the song
- 1 target variable: skipped or not
- the identifier of the common data point in the two datasets is the "track id" 

Skills utilized: 
- K-means clustering
- unsupervised learning
- plotting in seaborn and matplotlib
- scikit-learn 

Summary: Clusters are not informative, meaning another kind of ML model will be better.
Outcome: Models with the highest accuracy were logistic regression (98.8%) and Random Forest (99.1%). 


Main takeaway from the project: Even if individual variables have almost 0 correlation with the output variable, if we have a **high number** of these input features, we can combine them to create a very strong model. 

Next steps: 
- KNN clustering
- train on entire dataset (1 million rows) 

-------------------------------------------------------------------------------------------------------------
Initial challenges with the dataset: 
1. Understanding how to merge the two datasets
   - **Problem:** "tf" has 50k rows and "log" has 167k rows. Upon merging them with "inner", the program created a new dataframe with 167k rows (as opposed to 50k). Yet, there were apparently no duplicates nor no null values.
   - **Solution:** "tf" stands for track features and contains features about each song ONLY. Meanwhile, "log" tells us about the user history. When we combine the two datasets, the values in "tf" get duplicated for every song that repeats, because track features are identical with every song, but user history is not. Hence why the final merged dataset has 167k rows 
2. Understanding skipped_1, skipped_2, and skipped_3
   - **Context:** skipped_1, skipped_2, and skipped_3 are three boolean values that tell us WHEN the song was skipped
   - **Problem:** they are actually part of the target variable, but in a hierarchal chronological order
   - **Solution:** Remove them from the dataset as we are only concerned with whether or not the song is skipped - not _when_
3. Encoding categorical features
   - **Problem:** User behavior features like "why user ended previous song" are objects that can take on multiple string values such as "skip arrow", "problem with track", etc. How do we make this a numerical feature for the algorithm to understand?
   - **Solution:** 2 types of encoding
   -  1. Label encoding: For each user behavior feature, map each of the values that it can possibly take on to a number. "skip arrow" => 1, "problem with track" => 2, etc. This way, "why user ended previous song" becomes a numberical feature that takes on values 1 through 9, as opposed to different strings
       2. One-hot encoding: For each of the values in each user behavior feature, make it its own independent boolean. So we add a new feature "skip arrow" that takes on 1 for True, 0 for False. Then we add a new feature "problem with track" that has 1 for True, 0 for False, etc.
    
Other lessons learned: 
1. Make sure to normalize the data with scikit-learn's MinMaxScaler. This increases accuracy tremendously 
2. Always check for duplicate values, especially with clustering algorithms 
