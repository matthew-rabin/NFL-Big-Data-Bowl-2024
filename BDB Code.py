#------------------------------Import libraries-----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#--------------------------------PREP DATA---------------------------

#plays data
plays = pd.read_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\nfl-big-data-bowl-2024\\plays.csv')
    #drop columns that are not needed
plays = plays[['gameId', 'playId', 'defensiveTeam']]

#tackles data
tackles = pd.read_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\nfl-big-data-bowl-2024\\tackles.csv')
    #drop columns that are not needed
tackles = tackles.drop(columns=['assist', 'forcedFumble', 'pff_missedTackle'])

#tracking data
tracking = pd.read_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\nfl-big-data-bowl-2024\\tracking_week_1.csv')
    #limit to event not null
tracking = tracking[tracking['event'].notnull()]

#combine plays and tracking data
    #split ball tracking data from player tracking data
tracking_ball = tracking[tracking['nflId'].isnull()]
    #drop columns that are not needed
tracking_ball = tracking_ball[['gameId', 'playId', 'frameId', 'x', 'y']]
    #rename x and y for ball tracking
tracking_ball.rename(columns={'x': 'x_ball', 'y': 'y_ball'}, inplace=True)
    #remove ball tracking rows from player tracking data
tracking = tracking[tracking['nflId'].notnull()]
    #merge tracking data with plays data to limit to tracking data for only defensive players
tracking = pd.merge(plays,tracking,how='inner',left_on=['gameId','playId', 'defensiveTeam'],right_on=['gameId','playId', 'club'])
    #merge x_ball and y_ball as columns to the defensive player tracking dataframe
tracking = pd.merge(tracking,tracking_ball,how='inner',left_on=['gameId','playId', 'frameId'],right_on=['gameId','playId', 'frameId'])

#remove variables that are no longer needed
tracking = tracking.drop(columns=['playDirection', 'club'])
del tracking_ball, plays

#Add new variables
    #calculate difference in distance between player and ball for x and y coordinates
tracking['x_diff'] = tracking['x_ball'] - tracking['x']
tracking['y_diff'] = tracking['y_ball'] - tracking['y']

    #split tracking by +/- x_diff,y_diff
tracking_xypos = tracking.loc[(tracking['x_diff'] >= 0) & (tracking['y_diff'] >= 0)]
tracking_xpos_yneg = tracking.loc[(tracking['x_diff'] >= 0) & (tracking['y_diff'] < 0)]
tracking_xneg_ypos = tracking.loc[(tracking['x_diff'] < 0) & (tracking['y_diff'] >= 0)]
tracking_xyneg = tracking.loc[(tracking['x_diff'] < 0) & (tracking['y_diff'] < 0)]

    #calculate angle from the player towards the ball
tracking_xypos['angle_towards_ball'] = (np.degrees(np.arctan(tracking_xypos['x_diff']/tracking_xypos['y_diff'])))
tracking_xpos_yneg['angle_towards_ball'] = (180-abs(np.degrees(np.arctan(tracking_xpos_yneg['x_diff']/tracking_xpos_yneg['y_diff']))))
tracking_xneg_ypos['angle_towards_ball'] = (360-abs(np.degrees(np.arctan(tracking_xneg_ypos['x_diff']/tracking_xneg_ypos['y_diff']))))
tracking_xyneg['angle_towards_ball'] = (180+np.degrees(np.arctan(tracking_xyneg['x_diff']/tracking_xyneg['y_diff'])))

    #combine back to one DataFrame
tracking = pd.concat([tracking_xypos, tracking_xpos_yneg, tracking_xneg_ypos, tracking_xyneg])

    #remove DataFrames that are no longer needed
del tracking_xypos, tracking_xpos_yneg, tracking_xneg_ypos, tracking_xyneg

    #calculate distance from player to ball
tracking['dist_from_ball'] = np.sqrt((tracking['x_diff']**2)+(tracking['y_diff']**2))
  
      #difference of player orientation from angle_towards_ball
tracking['o_diff_from_towards_ball'] =  abs(tracking['angle_towards_ball'] - tracking['o'])

       #difference of player orientation from angle_towards_ball
tracking['dir_diff_from_towards_ball'] =  abs(tracking['angle_towards_ball'] - tracking['dir'])


#combine tracking data and tackle (outcome) data
tracking = pd.merge(tracking,tackles,how='left',left_on=['gameId','playId', 'nflId'],right_on=['gameId','playId', 'nflId'])
tracking = tracking.fillna({'tackle': 0})

    #remove DataFrames that are no longer needed
del tackles

#-------------------------------MODEL DATA---------------------------

    #array of features
X = tracking[['s', 'a', 'dist_from_ball', 'o_diff_from_towards_ball', 'dir_diff_from_towards_ball']].values
    #array for outcome variable
y = tracking["tackle"].values

    #split into train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    #Evaluate classification models
models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(),'Decision Tree': DecisionTreeClassifier()}
results = []

    #Visualize results
for model in models.values():
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
  
    #Test performance
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))
    #logistic regression model has the best results
    
    #select logistic regression from the three classification models. Create the model.
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test)

X_scaled = scaler.fit_transform(X)

#----------------Add column for tackle probability to tracking data
tackle_probability = logreg.predict_proba(X_scaled)[:, 1]
tracking['tackle_probability'] = tackle_probability

    #remove DataFrames that are no longer needed
del cv_results, kf, logreg, model, models, name, results, scaler, tackle_probability, test_score, X, X_scaled, X_test, X_test_scaled, X_train, X_train_scaled, y, y_pred, y_test, y_train

    #generate CSV    
tracking.to_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\nfl-big-data-bowl-2024\\output\\tracking.csv')