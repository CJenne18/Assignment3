## Importing required libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPClassifier
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import graphviz
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree._tree import TREE_LEAF
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, decomposition
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA, FastICA,TruncatedSVD
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.colors import LogNorm
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold

class ExpectationMaximizationTestCluster():
    def __init__(self, X, y, clusters, plot=False, targetcluster=3, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.targetcluster = targetcluster
        self.stats = stats

    def run(self):
        ll=[]
        homogeneity_scores=[]
        completeness_scores=[]
        rand_scores=[]
        silhouettes=[]
        bic=[]
        aic=[]
        model = GMM(covariance_type = 'diag')

        for k in self.clusters:
            model.set_params(n_components=k)
            model.fit(self.X)
            labels = model.predict(self.X)
            #print labels
            if k == self.targetcluster and self.stats:
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv(" cluster_em.csv", index=False, index_label=False, header=False)

                for i in range (0,self.targetcluster):
                    #print "Cluster {}".format(i)
                    cluster = pd_data.loc[pd_data.iloc[:,-2]==i].iloc[:,-2:]
                    print(cluster.shape[0])
                    #print float(cluster.loc[cluster.iloc[:,-1]==0].shape[0])/cluster.shape[0]
                    #print float(cluster.loc[cluster.iloc[:,-1]==1].shape[0])/cluster.shape[0]

            #meandist.append(sum(np.min(cdist(self.X, model.cluster_centers_, 'euclidean'), axis=1))/ self.X.shape[0])
            ll.append(model.score(self.X))
            print(model.score(self.X))
            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
            bic.append(model.bic(self.X))
            aic.append(model.aic(self.X))
            #silhouettes.append(metrics.silhouette_score(self.X, model.labels_ , metric='euclidean',sample_size=self.X.shape[0]))

        if self.gen_plot:
            #self.visualize()
            self.plot(ll, homogeneity_scores, completeness_scores, rand_scores, bic, aic)

    def visualize(self):
        """
        Generate scatter plot of Kmeans with Centroids shown
        """
        fig = plt.figure(1)
        plt.clf()
        plt.cla()

        X_new = decomposition.pca.PCA(n_components=2).fit_transform(self.X)
        model = GMM(n_components=self.targetcluster, covariance_type='full')
        labels = model.fit_predict(X_new)
        totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1),), axis=1)

        # for each cluster
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        fig = plt.figure()

        for clust in range(0, self.targetcluster):
            totz_clust = totz[totz[:,-2] == clust]
            print("Cluster Size")
            print(totz_clust.shape)

            benign = totz_clust[totz_clust[:,-1] == 1]
            malignant = totz_clust[totz_clust[:,-1] == 0]

            plt.scatter(benign[:, 0], benign[:, 1],  color=colors[clust], marker=".")
            plt.scatter(malignant[:, 0], malignant[:, 1],  color=colors[clust], marker="x")

            centroids = model.cluster_centers_
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                marker='x', s=169, linewidths=3, color="black",
                 zorder=10)
        #plt.title("Breast Cancer Clustering")
        plt.xlabel("1st Component")
        plt.ylabel("2nd Component")
        plt.show()

    def plot(self, ll, homogeneity, completeness, rand, bic, aic):
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, ll)
            plt.xlabel('Number of clusters')
            plt.ylabel('Log Probablility')
            plt.title('Match Predictor With EM (LVF) - Log Probability')
            plt.show()
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, homogeneity)
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title('Match Predictor With EM (LVF) - Homogeneity Score')
            plt.show()

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, completeness)
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title('Match Predictor With EM (LVF) - Completeness Score')
            plt.show()

class KMeansTestCluster():
    def __init__(self, X, y, clusters, plot=True, targetcluster=3, stats=False):
        self.X = X
        self.y = y
        self.clusters = clusters
        self.gen_plot = plot
        self.targetcluster = targetcluster
        self.stats = stats

    def run(self):
        meandist=[]
        homogeneity_scores=[]
        completeness_scores=[]
        v_measure = []
        rand_scores=[]
        silhouettes=[]

        for k in self.clusters:
            model = KMeans(n_clusters=k, max_iter=500, init='k-means++')
            labels = model.fit_predict(self.X)
            #print model.cluster_centers_
            #print cdist(self.X, model.cluster_centers_, 'euclidean')
            #print cdist(self.X, model.cluster_centers_, 'euclidean').shape
            if k == self.targetcluster and self.stats:
                #print labels
                nd_data = np.concatenate((self.X, np.expand_dims(labels, axis=1),np.expand_dims(self.y, axis=1)), axis=1)
                pd_data = pd.DataFrame(nd_data)
                pd_data.to_csv("Match Predictor-cluster_kmeans.csv", index=False, index_label=False, header=False)
                #print model.cluster_centers_
                print (cdist(self.X, model.cluster_centers_, 'euclidean').shape)

            #print np.min(np.square(cdist(self.X, model.cluster_centers_, 'euclidean')), axis = 1)
            min = np.min(np.square(cdist(self.X, model.cluster_centers_, 'euclidean')), axis = 1)
            print ("###")
            print (-model.score(self.X)/self.X.shape[0])
            #print min
            value = np.mean(min)
            meandist.append(value)

            homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            completeness_scores.append(metrics.completeness_score(self.y, labels))
            rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
        print (meandist)
        if self.gen_plot:
            #self.visualize()

            self.plot(meandist, homogeneity_scores, completeness_scores, rand_scores, silhouettes)

    def visualize(self):
        """
        Generate scatter plot of Kmeans with Centroids shown
        """
        fig = plt.figure(1)
        plt.clf()
        plt.cla()

        X_new = decomposition.pca.PCA(n_components=3).fit_transform(self.X)
        model = KMeans(n_clusters=self.targetcluster, max_iter=5000, init='k-means++')
        labels = model.fit_predict(X_new)
        totz = np.concatenate((X_new,  np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1),), axis=1)

        # for each cluster
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for clust in range(0, self.targetcluster):
            totz_clust = totz[totz[:,-2] == clust]
            print ("Cluster Size")
            print (totz_clust.shape)

            benign = totz_clust[totz_clust[:,-1] == 1]
            malignant = totz_clust[totz_clust[:,-1] == 0]

            ax.scatter(benign[:, 0], benign[:, 1], benign[:, 2], color=colors[clust], marker=".")
            ax.scatter(malignant[:, 0], malignant[:, 1], malignant[:, 2], color=colors[clust], marker="x")

        centroids = model.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3, color="black",
             zorder=10)

        # ax.title("Breast Cancer Clustering")
        ax.set_xlabel("1st Component")
        ax.set_ylabel("2nd Component")
        ax.set_zlabel("3rd Component")
        plt.show()

    def plot(self, meandist, homogeneity, completeness, rand, silhouettes):
            """
            Plot average distance from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, meandist)
            plt.xlabel('Number of clusters')
            plt.ylabel('Average within cluster SSE')
            plt.title('Match Outcome With K-Means(LVF) - Average within cluster SSE')
            plt.show()

            #plt.clf()

            """
            Plot homogeneity from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, homogeneity)
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title('Match Outcome With K-Means(LVF) - Homogeneity Score')
            plt.show()

            #plt.clf()

            """
                       Plot completeness from observations from the cluster centroid
                       to use the Elbow Method to identify number of clusters to choose
                       """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, completeness)
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title('Match Outcome With K-Means(LVF) - Completeness Score')
            plt.show()

## Loading all functions
def get_match_label(match):
    ''' Derives a label for a given match. '''

    #Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0,'match_api_id'] = match['match_api_id']

    #Identify match label
    if home_goals > away_goals:
        label.loc[0,'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0,'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0,'label'] = "Defeat"

    #Return label
    return label.loc[0]

def get_fifa_stats(match, player_stats):
    ''' Aggregates fifa stats for a given match. '''

    #Define variables
    match_id =  match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []

    #Loop through all players
    for player in players:

        #Get player ID
        player_id = match[player]

        #Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        #Identify current stats
        current_stats = stats[stats.date < date].sort_values(by = 'date', ascending = False)[:1]

        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])

        #Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)

        #Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)

    player_stats_new.columns = names
    player_stats_new['match_api_id'] = match_id

    player_stats_new.reset_index(inplace = True, drop = True)

    #Return player stats
    return player_stats_new.ix[0]

def get_fifa_data(matches, player_stats, path = None, data_exists = False):
    ''' Gets fifa data for all matches. '''

    #Check if fifa data already exists
    if data_exists == True:
        print("Fifa data exists..")
        fifa_data = pd.read_csv('fifa_data.csv')

    else:

        print("Collecting fifa data for each match...")
        start = time()

        #Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x :get_fifa_stats(x, player_stats), axis = 1)

        end = time()
        print("Fifa data collected in {:.1f} minutes".format((end - start)/60))

    #Return fifa_data

    fifa_data.to_csv('fifa_data.csv')
    return fifa_data

def get_overall_fifa_rankings(fifa, get_overall = False):
    ''' Get overall fifa rankings from fifa data. '''

    temp_data = fifa

    #Check if only overall player stats are desired
    if get_overall == True:

        #Get overall stats
        data = temp_data.loc[:,(fifa.columns.str.contains('overall_rating'))]
        data.loc[:,'match_api_id'] = temp_data.loc[:,'match_api_id']
    else:

        #Get all stats except for stat date
        cols = fifa.loc[:,(fifa.columns.str.contains('date_stat'))]
        temp_data = fifa.drop(cols.columns, axis = 1)
        data = temp_data

    #Return data
    return data

def get_last_matches(matches, date, team, x = 10):
    ''' Get the last x matches of a given team. '''

    #Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]

    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]

    #Return last matches
    return last_matches

def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    ''' Get the last x matches of two given teams. '''

    #Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])

    #Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = 'date', ascending = False).iloc[0:total_matches.shape[0],:]

        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")

    #Return data
    return last_matches

def get_goals(matches, team):
    ''' Get the goals of a specfic team from a set of matches. '''

    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

def get_shotson(matches,team):

    homeShot = int(matches.home_team)

def get_goals_conceided(matches, team):
    ''' Get the goals conceided of a specfic team from a set of matches. '''

    #Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())

    total_goals = home_goals + away_goals

    #Return total goals
    return total_goals

def get_wins(matches, team):
    ''' Get the number of wins of a specfic team from a set of matches. '''

    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())

    total_wins = home_wins + away_wins

    #Return total wins
    return total_wins

def get_match_features(match, matches, x = 10):
    ''' Create match specific features for a given match. '''

    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)

    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)

    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)


    #Define result data frame
    result = pd.DataFrame()

    #Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id

    #Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    #Return match features
    return result.loc[0]

def create_feables(matches, fifa, bookkeepers, get_overall = False, horizontal = True, x = 10, verbose = True, data_exists = False):
    ''' Create and aggregate features and labels for all matches. '''

    #Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)


    if verbose == True:
        print("Generating match features...")
    start = time()
    if data_exists == True:
        print("Match features data exists")
        match_stats = pd.read_csv("match_features.csv")
    else:
        #Get match features for all matches
        match_stats = matches.apply(lambda x: get_match_features(x, matches, x = 10), axis = 1)
        #Create dummies for league ID feature
        dummies = pd.get_dummies(match_stats['league_id']).rename(columns = lambda x: 'League_' + str(x))
        match_stats = pd.concat([match_stats, dummies], axis = 1)
        match_stats.drop(['league_id'], inplace = True, axis = 1)
        match_stats.to_csv("match_features.csv")
    end = time()

    if verbose == True:
        print("Match features generated in {:.1f} minutes".format((end - start)/60))

    if verbose == True:
        start = time()
        print("Generating match labels...")
        if data_exists == True:
            labels = pd.read_csv("labels.csv")
        else:
            #Create match labels
            labels = matches.apply(get_match_label, axis = 1)
            labels.to_csv('labels.csv')
    end = time()
    if verbose == True:
        print("Match labels generated in {:.1f} minutes".format((end - start)/60))

    if verbose == True:
        print("Generating bookkeeper data...")
    start = time()

    #Get bookkeeper quotas for all matches
#    bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal = True)
    #bk_data.loc[:,'match_api_id'] = matches.loc[:,'match_api_id']
    end = time()
    if verbose == True:
        print("Bookkeeper data generated in {:.1f} minutes".format((end - start)/60))

    #Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on = 'match_api_id', how = 'left')
    #features = pd.merge(features, bk_data, on = 'match_api_id', how = 'left')
    feables = pd.merge(features, labels, on = 'match_api_id', how = 'left')

    #print(feables)


    #Drop NA values
    feables.dropna(inplace = True)

    #Return preprocessed data
    return feables

def explore_data(features, inputs):
    ''' Explore data by plotting KDE graphs. '''
    #Compute and print label weights
    labels = inputs.loc[:,'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    #Store description of all features
    feature_details = features.describe().transpose()

    #Return feature details
    return feature_details

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

start = time()
conn = sqlite3.connect('database.sqlite')

#Defining the number of jobs to be run in parallel during grid search
n_jobs = 1 #Insert number of parallel jobs here

#Fetching required data tables
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)

#Reduce match data to fulfill run time requirements
rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

match_data.dropna(subset = rows, inplace=True)
#match_data = match_data.head(3000)

## Generating features, exploring the data, and preparing data for model training
#Generating or retrieving already existant FIFA data
fifa_data = get_fifa_data(match_data, player_stats_data, data_exists = True)

#Creating features and labels based on data provided
bk_cols = ['B365H', 'BWA', 'IWD', 'LBH', 'PSA', 'WHD', 'SJH', 'VCA', 'GBD', 'BSH']
bk_cols_selected = ['B365H', 'BWA']

feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall = True , data_exists = True)
labelencoder=LabelEncoder()
for column in feables.columns:
    feables[column] = labelencoder.fit_transform(feables[column])
inputs = feables.drop('match_api_id', axis = 1)

#Exploring the data and creating visualizations
labels = inputs.loc[:,'label']
features = inputs.drop('label', axis = 1)
feature_details = explore_data(features, inputs)

#Splitting the data into Train, Calibrate, and Test data sets
X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42,
                                                                        stratify = labels)
X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size = 0.3, random_state = 42,
                                                              stratify = y_train_calibrate)

kmeans = KMeansTestCluster(X_train, y_train, clusters=range(1,10), plot=True, targetcluster=3, stats=True)#kmeans.run()
#kmeans.run()
em = ExpectationMaximizationTestCluster(X_train,y_train,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
#em.run()

pca = PCA(n_components = 3)
S_pca = pca.fit_transform(features)

ica = FastICA(n_components = 3)
S_ica = ica.fit_transform(features)

rpg = random_projection.GaussianRandomProjection(n_components = 3 )
g_rpg = rpg.fit_transform(features)

spg = random_projection.SparseRandomProjection(n_components = 3)
s_rp = spg.fit_transform(features)


threshold = [.01,.02,.03,.04,.05,.1,.20,.25,.30,.4,.5, .6,.7,.8,.9, 1]

lvf = VarianceThreshold()
t_lvf = lvf.fit_transform(X_train)

components = range(1,31)
model = LinearSVC()
model.fit(X_train, y_train)
baseline = metrics.accuracy_score(model.predict(X_calibrate), y_calibrate)
acc = []

def lowV():
    for thresh in threshold:
        lvf = VarianceThreshold(threshold = thresh)
        spD = lvf.fit_transform(X_train)
        model = LinearSVC()
        model.fit(spD,y_train)
        test = lvf.transform(X_calibrate)
        acc.append(metrics.accuracy_score(model.predict(test), y_calibrate))
        # create the figure
    plt.figure()
    plt.suptitle("Accuracy of Variance Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.xlim([0, 1])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(threshold, [baseline] * len(acc), color = "r")
    plt.plot(threshold, acc)

    plt.show()
def rpgAcc():
    for comp in components:
        sp = FastICA(n_components = comp)
        spD = sp.fit_transform(X_train)
        model = LinearSVC()
        model.fit(spD,y_train)
        test = sp.transform(X_calibrate)
        acc.append(metrics.accuracy_score(model.predict(test), y_calibrate))
        # create the figure
    plt.figure()
    plt.suptitle("Accuracy of ICA on Match Prediction")
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    plt.xlim([0, 31])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(acc), color = "r")
    plt.plot(components, acc)

    plt.show()
rpgAcc()

def scatters():
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('2', '0' , '1'),
                            ('red', 'green' , 'blue')):
            plt.scatter(S_pca[labels==lab, 0],
                        S_pca[labels==lab, 1],
                        S_pca[labels==lab, 2],
                        label=lab,
                        c=col)
        plt.title("RP-S Scatter")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='upper right' )
        plt.tight_layout()
        plt.show()

def runPCA1():

    n_comp = 6
    pca = PCA(n_components = n_comp)
    pca.fit(X_train)
    pca_cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Plot the Information Gain graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca_cumsum, color = 'purple', marker = 'o', ms = 5, mfc = 'red')
    fig.suptitle('Soccer Match - Cummulative Information Gain', fontsize=18)
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Cummulative Variance Ratio', fontsize=12)
    plt.grid(True)
    ax.set_xlim([0,25])
    ax.set_ylim([0.0,1.0])

    plt.show()

    print('k \t explained variance ratio')

    for k in range(1,6):
        print ("%d \t %s" % (k, '{0:.2f}%'.format(pca.explained_variance_ratio_[k-1] * 100)) )

    pca_mod = PCA(n_components = 1)
    train_data_transformed = pca_mod.fit_transform(X_train)
    kmeans = KMeansTestCluster(train_data_transformed,y_train,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeans.run()
    em = ExpectationMaximizationTestCluster(train_data_transformed,y_train,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    em.run()
def runICA():
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X_train)  # Get the estimated sources
    A_ = ica.mixing_  # Get estimated mixing matrix

    plt.figure()

    models = [X_train, S_]
    names = ['Observations (mixed signal)',
            'ICA recovered signals']
    colors = ['red', 'steelblue']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()

    kmeansPCA = KMeansTestCluster(S_,y_train,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeansPCA.run()
    emPCA = ExpectationMaximizationTestCluster(S_,y_train,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    emPCA.run()
def P2():

    pca_mod = PCA(n_components = 2)
    train_data_transformed = pca_mod.fit_transform(X_train)

    kmeans = KMeansTestCluster(train_data_transformed,y_train,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeans.run()
    em = ExpectationMaximizationTestCluster(train_data_transformed,y_train,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    em.run()
def guessAcc():

    n_comp = 3
    pca_mod = PCA(n_components = n_comp)
    train_data_transformed = pca_mod.fit_transform(X_train)

    # Pull out positive and negative training data
    positive = train_data_transformed[y_train == 1, :]
    negative = train_data_transformed[y_train == 0, :]

    # Fit a positive model
    clf_pos = GaussianMixture(n_components=4, covariance_type='full')
    clf_pos.fit(positive)

    # Fit a negative model
    clf_neg = GaussianMixture(n_components=4, covariance_type='full')
    clf_neg.fit(negative)

    # Apply the PCA transformation to the test data:
    test_data_transformed = pca_mod.transform(X_test)

    # Predict the test data
    test_probs_under_pos = clf_pos.score_samples(test_data_transformed)

    # Predict the test data
    test_probs_under_neg = clf_neg.score_samples(test_data_transformed)

    # Predict positive by if logprob uder positive is greater
    test_preds = np.zeros(600)
    for i in range(600):
        if test_probs_under_pos[i] > test_probs_under_neg[i]:
            test_preds[i] = 1

    # Report the accuracy
    correct, total = 0, 0
    for pred, label in zip(test_preds, y_test):
        if pred == label: correct += 1
        total += 1
    print('total: %3d  correct: %3d  accuracy: %3.2f' %(total, correct, 1.0*correct/total))
