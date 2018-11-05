from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
## Importing required libraries
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, decomposition
from scipy.spatial.distance import cdist
from matplotlib.ticker import MaxNLocator
from sklearn import metrics, decomposition
from sklearn.decomposition import PCA , FastICA,TruncatedSVD
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.colors import LogNorm
from scipy.stats import kurtosis
from sklearn import random_projection
from sklearn.random_projection import SparseRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
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

        plt.xlabel("1st Component")
        plt.ylabel("2nd Component")
        plt.show()

    def plot(self, ll, homogeneity, completeness, rand, bic, aic):

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, ll)
            plt.xlabel('Number of clusters')
            plt.ylabel('Log Probablility')
            plt.title('Mushrooms With EM (LVF) - Log Probability')
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
            plt.title('Mushrooms With EM (LVF)- Homogeneity Score')
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
            plt.title('Mushrooms With EM (LVF) - Completeness Score')
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
        kurtosis_s = []

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
                pd_data.to_csv("cluster_kmeans_shrooms.csv", index=False, index_label=False, header=False)
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
            kurtosis_s.append(kurtosis(labels))
        print (meandist)
        if self.gen_plot:
            #self.visualize()

            self.plot(meandist, homogeneity_scores, completeness_scores, rand_scores, silhouettes, kurtosis_s)
            self.visualize()
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
        ax.set_xlabel("1st Component")
        ax.set_ylabel("2nd Component")
        ax.set_zlabel("3rd Component")
        plt.show()

    def plot(self, meandist, homogeneity, completeness, rand, silhouettes, kurtosis):
            """
            Plot average distance from observations from the cluster centroid
            to use the Elbow Method to identify number of clusters to choose
            """
            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, meandist)
            plt.xlabel('Number of clusters')
            plt.ylabel('Average within cluster SSE')
            plt.title('Mushroom With K-Means (LVF) - Average within cluster SSE')
            plt.show()

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, homogeneity)
            plt.xlabel('Number of clusters')
            plt.ylabel('Homogeneity Score')
            plt.title('Mushroom With K-Means (LVF) - Homogeneity Score')

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, completeness)
            plt.xlabel('Number of clusters')
            plt.ylabel('Completeness Score')
            plt.title('Mushroom With K-Means (LVF) - Completeness Score')
            plt.show()

            ax = plt.figure().gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(self.clusters, kurtosis)
            plt.xlabel('Number of clusters')
            plt.ylabel('Kurtosis Score')
            plt.title('Mushroom With K-Means (LVF) - Kurtosis Score')
            plt.show()

df = pd.read_csv('mushrooms.csv', nrows = 8000)
#set y to our target
Y=df['class']

#encode labels
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


#Drop target from our test and train
X=df.drop(['class'], axis=1)
X_std = StandardScaler().fit_transform(X)
#set y to our target
#Y=df['class']

#set train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)


"""
Run Kmeans Clusters and Em clusters
"""
start = time()

pca = PCA(n_components = 3)
S_pca = pca.fit_transform(X)

ica = FastICA(n_components = 3)
S_ica = ica.fit_transform(X)

rpg = random_projection.GaussianRandomProjection(n_components = 3)
S_rpg = rpg.fit(X).transform(X)

spg = random_projection.SparseRandomProjection(n_components = 3)
S_spg = spg.fit(X).transform(X)


threshold = [0 , .01,.02,.03,.04,.05,.1,.20,.25,.30,.4,.5, .6,.7,.8,.9, 1,5, 10]

lvf = VarianceThreshold()
t_lvf = lvf.fit_transform(X)
kmeans = KMeansTestCluster(t_lvf,Y,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
#kmeans.run()
em = ExpectationMaximizationTestCluster(t_lvf,Y,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
#em.run()

components = range(1,22)

acc = []

def lowV():
    for thresh in threshold:
        lvf = VarianceThreshold(threshold = thresh)
        spD = lvf.fit_transform(X_train)
        model = LinearSVC()
        model.fit(spD,y_train)
        test = lvf.transform(X_test)
        acc.append(metrics.accuracy_score(model.predict(test), y_test))
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
        sp = PCA(n_components = comp)
        spD = sp.fit_transform(X_train)
        model = MLPClassifier()
        model.fit(spD,y_train)
        test = sp.transform(X_test)
        acc.append(metrics.accuracy_score(model.predict(test), y_test))
        # create the figure
    plt.figure()
    plt.suptitle("Accuracy of PCA on Mushrooms")
    plt.xlabel("Components")
    plt.ylabel("Accuracy")
    plt.xlim([0, 23])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(acc), color = "r")
    plt.plot(components, acc)

    plt.show()
rpgAcc()
def scatters():
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('p', 'e'),
                            ('red', 'green')):
            plt.scatter(S_spg[Y==lab, 0],
                        S_spg[Y==lab, 1],
                        label=lab,
                        c=col)
        plt.title("RP-S Scatter")
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='upper right' )
        plt.tight_layout()
        plt.show()
def runPCA():

    n_comp = 22
    pca = PCA(n_components = n_comp)
    pca.fit(X_train)
    pca_cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Plot the Information Gain graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca_cumsum, color = 'purple', marker = 'o', ms = 5, mfc = 'red')
    fig.suptitle('Mushrooms - Cummulative Information Gain', fontsize=18)
    plt.xlabel('Number of Components', fontsize=14)
    plt.ylabel('Cummulative Variance Ratio', fontsize=12)
    plt.grid(True)
    ax.set_xlim([0,25])
    ax.set_ylim([0.0,1.0])

    plt.show()

    print('k \t explained variance ratio')

    for k in range(1,22):
        print ("%d \t %s" % (k, '{0:.2f}%'.format(pca.explained_variance_ratio_[k-1] * 100)) )
def ClusterWithPCA():

    pca_mod = PCA(n_components = 3)
    train_data_transformed = pca_mod.fit_transform(X)

    kmeansPCA = KMeansTestCluster(train_data_transformed,Y,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeansPCA.run()
    emPCA = ExpectationMaximizationTestCluster(train_data_transformed,Y,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    emPCA.run()
def scatterPlots():

    plt.figure(figsize=(15, 60))

    # Project the data to 2D
    n_comp = 2
    pca_mod = PCA(n_components = 2)
    train_data_transformed = pca_mod.fit_transform(X_train)

    for i in range(16):

        # Set up the plot
        ax = plt.subplot(8, 2, i+1)

        # Set up k-means
        km = KMeans(n_clusters=i+1, init='k-means++')

        # Train the clustering algorithm
        clstrs = km.fit(train_data_transformed)

        # Find the center of each cluster, and the distances to each point
        centers = [(clstrs.cluster_centers_[j,0],clstrs.cluster_centers_[j,1]) for j in range(i+1)]
        dists = km.transform(train_data_transformed)

        # The scatterplot
        ax.scatter(train_data_transformed[:, 0], train_data_transformed[:, 1], edgecolors='black',color='blue',marker='o')
        plt.title('KMeans clustering on 2D data, k = %i' %(i+1))
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Component 2")
        # Add the cluster centers and a circle encompassing all points in the cluster
        for j in range(i+1):
            circle = plt.Circle(centers[j], np.max(dists[clstrs.labels_ == j,j]),edgecolor='black',fill=False)
            my_center = plt.Circle(centers[j], 0.05, color='red')
            ax.add_artist(circle)
            ax.add_artist(my_center)
            plt.show()
def guessAcc():

    n_comp = 2
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
    test_preds = np.zeros(1124)
    for i in range(1124):
        if test_probs_under_pos[i] > test_probs_under_neg[i]:
            test_preds[i] = 1

    # Report the accuracy
    correct, total = 0, 0
    for pred, label in zip(test_preds, y_test):
        if pred == label: correct += 1
        total += 1
    print('total: %3d  correct: %3d  accuracy: %3.2f' %(total, correct, 1.0*correct/total))
def runICA():
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)  # Get the estimated sources
    #A_ = ica.mixing_  # Get estimated mixing matrix

    plt.figure()

    models = [X, S_ ]
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

    kmeansPCA = KMeansTestCluster(S_,Y,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeansPCA.run()
    emPCA = ExpectationMaximizationTestCluster(S_,Y,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    emPCA.run()
def runRPG():
    trans = random_projection.GaussianRandomProjection(n_components = 3)
    X_new = trans.fit_transform(X)

    kmeansPCA = KMeansTestCluster(X_new,Y,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeansPCA.run()
    emPCA = ExpectationMaximizationTestCluster(X_new,Y,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    emPCA.run()
def runRPS():
    trans = random_projection.SparseRandomProjection(n_components = 3)
    X_new = trans.fit_transform(X)

    kmeansPCA = KMeansTestCluster(X_new,Y,clusters=range(1,10), plot = True, targetcluster = 3 , stats = True)
    kmeansPCA.run()
    emPCA = ExpectationMaximizationTestCluster(X_new,Y,clusters=range(1,31), plot=True, targetcluster=3, stats=True)
    emPCA.run()

end = time()
print("PCA ran in {:.1f} minutes".format((end - start)/60))
