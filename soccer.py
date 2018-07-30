import sqlite3
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def get_player_stats(query):
    '''
    INPUT: SQL Query
    OUTPUT: Dataframe of Player Statistics
    '''

    database = 'database.sqlite'
    conn = sqlite3.connect(database)
    player_stats = pd.read_sql(query,conn)

    return player_stats

def get_player_data(query):
    '''
    INPUT: SQL Query
    OUTPUT: Dataframe of player meta data
    '''
    database = 'database.sqlite'
    conn = sqlite3.connect(database)
    player_meta_data = pd.read_sql(query, conn)

    return player_meta_data

def get_player_names(df1,df2):
    '''
    INPUT: Player meta data and player statistics df
    OUTPUT: List of player names
    '''
    df1 = df1.reset_index()
    mergeddf = df1.merge(df2, on = 'player_api_id')
    names = mergeddf['player_name']

    return names

def prep_player_data(df):
    '''
    INPUT: Dataframe of player statistics
    OUTPUT: Cleaned Dataframe with dummy variables
    '''
    work_rate_dict = {'low': 0, 'medium': 1, 'high': 2}
    pref_foot_dict = {'left': 0, 'right': 1, 'None': 2}

    df = df.loc[(df['attacking_work_rate'].isin(work_rate_dict.keys())) &
                (df['defensive_work_rate'].isin(work_rate_dict.keys()))].copy()

    df.loc[:, 'preferred_foot'] = df.loc[:, 'preferred_foot'].map(pref_foot_dict)
    df.loc[:, 'attacking_work_rate'] = df.loc[:, 'attacking_work_rate'].map(work_rate_dict)
    df.loc[:, 'defensive_work_rate'] = df.loc[:, 'defensive_work_rate'].map(work_rate_dict)

    columns = ['potential', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate',
              'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
              'dribbling', 'curve', 'free_kick_accuracy', 'long_passing', 'ball_control',
              'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
              'jumping', 'stamina', 'strenght', 'long_shots', 'aggression', 'interceptions',
              'positioning', 'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
              'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes', 'player_api_id']
    df = df.loc[:, player_stats.columns.isin(columns)]

    return df

def mean_player_data(df):
    '''
    INPUT: Cleaned Dataframe
    OUTPUT: Aggregated Dataframe (mean)
    '''
    return df.groupby(['player_api_id']).mean()

def standardize_data(df):
    '''
    INPUT: Dataframe
    OUTPUT: Standardized Dataframe
    '''
    df_std = df.drop(['preferred_foot','attacking_work_rate','defensive_work_rate'], axis = 1)
    df_dummies = df[['preferred_foot','attacking_work_rate','defensive_work_rate']].values
    scaler = StandardScaler()
    scaler.fit(df_std)
    new_x = scaler.transform(df_std)
    final_data = np.concatenate((new_x,df_dummies), axis = 1)

    return final_data

def scree_plot(pca):
    '''
    INPUT: PCA model
    OUTPUT: Scree Plot of PCA model
    '''

    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    cum_var = np.cumsum(vals)
    ax = plt.subplot(111)

    ax.plot(range(len(vals) + 1), np.insert(cum_var, 0, 0), color = 'r', marker = 'o')
    ax.bar(range(len(vals)), vals, alpha = 0.8)

    ax.axhline(0.9, color = 'g', linestyle = "--")
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    plt.title("Scree Plot for Soccer Data", fontsize=16)


def PCA_model(n_components, X):
    '''
    INPUT: Number of principal components, standardized data
    OUTPUT: pca model and transformed data
    '''
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    return pca, X_pca

def PCA_components(X, pca, cols):
    '''
    INPUT: Dataframe, pca model, column labels
    OUTPUT: Dataframe to interpret principal components
    '''
    X_pca = pca.fit_transform(X)
    comps = pca.components_
    df_pca = pd.DataFrame(comps, columns= cols)
    df_pca = df_pca.abs()
    nlargest = 5
    order = np.argsort(-df_pca.values, axis=1)[:, :nlargest]
    result = pd.DataFrame(df_pca.columns[order], columns=['top{}'.format(i) for i in range(1, nlargest+1)])

    return result

def silhouette_charts(X, k_list):
    '''
    INPUT: Dataframe, List of k values
    OUTPUT: Silhouette charts for each value of k
    '''
    for n_clusters in k_list:
    # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

def k_means_model(n_clusters, X):
    '''
    INPUT: number of clusters, pca transformed data
    OUTPUT: k-means clustering model, labels, and centers
    '''
    kmeans = KMeans(n_clusters).fit(X)
    centers = kmeans.cluster_centers_
    k_labels = kmeans.labels_

    return kmeans, centers, k_labels

def name_clusters(names, k_labels):
    '''
    INPUT: List of names, list of cluster labels
    OUTPUT: Dataframe of player names and assigned cluster
    '''
    names_clusters = pd.DataFrame({'Player': names,'Cluster': k_labels})

    return names_clusters

def K_nearest(player, nameslist, cluster_list, X):
    '''
    INPUT: Player name, List of names, list of clusters, Dataframe
    OUTPUT: Five closest neighbors and indices
    '''
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    idx = np.where(nameslist == player)[0][0]
    dist = distances[idx][1:]
    ind = indices[idx][1:]
    five_names = np.array(nameslist[ind])

    return dist, five_names, ind, indices

def five_nearest_radar(player, nameslist, fivenearest, indices, cols, X):
    '''
    INPUT: Player name, nameslist, list of five nearest n_neighbors
    indices, column labels, Dataframe
    OUTPUT: Radar chart of selected player and 5 nearest neighbors
    '''
    idx = np.where(nameslist == player)[0][0]
    ind = indices[idx]
    cols = cols
    feats = np.array(cols)
    stats = X.loc[ind[0],feats].values
    stats1 = X.loc[ind[1],feats].values
    stats2 = X.loc[ind[2],feats].values
    stats3 = X.loc[ind[3],feats].values
    stats4 = X.loc[ind[4],feats].values
    stats5 = X.loc[ind[5],feats].values


    angles=np.linspace(0, 2*np.pi, len(feats), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    stats1 = np.concatenate((stats1,[stats1[0]]))
    stats2 = np.concatenate((stats2,[stats2[0]]))
    stats3 = np.concatenate((stats3,[stats3[0]]))
    stats4 = np.concatenate((stats4,[stats4[0]]))
    stats5 = np.concatenate((stats5,[stats5[0]]))


    fig=plt.figure(figsize = (18,10))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2, label = player)
    ax.fill(angles, stats, alpha=0.25)
    ax.plot(angles,stats1,'o-', linewidth=2, label = fivenearest[0])
    ax.fill(angles,stats1, alpha= 0.25)
    ax.plot(angles,stats2,'o-', linewidth=2, label = fivenearest[1])
    ax.fill(angles,stats2, alpha= 0.25)
    ax.plot(angles,stats3,'o-', linewidth=2, label = fivenearest[2])
    ax.fill(angles,stats3, alpha= 0.25)
    ax.plot(angles,stats4,'o-', linewidth=2, label = fivenearest[3])
    ax.fill(angles,stats4, alpha= 0.25)
    ax.plot(angles,stats5,'o-', linewidth=2, label = fivenearest[4])
    ax.fill(angles,stats5, alpha= 0.25)
    ax.set_thetagrids(angles * 180/np.pi, feats)
    ax.set_title('Top Five Most Similar to' + ' ' + player)
    ax.legend()
    ax.grid(True)

if __name__ == '__main__':

    # Data Cleaning/Standardization
    player_stats = get_player_stats("SELECT * FROM Player_Attributes;")
    player_data = get_player_data("SELECT * FROM Player;")
    clean_df = prep_player_data(player_stats)
    mean_player_df = mean_player_data(clean_df)
    names = get_player_names(mean_player_df,player_data)
    std_data = standardize_data(mean_player_df)

    # EDA


    # PCA
    pca, X_pca = PCA_model(11,std_data)
    scree_plot(pca)
    pca_comps = PCA_components(std_data, pca, mean_player_df.columns)

    #K-Means
    # silhouette_charts(X_pca,range(2,12))
    kmeans, centers, k_labels = k_means_model(6,X_pca)
    name_clusters = name_clusters(names, k_labels)

    # KNN
    dist, five_names, ind, indices = K_nearest('Jerome Boateng', names, name_clusters, X_pca)
    mean_player_df = mean_player_df.reset_index()
    df_mean = mean_player_df.drop(['player_api_id'], axis = 1)
    five_nearest_radar('Jerome Boateng', names, five_names, indices, df_mean.columns, df_mean)
