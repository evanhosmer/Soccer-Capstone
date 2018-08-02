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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def players_and_teams(df):
    '''
    INPUT: Dataframe of teams and countries
    OUTPUT: Dataframe with aggregated stats by league
    '''
    database = 'database.sqlite'
    conn = sqlite3.connect(database)
    query = """SELECT * FROM Player_Attributes a
           INNER JOIN (SELECT player_name, player_api_id AS p_id FROM Player) b ON a.player_api_id = b.p_id;"""
    player_stats_wname = pd.read_sql(query,conn)
    league_data = pd.read_sql("SELECT * FROM League;", conn)
    country_data = pd.read_sql("SELECT * FROM Country;",conn)
    player_mean_stats = player_stats_wname.groupby(['player_name']).mean()
    players_data_2 = player_mean_stats.reset_index()
    players_final = pd.merge(players_data_2,df, on = 'player_fifa_api_id')
    cols_drop = ['id_x','player_fifa_api_id','player_api_id_x','Unnamed: 0', 'id_y','player_api_id_y','player_name_y',
            'num_uniq_team']
    df_players = players_final.drop(cols_drop,axis = 1)
    league_country = pd.merge(league_data,country_data, on = 'id')
    league_country.rename(columns={'name_y':'country'}, inplace=True)
    final_players = pd.merge(df_players,league_country, on = 'country').drop(['p_id','id','country_id'], axis = 1)
    final_players.drop(['birthday','height','weight'], axis = 1, inplace = True)
    no_gks = final_players[final_players['gk_diving'] < 30].dropna()
    mean_league = no_gks.groupby(['name_x']).mean().drop(['gk_diving','gk_handling','gk_kicking',
    'gk_positioning','gk_reflexes'],axis = 1)
    leagues = mean_league.reset_index()['name_x']
    mean_leagues = mean_league.reset_index().drop(['name_x'], axis = 1)
    scaler = StandardScaler()
    scaler.fit(mean_leagues)
    std_data = scaler.transform(mean_leagues)
    cols = mean_league.columns

    return leagues, std_data, cols, player_mean_stats

def no_gks(df):
    '''
    INPUT: Dataframe
    OUTPUT: Dataframe with goalkeepers removed from the dataset
    '''

    no_gk = df[df['gk_diving'] < 30]
    no_gk_df = no_gk.drop(['gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes'], axis = 1)

    return no_gk_df

def data_plot(data):
    '''
    INPUT: PCA transformed data
    OUTPUT: Plot of data
    '''

    plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

def country_plots(leagues, std_data, cols):
    '''
    INPUT: Leagues df, aggegated league stats df, column names
    OUTPUT: Plots comparing leagues
    '''
    cols = np.insert(cols,0,'name_x')
    df = pd.DataFrame(std_data)
    df.columns = cols
    z_scores_name = pd.concat([leagues,df], axis = 1)
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(221)

    x = z_scores_name['finishing'].values
    y = z_scores_name['name_x'].values

    ax.barh(y, x, align='center',
            color='green', ecolor='black')
    ax.set_yticks(range(len(y)))
    ax.set_yticklabels(y)
    ax.invert_yaxis()
    ax.set_xlabel('Finishing (z-score)')
    ax.set_title('Which league has the best goal scorers?')

    ax = fig.add_subplot(222)

    x2 = z_scores_name['overall_rating'].values
    y2 = z_scores_name['name_x'].values

    ax.barh(y2, x2, align='center',
            color='green', ecolor='black')
    ax.set_yticks(range(len(y2)))
    ax.set_yticklabels(y2)
    ax.invert_yaxis()
    ax.set_xlabel('Overall Rating (z-score)')
    ax.set_title('Which league has the best players?')

    ax = fig.add_subplot(223)

    x3 = z_scores_name['dribbling'].values
    y3 = z_scores_name['name_x'].values

    ax.barh(y3, x3, align='center',
            color='green', ecolor='black')
    ax.set_yticks(range(len(y3)))
    ax.set_yticklabels(y3)
    ax.invert_yaxis()
    ax.set_xlabel('Dribbling rating (z-score)')
    ax.set_title('Which league has the best dribblers?')

    ax = fig.add_subplot(224)

    x4 = z_scores_name['sprint_speed'].values
    y4 = z_scores_name['name_x'].values

    ax.barh(y4, x4, align='center',
            color='green', ecolor='black')
    ax.set_yticks(range(len(y4)))
    ax.set_yticklabels(y4)
    ax.invert_yaxis()
    ax.set_xlabel('Sprint Speed (z-score)')
    ax.set_title('Which league has the fastest players?')

    plt.tight_layout()

def player_plots(df):
    '''
    INPUT: Dataframe
    OUTPUT: Player comparison charts
    '''
    overall = df.sort_values(by=['overall_rating'], ascending = False)
    speed = df.sort_values(by=['sprint_speed'], ascending = False)
    dribbling = df.sort_values(by=['dribbling'], ascending = False)
    ball_control = df.sort_values(by=['ball_control'], ascending = False)
    df['total_passing'] = (df['long_passing'] + df['short_passing'] + df['vision'] + df['crossing']) / 4
    df['total_defending'] = (df['standing_tackle'] + df['sliding_tackle'] + df['marking'] + df['interceptions']) / 4
    passing = df.sort_values(by=['total_passing'], ascending = False)
    defending = df.sort_values(by=['total_defending'], ascending = False)

    fig = plt.figure(figsize = (12,6))

    ax = fig.add_subplot(231)
    x = overall.iloc[0:10]['overall_rating'].values
    y = overall.iloc[0:10]['player_name_x'].values

    ax.barh(y, x, align='center',
            color='green', ecolor='black')
    ax.set_yticks(range(len(y)))
    ax.set_yticklabels(y)
    ax.set_xlim(75,100)
    ax.invert_yaxis()
    ax.set_xlabel('Overall Rating')
    ax.set_title('10 best players by overall rating')

    ax2 = fig.add_subplot(232)
    x2 = speed.iloc[0:10]['sprint_speed'].values
    y2 = speed.iloc[0:10]['player_name_x'].values

    ax2.barh(y2, x2, align='center',
            color='green', ecolor='black')
    ax2.set_yticks(range(len(y2)))
    ax2.set_yticklabels(y2)
    ax2.set_xlim(75,100)
    ax2.invert_yaxis()
    ax2.set_xlabel('Sprint Speed')
    ax2.set_title('10 Fastest Players')

    ax3 = fig.add_subplot(233)
    x3 = dribbling.iloc[0:10]['dribbling'].values
    y3 = dribbling.iloc[0:10]['player_name_x'].values

    ax3.barh(y3, x3, align='center',
            color='green', ecolor='black')
    ax3.set_yticks(range(len(y3)))
    ax3.set_yticklabels(y3)
    ax3.set_xlim(85,100)
    ax3.invert_yaxis()
    ax3.set_xlabel('Dribbling Rating')
    ax3.set_title('10 Best Dribblers')

    ax4 = fig.add_subplot(234)
    x4 = ball_control.iloc[0:10]['ball_control'].values
    y4 = ball_control.iloc[0:10]['player_name_x'].values

    ax4.barh(y4, x4, align='center',
            color='green', ecolor='black')
    ax4.set_yticks(range(len(y4)))
    ax4.set_yticklabels(y4)
    ax4.set_xlim(75,100)
    ax4.invert_yaxis()
    ax4.set_xlabel('Ball Control Rating')
    ax4.set_title('10 Best Ball Controllers')

    ax5 = fig.add_subplot(235)
    x5 = passing.iloc[0:10]['total_passing'].values
    y5 = passing.iloc[0:10]['player_name_x'].values

    ax5.barh(y5, x5, align='center',
            color='green', ecolor='black')
    ax5.set_yticks(range(len(y5)))
    ax5.set_yticklabels(y5)
    ax5.set_xlim(75,100)
    ax5.invert_yaxis()
    ax5.set_xlabel('Passing Rating')
    ax5.set_title('10 Best Passers')

    ax6 = fig.add_subplot(236)
    x6 = defending.iloc[0:10]['total_defending'].values
    y6 = defending.iloc[0:10]['player_name_x'].values

    ax6.barh(y6, x6, align='center',
            color='green', ecolor='black')
    ax6.set_yticks(range(len(y6)))
    ax6.set_yticklabels(y6)
    ax6.set_xlim(75,100)
    ax6.invert_yaxis()
    ax6.set_xlabel('Defending Rating')
    ax6.set_title('10 Best Defenders')

    plt.tight_layout()

def single_radar_plot(player1,player2,df):
    '''
    INPUT: Two player names to compare, dataframe
    OUTPUT: Radar Plot
    '''
    df2 = df.reset_index()
    df3 = df2.drop(['id','player_fifa_api_id','player_api_id','gk_diving','gk_handling','gk_kicking',
    'gk_positioning','gk_reflexes','p_id'], axis = 1)
    cols = df3.columns
    feats = np.array(cols[1:])
    stats = df3[df3['player_name'] == player1].values[:,1:]
    stats2 = df3[df3['player_name'] == player2].values[:,1:]
    angles=np.linspace(0, 2*np.pi, len(feats), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))
    stats2 = np.concatenate((stats2,[stats2[0]]))
    fig=plt.figure(figsize = (18,10))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2, label = player1)
    ax.fill(angles, stats, alpha=0.25)
    ax.plot(angles,stats2,'o-', linewidth=2, label = player2)
    ax.fill(angles,stats2, alpha= 0.25)
    ax.set_thetagrids(angles * 180/np.pi, feats)
    ax.set_title(player1 + ' ' + 'and' + ' ' + player2)
    ax.legend()
    ax.grid(True)

def corr_heat(df):
    '''
    INPUT: Dataframe
    OUTPUT: Correlation Heat Map
    '''
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},xticklabels=corr.index, yticklabels=corr.columns)
    plt.xticks(rotation=60, ha="right")
    plt.yticks(rotation=0)
    ax.set_title("Correlation Heat Map")

def feat_dist(df):
    '''
    INPUT: Dataframe
    OUTPUT: Feature Distributions
    '''
    df.hist(figsize = (18,8))
    plt.tight_layout()

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

def normalize_data(df):
    '''
    INPUT: Dataframe
    OUTPUT: Normalized data between 0 and 1 for NMF
    '''
    df_std_norm = df.drop(['preferred_foot','attacking_work_rate','defensive_work_rate'], axis = 1)
    df_dummies_norm = df[['preferred_foot','attacking_work_rate','defensive_work_rate']].values
    scaler = MinMaxScaler()
    new_x_norm = scaler.fit_transform(df_std_norm)
    final_data_norm = np.concatenate((new_x_norm,df_dummies_norm), axis = 1)

    return final_data_norm

def K_nearest(player, nameslist, cluster_list, X):
    '''
    INPUT: Player name, List of names, list of clusters, Dataframe
    OUTPUT: Five closest neighbors and indices
    '''
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    idx = np.where(nameslist == player)[0][0]
    # Pull indices and distances of the 5 nearest neighbors
    dist = distances[idx][1:]
    ind = indices[idx][1:]
    # pull names from series of player names
    five_names = np.array(nameslist[ind])

    return dist, five_names, ind, indices

def five_nearest_radar(player, nameslist, fivenearest, indices, cols, X):
    '''
    INPUT: Player to compare, list of player names, list of five nearest neighbors
    indices of 5 nearest, column labels, Dataframe
    OUTPUT: Radar chart of selected player and 5 nearest neighbors
    '''

    # Find index of desired player
    idx = np.where(nameslist == player)[0][0]
    # Pull indices of 5 nearest neighbors for that player
    ind = indices[idx]
    cols = cols
    feats = np.array(cols)
    #Get attributes of 5 nearest neighbors
    stats = X.loc[ind[0],feats].values
    stats1 = X.loc[ind[1],feats].values
    stats2 = X.loc[ind[2],feats].values
    stats3 = X.loc[ind[3],feats].values
    stats4 = X.loc[ind[4],feats].values
    stats5 = X.loc[ind[5],feats].values

    # Set up angles for radar plot
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
    ax.legend(loc = 'upper right')
    ax.grid(True)

    plt.show()

def nmf_model(n_components, max_iter, df):
    '''
    INPUT: number of componenets, max iterations, dataframe
    OUTPUT: W and H matrices
    '''
    model = NMF(n_components = n_components, max_iter = max_iter)
    model.fit(df)
    W = model.fit_transform(df)
    H = model.components_

    return W,H

def top_five_pertopic(H, cols):
    '''
    INPUT: H matrix, column names list
    OUTPUT: Five most weighted features per topic
    '''
    array = np.argsort(H)

    # For each topic, pull the 5 most highly weighted features
    top_fives = []
    for idx in range(len(array)):
        top_five = array[idx][-5:]
        top_five_r = top_five[::-1]
        top_fives.append(top_five_r)

    # Replace indices with feature names
    five_names = []
    for idx in range(len(top_fives)):
        arr = top_fives[idx]
        names = []
        five_names.append(names)
        for val in arr:
            name = cols[val]
            names.append(name)

    return five_names

if __name__ == '__main__':

    # Data Cleaning/Standardization
    player_stats = get_player_stats("SELECT * FROM Player_Attributes;")
    player_data = get_player_data("SELECT * FROM Player;")
    clean_df = prep_player_data(player_stats)
    mean_player_df = mean_player_data(clean_df)
    names = get_player_names(mean_player_df,player_data)
    std_data = standardize_data(mean_player_df)
    no_gks_df = no_gks(mean_player_df)
    std_data_no_gk = standardize_data(no_gks_df)

    # EDA
    data_w_country = pd.read_csv('playerteams.csv')
    leagues, s_data, cols, player_mean_stats = players_and_teams(data_w_country)
    # country_plots(leagues,s_data)
    # player_plots(no_gks_df)

    # PCA
    pca, X_pca = PCA_model(11,std_data)
    # scree_plot(pca)
    pca_comps = PCA_components(std_data, pca, mean_player_df.columns)
    # single_radar_plot('Mesut Oezil', 'Andres Iniesta', player_mean_stats)
    # data_plot(X_pca)

    #K-Means
    # silhouette_charts(X_pca,range(2,12))
    kmeans, centers, k_labels = k_means_model(6,X_pca)
    name_clusters = name_clusters(names, k_labels)

    # KNN
    dist, five_names, ind, indices = K_nearest('Romelu Lukaku', names, name_clusters, X_pca)
    mean_player_df = mean_player_df.reset_index()
    df_mean = mean_player_df.drop(['player_api_id'], axis = 1)
    five_nearest_radar('Romelu Lukaku', names, five_names, indices, df_mean.columns, df_mean)

    # NMF
    norm_data = normalize_data(df_mean)
    W,H = nmf_model(7,250,norm_data)
    top_five = top_five_pertopic(H, df_mean.columns)
