# Clustering Soccer Players

![genre](images/example.png)

## Motivation:

One of my biggest passions in life lies within the sport of soccer. I love everything about the game, from watching casually, to analyzing tactics, analyzing soccer data has always been something I was interested in. Having played the game all my life, I have always been interested in the numerous different playstyles that soccer players can fall into. No soccer players truly play the exact same way and it can be difficult to compare players as a result. When watching soccer, I always find myself saying "I wish we had a player like player X", this project helps to address that statement. This project aims to use data on soccer player attributes in order to draw meaningful comparisons between players based on their playstyle. 

## Project Description:

For this project, I explored soccer player attributes from the EA Sports video game FIFA in an attempt to cluster players based on playstyle. This project addresses two main questions:
  - Can players be clustered based on their playstyle?
  - What players are most similar to a given player that have the same playstyle?

Based on prior knowledge, soccer players can generally be identified by the following playstyles:

  - Forwards: Winger, False 9, Target Forward, Striker
  - Midfielders: Deep Lying Playmaker, Advanced Playmaker, Holding Midfielder, Box to Box Midfielder
  - Defenders: Centerback, Fullback, Wingback
  - Goalkeepers

This project aims to cluster players according to similar playstyles, and use these clusters as means of player comparison. Both hard clustering (K-Means) and soft clustering methods (NMF) were used to attempt to cluster players. For players in the same cluster, K nearest neighbors was used to determine the most similar players to that player based on playstyle. 

## Data:

Data for this project came from a European soccer dataset on kaggle which can be found [here](https://www.kaggle.com/hugomathien/soccer/kernels). This data came in the form of a SQL database with 7 different tables. The main table used in this project was the Player_Attributes table which contained the FIFA attribute ratings for each player. Each attribute is on a scale from 0-100 with the attributes being determined by EA Sports algorithms based on player performance. 

This dataset contained FIFA ratings from FIFA 2012-2016 and each player had multiple rows with their attributes over this time period. The different data points for each player indicated that players "form", or performance at the time when the attributes were updated. The dataset contained ~9950 players with 36 features for each player. The data contained the following features:

![cols](images/cols.png)


## Data Cleaning:

The following actions were performed to clean the data:
  - Removed players with NaN's.
  - Aggregated player attributes over time to the mean values.
  - Removed statistics not representative of playstyle.
  - Standardized/normalized data.

## EDA:
Examining the distribution of features:

![feat](images/feat_distributions.png)

Looking at players by league:

![comps](images/league_comps.png)

Looking at the best players by attribute:

![comp](images/10_best_players.png)

Before clustering, the dimensionality of the data was a concern. This dataset had a problem with multicolinearity exemplified by high variance inflation factors. Many of the features were very highly correlated with eachother as shown with a correlation heat map. 

![corr](images/corr_heat.png)

PCA was utilized before clustering with 11 principal components in accordance with the scree plot show below.

![pca](images/pca_scree2.png)

To attempt to interpret the principal components, the components of each principal component were analyzed according to the top 5 weighted features for each principal component. 

![pca](images/PCA_comps2.png)

Ultimately the principal componenets were not very interpretable and did not provide any discernable insights. 

Visualizing the data in 2D after PCA:

![blob](images/data_blob2.png)

An example of a radar chart used to compare players is shown below with the players Mesut Ozil and Andres Iniesta.

![radar](images/radarexample.png)

## Clustering:

The two main types of clustering used in this project were a hard clustering method in K-means, and a soft clustering method in Non Negative Matrix Factorization. The goal was to generate player clusters that could be used for player comparison. Other clustering techniques such at t-SNE, DBSCAN, and HDBSCAN were experimented with but did not produce interpretable results for this data. 

## K-means:

The first attempt at clustering the data was made with K-means clustering. I attempted to cluster with a range of clusters from 2 to 12 and observed the resulting silhouette plots and cluster visualizations. Cluster values of 4,5, and 6 returned the bset clusters based on silhoutte score and interpretability. The resulting silhouette plots for these number of clusters are shown below. 

![k-means](images/sil3.png)

Average silhouette score: .283

![k-means](images/sil.png)

Average silhouette score : .263

![k-means](images/sil2.png)

Average silhouette score: .225

Ultimately, 6 clusters seemed to be the optimal number of clusters to differentiate playstyles as much as possible. After 6 clusters, the clustering became extremely messy and non interpretable. After looking at the cluster centers and players within the clusters, I labeled the clusters as following:
  - Cluster 1: Center Backs
  - Cluster 2: Forwards/Advanced Playmakers
  - Cluster 3: Goalkeepers
  - Cluster 4: Full Backs/Center Backs
  - Cluster 5: Box to Box Midfielders/Holding Midfielders
  - Cluster 6: Wingers/Wingbacks
  
To visualize that these clusters are indeed doing at least a decent job in seperating the players, I observed a radar chart from two players in opposite clusters (Cluster 2 and Cluster 6).
 
![diff](images/oz_celo.png)

These radar charts do in fact not line up well, which shows that these clusters are at least seperating players fairly well based on position and attribute ratings. The clustering also obviously did a good job in seperating goalkeepers from field players, but this should be expected due to how goalkeepers are rated in FIFA. 

## NMF:

Using NMF, the goal was to soft cluster players into latent topics or in this case playstyles. From these latent topics, the top 5 most weighted features were analyzed in order to assign the topic to a playstyle. NMF was performed using 7 latent topics with the resulting latent topics and labels shown below.

![nmf-2](images/nmf_topics.png)

From these labels, individual players could be interpreted based on how highly they weight on a particular playstyle. 9 players were selected who I had strong intuition as to what playstyle they should weight most heavily on. These players are shown in the plot below.

![nmf](images/nmf_weights.png)

Ultimately, the soft clustering proved to be a bit more insightful into a particular players playstyle. Soccer players are unique in that they often exhibit characteristics and have attributes that could be translated to many different positions. This soft clustering gave a better insight into what combination of attributes and playstyles a player weights most on, rather than a hard cluster assignment as with k-means. The NMF results were ultimately the most interpretable and lined up most closely with my prior intuition and knowledge of these players. 

## Player Comparisons:

Player comparison was done using K Nearest Neighbors with a player and its five nearest neighbors in the same cluster. The player and their five nearest neighbors can be compared visually using radar charts. Some results of this are shown using the radar charts below for 5 selected players. 

![radar1](images/Luka_Modric.png)

![radar2](images/Lionel_Messi.png)

![radar3](images/Mesut_Ozil.png)

![radar4](images/Clint_Dempsey.png)

![radar5](images/Ronaldo.png)

While the data seemed to be unfit for clustering in general, the player comparisons managed to be fairly interpretable. When analyzing the radar charts, the player comparisons made sense both intuitively and visually. The comparable players map together fairly well as shown by the radar charts. While the clustering did not do a fantastic job of clustering players by playstyle, the clusters do seem to still cluster very similar players based on their attributes and position. 

## Future Work:

- Obtain real player statistics.
- Gather more granular data (web scraping).
- Obtain transfer market data for each player.
- Compare players based on performance and price.
- Transfer market recommender for low budget teams.
- Make historical comparisons between players.

## References:

-   https://www.kaggle.com/hugomathien/soccer/kernels
