# Clustering Soccer Players

![genre](images/example.png)

## Motivation:

One of my biggest passions in life lies within the sport of soccer. I love everything about the game, from watching casually, to analyzing tactics, analyzing soccer data has always been something I was interested in. This project aims to use data on soccer player attributes in order to draw better comparisons between players.

## Project Description:

For this project, I explored soccer player attributes from the EA Sports video game FIFA in an attempt to cluster players based on playstyle. This project addresses two main questions:
  - Can players be clustered based on their playstyle?
  - What players are most similar to player X that have the same playstyle?

Based on prior knowledge, soccer players can generally be identified by the following playstyles:

  - Winger, False 9, Target Forward, Striker
  - Deep Lying Playmaker, Advanced Playmaker, Holding Midfielder, Box to Box Midfielder
  - Centerback, Fullback, Wingback

## Data:

Data for this project comes from a dataset on kaggle which can be found [here](https://www.kaggle.com/hugomathien/soccer/kernels). This data comes in the form of a SQL database with 7 different tables. The main table used in this project was the Player_Attributes table which contained the FIFA attribute ratings for each player.

This dataset contained FIFA ratings from FIFA 12-16 and each player had multiple rows with their attributes over this time period. The data contained the following features:

![cols](images/cols.png)


## Data Cleaning:

The following actions were performed to clean the data:
  - Remove players with NaN's
  - Aggregate player attributes over time to the mean values
  - Remove Goalkeepers and Goalkeeper statistics
  - Standardize data

## EDA:
Examining the distribution of features:

![feat](images/feat_distributions.png)

Looking at players by league:

![comps](images/league_comps.png)

Looking at the best players by attribute:

![comp](images/10_best_players.png)

Before clustering, the dimensionality of the data was a concern. This dataset had a problem with multicolinearity as shown by high Variance Inflation Factors.

![corr](images/corr_heat.png)

PCA was utilized before clustering with 12 principal components in accordance with the scree plot show below.

![pca](images/pca_scree.png)

Visualizing the data in 2D after PCA:

![blob](images/data_blob.png)

An example of a radar chart used to compare players is shown below with the players Mesut Ozil and Andres Iniesta.

![radar](images/radarexample.png)

## Clustering:

The two main types of clustering used in this project were a hard clustering method in K-means, and a soft clustering method in Non Negative Matrix Factorization. The goal was to generate player clusters that could be used for player comparison.

## K-means:

![k-means](images/sil_no_gk5.png)

Average silhouette score : 0.181

## NMF:

Using NMF, the goal was to soft cluster players into latent topics or in this case playstyles. NMF was performed using 7 latent topics with the resulting latent topics shown below.

![nmf-2](images/nmf_topics.png)

From these labels, individual players could be interpreted based on how highly they weight on a particular playstyle. 9 players were selected who I had strong intuition as to what playstyle they should weight most heavily on. These players are shown in the plot below. 

![nmf](images/nmf_weights.png)

## Results:

## Player Comparisons:

Player comparison was done using K Nearest Neighbors with a player and its five nearest neighbors in the same cluster.

## Future Work:

- Obtain real player statistics.
- Gather more granular data (web scraping).
- Obtain transfer market data for each player.
- Compare players based on performance and. price
- Make historical comparisons between players.

## References:

-   https://www.kaggle.com/hugomathien/soccer/kernels
