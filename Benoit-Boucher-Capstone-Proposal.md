# Udacity's Machine Learning Engineer Nanodegree
## Capstone Proposal
Benoît Boucher, 2018-07-20

## Proposal

### Domain Background

Contxtful is a web marketing company that specializes in using smartphone telemetry data, such as the gyroscope and acceleration sensors, to detect the context of a user. For instance, is he at rest or walking? Is he looking at his screen or not?

This information helps Contxtful push context-aware ads to people when they are most receptive.

One of the problems that Contxtful encountered is detecting the type of phone used to generate this data. This is relevant because every phone has different sensors which report their own version of sensor data that can be classified into their different categories (active, at rest, looking at screen, etc.). Knowing the make and model of the phone would allow Contxtful to develop and use make-model specific models, which would result in faster compute time and more accurate results.

Up until now, Contxtful has been extracting the make-model information from a special collector agent, but this agent has problems of its own and it would be preferable to skip it entirely.

My first personal motivation for working with Contxtful is because it is a wonderful opportunity to work on a real-world problem, as opposed to a toy problem. Secondly, the tech lead is an acquaintance of mine, so I'm helping a friend. Finally, this is a great foot in the door for future collaboration or employment.
 

### Problem Statement

The primary goal is to identify the phones make-model based solely on their telemetry data. This can be measured by a simple label classification accuracy score.

As a secondary target, it would be interesting to cluster the phones and see if a natural separation of the samples occur based on their make-models. This can be measured by weighting the representation of each phone make-models into the clusters.   

### Datasets and Inputs

The dataset is derived from a measure of a phone's telemetry. What Contxtfull calls a "raw_data_vector" is generated every second and includes all data points from the various sensors (X-Y-Z gyroscope, X-Y-Z accelerometer, etc) sampled at the phone's sample rate. This rate varies by model, but is generally in the range of 16-32 Hz. Consequently, the raw_data_vector do not have a consistent length. 

The "raw_data_vector" will not be available to us for our ML models.

What will be used in our ML models is what Contxtfull refers to as "full_stat_vector". This is a vector of consistent length that includes a slew of various statistical metrics. For instance: maximum_gyroscope_speed_Z, first_derivate_linear_acceleration_Z, mean_euclidean_speed, etc. The total number of features is 1652.
 
 The "full_stat_vector" will be provided without labels, preventing us from knowing specifically what each column represents.

Telemetry data is collected on phones who browse websites on which the Contxtful plugin is installed. The specific websites wishes to remain anonymous.

Approximately 12 000 samples (full_stat_vector) will be provided.


### Solution Statement

The solution to the primary goal is to apply a Supervised Learning model. Many classifiers will be tried until one with a good accuracy is found. The accuracy of predicted labels is the metric to use.

For the secondary goal, various Unsupervised Learning models will be tried. A metric that measures the disparity of the phones into each cluster will indicate how good they are split. 

A Random Search or Grid Search algorithm will be run to optimize the hyper-parameters. The best hyper-parameters found will be saved to disk as to allow anyone to be able to retrain the same model and obtain the same results.

In fact, to be completely replicable, we will provide access to the Python code behind every model tried.


### Benchmark Model

For the classification problem, Contxtful has obtained good results using Linear Discriminent Analysis (LDA), which is also in line with the idea to "cluster" the data since at its core, it is a supervised dimensionality reduction algorithm. 

For the clustering problem, Contxtful did not obtain satisfying results. However, their current best approach is with the Gaussian Mixture.

The results to both problems are measurable. While the classificition is a simple accuracy metric, the clustering will be measured by a homebrew metric because the problem is very specific in nature. In both cases, visual aid can be provided:
* A bar chart of accuracy per label for Classification.
* A series of bar charts showing what each cluster contain in terms of labels. 

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

**Classification Accuracy:** *(# of correctly predicted labels) / (total # of labels)*. This is a simple and widely used metric. 

**Clustering Score** Because of the unique nature of the problem (specifically: verifying that labels are grouped into unique clusters), we came up with a homemade metric to measure the quality of our clustering. The basis for the calculation is a matrix where the rows are the true labels, the columns are the new clusters, and each item(i,j) in the matrix is the percentage of samples that has label(i) and was classified into cluster(j). Every entry in this matrix will be passed into the following special function:
 
 *f(x) = 2 * |x - 0.5|* 

This function was calibrated to return a score close to 1 if the input ratio is either close to 0 or 1. A good score signifies that the labels were indeed clustered into clusters in an exclusive fashion. In other words, a specific label should be found inside a unique cluster and nowhere else. When then average the score of every item in the label-cluster matrix to give the final score.

There is a caveat to this metric: while it works well when clusters are formed, it gives extremely good results when no clustering occur. Specifically, take the case in which a clustering technique returns are 4 different clusters. If 3 clusters contain only a single label and the fourth cluster contains 100% of the remaining, the Clustering Score will give a score close to 100% because technically, the labels are highly localised into single clusters, even though in reality, no clustering to speak of would have taken place.

### Project Design
_(approx. 1 page)_

The following workflow will be repeated twice, once for each separate objective:
**General Workflow**
1. Reproduce the results of Contxtful.
2. Try various algorithms appropriate for the goal. See the list below.
3. Once a good approach is found, optimize the hyperparameters, if any. This can be done via BIC score for Gaussian Mixture, or Random Search Cross Validation for all other techniques.
4. Present the results.

**Algorithms for Classification**
1. Support Vector Machine
2. Neural Network
3. Random Forest

**Algorithms for Clustering**
1. Mean Shift
2. Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
3. Agglomerative
4. K Means