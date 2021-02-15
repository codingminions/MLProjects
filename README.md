# Machine-Learning
List of Machine Learning Projects implemented

1. Board Game Prediction:
    * Reviews can make or break a product; as a result, many companies take drastic measures to ensure that their product receives good reviews. 
    * A linear regression model was used to predict the average review a board game will receive based on characteristics such as minimum and maximum number of players, playing time, complexity, etc.
    * A correlation matrix was created to explore the relationships between parameters.
    * Mean squared error was used as a performance metric.
    * Performance of this model was compared with an ensemble method called RandomForestRegressor.
    * Jupyter Notebook: https://nbviewer.jupyter.org/github/codingminions/Machine-Learning/blob/main/Board_Game_Prediction/Board_Game_Review.ipynb

2. Breast Cancer Detection:
    * This breast cancer database was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg. [Citation: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/]
    * Two machine learning models were used to get the most accurate result: KNN - KNeighborsClassifier , SVM - Support Vector Machine.
    * Cross Val score was used to estimate the skill of these machine learning model on unseen data. 
    * Precision and Recall were used as performance metric.
    * Performance of these two models was compared with Support Vector Classifier(SVC).
    * Jupyter Notebook: https://nbviewer.jupyter.org/github/codingminions/Machine-Learning/blob/main/Breast_Cancer_Prediction/Breast%20Cancer.ipynb

3. Credit Card Fraud Detection:
    * Using a dataset of of nearly 28,500 credit card transactions and multiple unsupervised anomaly detection algorithms, transactions with a high probability of being credit card fraud was identified.
    * Two machine learning algorithms were deployed: Local Outlier Factor (LOF) and Isolation Forest Algorithm.
    * Parameter histograms and correlation matrices were used to gain a better understanding of the underlying distribution of data in our data set.
    * We used outlier algorithms since the number of fraud cases were way less than valid transactions.
    * Jupyter Notebook: https://nbviewer.jupyter.org/github/codingminions/Machine-Learning/blob/main/Credit_Card_Fraud_Detection/Credit%20Card%20Fraud%20Detection.ipynb

4. K Means Clustering for Imagery Analysis:
    * K-means algorithm was used to perform image classification.
    * MNIST dataset was used for this project.
    * Since the images are stored as 2 Dimensional Array , we preprocess them using reshape function to convert to 1 Dimensional array.
    * Mini-batch implementation of k-means clustering was implemented on these 1D values. 
    * Accuracy was used as the Performance metric.
    * The model was tested against other number of clusters and inertia,Homogeneity of these clusters were calculated.
    * Jupyter Notebook: 

5. NLTK: 
    * Various aspects of NLP was identified and deployed on text corpus.
    * Concepts Covered: Tokenize , Stemming , Stop words with NLTK, POS Tagging, Chunking, Named Entity Recognition and Text Classification.
    * SklearnClassifier was used to perform text classification on movie reviews.
    * Jupyter Notebook: 

6. Principle Component Analysis (PCA):
    * Mapping high dimensional data to a lower dimensional space is a necessary step for projects that utilize data compression or data visualizations.
    * K Means clustering was performed on the well known Iris data set.
    * Elbow method was used to determine the accurate number of clusters used in the process.
    * PCA was used to reduce the number of features in the dataset to 2.
    * Homogeneity and Completeness of Reduced and Non-Reduced data was compared.
    * Jupyter Notebook: 

7. Stock Market Prediction: 
    * Dictionary of Companies and their stock market abbreviations was created.
    * Closing and Opening Data of each of the companies was checked.
    * Daily Stock Movement was calculated.
    * Data was normalised and KMeans Clustering algorithm was implemented.
    * PCA was used to reduce the number of features.
    * Jupyter Notebook:  
