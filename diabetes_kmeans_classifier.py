import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.cluster import KMeans


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    print ("\n ---------- K MEANS CLUSTERING ON DIABETES DATA----------------\n")
    data = pd.read_csv("./data.csv")   #importing files using pandas
    dataset_new = data
    dataset_new[[
        "Glucose", "BloodPressure", 
        "SkinThickness", "Insulin", 
        "BMI"]] = dataset_new[[
            "Glucose", "BloodPressure", 
            "SkinThickness", "Insulin", 
            "BMI"]].replace(0, np.NaN) 

    # Replacing NaN with mean values
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
    dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

    # Feature scaling using MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    dataset_scaled = sc.fit_transform(dataset_new)

    data1 = pd.DataFrame(dataset_scaled)
    # Selecting features - [Glucose, Insulin, BMI]
    X = data1.iloc[:, [1, 4, 5]].values
    Y = data1.iloc[:, 8].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

    # Checking dimensions
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_test shape:", Y_test.shape)

    KMeans_Clustering = KMeans(n_clusters =2, random_state=0)
    KMeans_Clustering.fit(X_train)

    print(KMeans_Clustering.cluster_centers_)

    #prediction using kmeans and accuracy
    kpred = KMeans_Clustering.predict(X_test)
    print('Classification report:\n\n', sklearn.metrics.classification_report(Y_test,kpred))

    outcome_labels = sorted(data.Outcome.unique())
    sns.heatmap(
        confusion_matrix(Y_test, kpred),
        annot=True,
        xticklabels=outcome_labels,
        yticklabels=outcome_labels
    )

    # Fit again and plot
    KMeans_Clustering = KMeans(n_clusters =2, random_state=0)
    KMeans_Clustering.fit(X)

    plt.scatter(data1.iloc[:, [1]].values,data1.iloc[:, [5]].values, c=KMeans_Clustering.labels_, cmap='rainbow')