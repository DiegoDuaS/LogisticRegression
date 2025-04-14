from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def breif_clustering(X, n_clusters):

    X_pca = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)

    X['Cluster'] = km.fit_predict(X_pca)
    centroides = km.cluster_centers_
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=X['Cluster'], palette="tab10", legend="full")
    # Plot centroids
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label="Centroids")

    plt.title(f"K-Means Clustering (Reducido con PCA) - 3 Clusters")
    plt.legend()
    plt.show()
    return X

def drop_many_nulls(df):
    df = df.copy()  # Evitar modificar el dataframe original
    
    # Eliminar las variables que no queremos en el análisis de clusters
    drop_columns = [
        'Id', 'PoolArea', 'MiscVal', 'BsmtFinSF2', 'BsmtFinSF1', 'MasVnrArea',
        'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Alley', 'ExterCond',
        'BsmtHalfBath', 'KitchenAbvGr', 'PoolQC', 'Fence', 'MiscFeature', 'MiscFeature',
        'FireplaceQu', 'MasVnrType', 
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    return df

from sklearn.preprocessing import LabelEncoder
import pandas as pd

def trans_categorical(df):
    # Variables ordinales con asignación de valores
    ordinal_mappings = {
        'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
    }

    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0)

    # Variables nominales -> Label Encoding (sin One-Hot)
    nominal_cols = [
        'LotShape', 'LotConfig', 'Neighborhood', 'BldgType',
        'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'Foundation', 'BsmtFinType1', 'GarageType', 'MSZoning', 'Street',
        'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2', 'RoofMatl',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional',
        'SaleCondition', 'SaleType', 'GarageQual', 'GarageCond', 'CentralAir', 'PavedDrive'
    ]

    for col in nominal_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def aic_bic(lr_model, X_test, y_test):
    yproba = lr_model.predict_proba(X_test)[:, 1]
    eps = 1e-15
    yproba = np.clip(yproba, eps, 1 - eps)
    log_likelihood = np.sum(y_test * 
                            np.log(yproba) + (1 - y_test) * 
                            np.log(1 - yproba))
    n = X_test.shape[0]  # number of observations
    k = X_test.shape[1] + 1  # number of parameters (features + intercept)

    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return aic, bic

def cm_and_metrics(y_test, y_pred, title):
    print("Precision:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    # Confusion Matrix
    cm_custom = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_custom, annot=True,cmap='Blues', fmt='d')
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión Cara o No ({title})")
    plt.show()