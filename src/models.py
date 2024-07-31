import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder 

import yaml
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import os

from data_preprocessing import PenguinDataPreprocessor

class PenguinModels:
    def __init__(self):
        self.load_config()
        self.prepare_data()
        # Storing variance info together
        self.explained_variance_info = {}
        
    def load_config(self):
        try:
            with open('src/config.yaml', 'r') as file:
                self.CONFIG = yaml.safe_load(file)
            self.model_names = {
                'k_means': self.CONFIG['model_1_name'],
                'gaussian_mixture': self.CONFIG['model_2_name'],
                'dbscan': self.CONFIG['model_3_name']
            }    
        except FileNotFoundError:
            print("Config file not found. Send help.")
            # What if we just trusted the config file loads successfully?
    
        print("\nConfiguration loaded successfully:", self.CONFIG)
        
    
    def prepare_data(self):
        print("\nStarting data preparation ...")
        # Class is in data_preprocessing.py file, returns preprocessed df
        preprocessor = PenguinDataPreprocessor()
        # Preprocess data
        self.df = preprocessor.fit_transform()
        # Assign entire preprocessed dataframe to self.X, all columns to be used for clustering
        self.X = self.df 
        # I really like this way of showing the features for smaller datasets
        print("\nData preparation completed.")
        
        print("\nFeatures used for clustering:")
        print(self.X.columns.tolist())
        
    
    def create_visualization_directory(self):
        directory = "model_visualizations"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
        
        
         # Not doing cross-validation for clustering
        
    def perform_kmeans(self):
        print("\nPerforming K-means clustering ...")
        # Starting with positive infinity when looking for lowest inertia
        best_inertia = float('inf')
        best_model = None
        best_params = None
        
        for n_clusters in self.CONFIG['k_means_params']['n_clusters']:
            for init in self.CONFIG['k_means_params']['init']:
                kmeans = KMeans (n_clusters=n_clusters, 
                                init=init, 
                                random_state=self.CONFIG['general']['random_state'])
                kmeans.fit(self.X)
                # Look up the difference in metrics
                # For inertia lower = better
                inertia = kmeans.inertia_
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_model = kmeans
                    best_params = {'n_clusters': n_clusters, 'init': init}
                   
        if best_model is not None:
            # Fit model then predict, model is fit above
            self.df['k_means_cluster'] = best_model.predict(self.X)
            print("K-means clustering completed.")
            print(f"Best parameters: {best_params}")
        else: 
            print("K-means clustering failed to find a suitable model.")
        
        
    def perform_gaussian_mixture(self):
        print("\nPerforming Gaussian Mixture Model clustering ...")
        best_bic = float('inf')
        best_model = None
        best_params = None
        
        for n_components in self.CONFIG['gaussian_mixture_params']['n_components']:
            for covariance_type in self.CONFIG['gaussian_mixture_params']['covariance_type']:
                gmm = GaussianMixture(n_components=n_components, 
                                    covariance_type=covariance_type,
                                    random_state=self.CONFIG['general']['random_state'])
                gmm.fit(self.X)
                bic = gmm.bic(self.X)
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_params = {'n_components': n_components, 'covariance_type': covariance_type}
        
        if best_model is not None:
            self.df['gaussian_mixture_cluster'] = best_model.predict(self.X)
            print("Gaussian Mixture Model clustering completed.")
            print(f"Best parameters: {best_params}")
        else:
            print("No valid model found.")
        
        
    def perform_dbscan(self):
        print("\nPerforming DBSCAN clustering ...")
        best_score = float('-inf')
        best_model = None
        best_params = None
        
        for eps in self.CONFIG['dbscan_params']['eps']:
            for min_samples in self.CONFIG['dbscan_params']['min_samples']:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.X)
                
                
                # Making sure that not all samples belong in the same cluster
                # In order to calculate silhouette score
                if len(set(labels)) > 1:
                    score = silhouette_score(self.X, labels)
                    if score > best_score:
                        best_score = score
                        best_model = dbscan
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        
        if best_model is not None: 
            self.df['dbscan_cluster'] = best_model.fit_predict(self.X)
            print("DBSCAN clustering completed.")
            print(f"Best parameters: {best_params}")
        else:
            print("DBSCAN could not find a suitable clustering. Try different parameter ranges.")
        
        
    def visualize_clusters(self, model_name):
        print(f"\nVisualizing clusters for {model_name} ...")
        cluster_column = f'{model_name}_cluster'
        print("Done.\n")
        
        # Checking if clustering has actually been performed before visualizing it
        # Whether cluster column exists in the df
        
        if cluster_column not in self.df.columns:
            print(f"Error: {cluster_column} not found in dataframe.")
            return
        
        # Using all columns
        feature_cols = [col for col in self.df.columns if col != cluster_column]
        
        # Creating visualization directory to auto put into the file
        viz_dir = self.create_visualization_directory()
        
        
        # Pairplot
        plt.figure(figsize=(20, 20))
        sns.pairplot(self.df, hue=cluster_column, vars=feature_cols, plot_kws={'alpha': 0.6})
        plt.suptitle(f'Pairplot of Features - {model_name.upper()}', y=1.02)
        plt.savefig(os.path.join(viz_dir, f'{model_name}_pairplot.png'))
        plt.close()
        
        # 3D scatter plot with PCA ***
        
        # Performing PCA to reduce components from 6 > 3 for 3D scatter plot
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.df[feature_cols])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            pca_result[:, 2],
            c=self.df[cluster_column],
            cmap='viridis'
        )
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.title(f'3D PCA Cluster Visualization - {model_name.upper()}')
        plt.colorbar(scatter)
        plt.savefig(os.path.join(viz_dir, f'{model_name}_3d_pca.png'))
        plt.close()
        
        
        explained_variance = pca.explained_variance_ratio_
        total_explained_variance = sum(explained_variance)
        
        self.explained_variance_info[model_name] = {
            'ratio': explained_variance,
            'total': total_explained_variance
        }
        
        
    def run_all_models(self):
        self.perform_kmeans()
        self.visualize_clusters(self.model_names['k_means'])
        
        self.perform_gaussian_mixture()
        self.visualize_clusters(self.model_names['gaussian_mixture'])
        
        self.perform_dbscan()
        self.visualize_clusters(self.model_names['dbscan'])
        
        self.write_consolidated_variance_info()
    
    def write_consolidated_variance_info(self):
        viz_dir = self.create_visualization_directory()
        with open(os.path.join(viz_dir, 'consolidated_explained_variance.txt'), 'w') as f:
            for model_name, info in self.explained_variance_info.items():
                f.write(f"\n{model_name} Explained Variance:")
                f.write(f"\nExplained variance ratio: {info['ratio']}")
                f.write(f"\nTotal explained variance: {info['total']:.2f}\n")
    

penguin_models = PenguinModels()
penguin_models.run_all_models()