import pandas as pd
import numpy as np

from numba import njit, prange
from category_encoders import OrdinalEncoder

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import time


class LeaderAlgorithms:
    """
    A class to perform the clustering algorithms fo the Leader family with custom similarity functions.

    Attributes:
    ----------
    data : pd.DataFrame
        The input dataset.
    varinfo : pd.DataFrame
        DataFrame containing metadata about the variables in the dataset. Required columns: ['name', 'type'] where 'type' 
        can be ['Binary', 'Categorical', 'Integer', 'Continuous].
    weights : list or np.ndarray, optional
        Weights for each feature in the dataset. If None, then uniform weight will be applied. Default None.
    similarity_func : str, optional
        The similarity function to use, either 'gower' or 'euclidean'. Default 'gower'.
    label_encoders : dict
        Encoders used for categorical variables.
    types : dict
        Data types of the features.
    ranges : dict
        Ranges of the Integer and Continuous features.
    pairwise_similarity : function
        The function used to calculate pairwise similarity.
    leader_similarities : np.ndarray or None
        Similarities between leaders after clustering.
    """
    def __init__(self, data, varinfo, weights=None, similarity_func = 'gower', seed=None, verbose = 1):
        """
        Initializes the LeaderAlgorithms object with data, variable information, and optional weights and similarity function.

        Parameters:
        ----------
        data : pd.DataFrame
            The input dataset.
        varinfo : pd.DataFrame
            DataFrame containing metadata about the variables in the dataset. Required columns: ['name', 'type'] where 'type' 
            can be ['Binary', 'Categorical', 'Integer', 'Continuous].
        weights : list or np.ndarray, optional
            Weights for each feature in the dataset. If None, then uniform weight will be applied. Default None.
        similarity_func : str, optional
            The similarity function to use, either 'gower' or 'euclidean'. Default 'gower'.
        """
        if seed != None:
            np.random.seed(seed)
        if 'role' in varinfo.columns:
            varinfo = varinfo[varinfo['role'] == 'Feature'].reset_index(drop=True)
        self.check_formats(data, varinfo, weights, similarity_func)
        self.varinfo = varinfo
        self.data, self.label_encoders = self.preprocess_data(data, varinfo)
        self.types, self.ranges, self.weights = self.preprocess_varinfo(self.data, varinfo, weights)
        self.pairwise_similarity = self._gower_pairwise if similarity_func == 'gower' else self._euclidean_pairwise if similarity_func =='euclidean' else None 
        if verbose != 0:
            self.plot_similarity_histogram()
        self.leader_similarities = None
        


    def check_formats(self, data, varinfo, weights, similarity_func):
        """
        Checks the formats of the input data, varinfo, weights, and similarity function for validity.

        Parameters:
        ----------
        data : pd.DataFrame
            The input dataset.
        varinfo : pd.DataFrame
            DataFrame containing metadata about the variables in the dataset.
        weights : list or np.ndarray, optional
            Weights for each feature in the dataset.
        similarity_func : str
            The similarity function to use, either 'gower' or 'euclidean'.

        Raises:
        ------
        TypeError: If the data, varinfo, or weights are not of the correct type.
        ValueError: If the weights and varinfo lengths do not match, or if the similarity function is invalid.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")
        if not isinstance(varinfo, pd.DataFrame):
            raise TypeError("varinfo must be a pandas.DataFrame")
        if weights is not None:
            if not isinstance(weights, (list, np.ndarray)):
                raise TypeError("weights must be a list or a numpy.ndarray")
            elif len(weights) != varinfo['name'].nunique():
                raise ValueError("weights and varinfo must have the same lengths")
            
        if not isinstance(similarity_func, str) or similarity_func not in ['gower', 'euclidean']:
            raise ValueError("'similarity_func' must be 'gower' or 'euclidean'")
        elif similarity_func == 'euclidean' and ('Binary' in varinfo['type'] or 'Categorical' in varinfo['type']):
            raise ValueError("'simialrity_func' cannot be 'euclidean' because there are Binary or Categorical features in 'varinfo'")
        
        # Extract feature names from varinfo and data
        varinfo_features = varinfo['name'].tolist()
        data_features = data.columns.tolist()

        # Check if both lists match
        if data_features != varinfo_features:
            raise ValueError("data columns and varinfo Features must be equal")


    def check_binary_column(self, data, col_name):
        """
        Checks if a given column in the data is binary, containing only True or False values.

        Parameters:
        ----------
        data : pd.DataFrame
            The input dataset.
        col_name : str
            The name of the column to check.

        Raises:
        ------
        ValueError: If the column contains values other than True or False.
        """
        # Convert the column to a set of unique values and remove NaN if present
        unique_values = set(data[col_name].dropna().unique())
        
        valid_values = {True, False}
        # Check if the unique values are a subset of the valid binary values
        if not unique_values.issubset(valid_values):
            raise ValueError(f"Binary column '{col_name}' expected values {{True, False}}, instead obtained {unique_values}")


    def preprocess_data(self, data, varinfo):
        """
        Preprocesses the input data based on the variable information.

        Parameters:
        ----------
        data : pd.DataFrame
            The input dataset.
        varinfo : pd.DataFrame
            DataFrame containing metadata about the variables in the dataset.

        Returns:
        -------
        data_np : np.ndarray
            The preprocessed data as a numpy array.
        label_encoders : dict
            Dictionary containing label encoders for categorical variables.
        """
        # Replace various representations of missing values with np.nan
        data = data.replace(['nan', 'NaN', np.nan, 'X', '-', '---', 'NA', 'N/A', None], np.nan)  

        label_encoders = {}

        for _, row in varinfo.iterrows():
            col_name = row['name']
            if row['type'] == 'Categorical':
                # Ordinal Encoder for categorical columns
                encoder = OrdinalEncoder(handle_missing='return_nan', handle_unknown='return_nan')
                data[col_name] = encoder.fit_transform(data[col_name])              
                label_encoders[col_name] = encoder
            if row['type'] == 'Binary': 
                self.check_binary_column(data, col_name)
                data[col_name] = data[col_name].map({True:1, False:0}) 


        # Convert the preprocessed data to a numpy array
        data_np = data.to_numpy(dtype=np.float64)
            
        return data_np, label_encoders


    def preprocess_varinfo(self, data, varinfo, weights):
        """
        Preprocesses the variable information and weights for the dataset.

        Parameters:
        ----------
        data : np.ndarray
            The preprocessed data as a numpy array.
        varinfo : pd.DataFrame
            DataFrame containing metadata about the variables in the dataset.
        weights : list or np.ndarray, optional
            Weights for each feature in the dataset.

        Returns:
        -------
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        """
        # Map variable types to integers
        # 0: Binary     1: Categorical     2: Integer     3: Continuous
        types = varinfo['type'].apply(lambda x: 0 if x == 'Binary' else 1 if x== 'Categorical' else 2 if x == 'Integer' else 3 if x == 'Continuous' else -9999).to_numpy(np.int64)
        if np.any(types == -9999):
            raise ValueError("Not supported datatype, only types supported are: Binary, Categorical, Integer and Continuous")

        # Calculate ranges for numerical features
        ranges = np.array([np.nanmax(data[:,i]) - np.nanmin(data[:,i]) if row['type'] in ['Integer', 'Continuous'] else 0 
                        for i, row in varinfo.iterrows()], dtype=np.float64)
        
        # Set weights to ones if not provided
        if weights is None:
            weights = np.ones(len(types))

        if len(types) != len(ranges) or len(types) != len(weights):
            raise ValueError("types, ranges, and weights should have the same length")

        return types, ranges, weights
    
    
    def change_data(self, new_data):
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("new_data must be a pandas.DataFrame")
        self.data, self.label_encoders = self.preprocess_data(new_data, self.varinfo)


    def plot_similarity_histogram(self, subsample=100):
        """
        Plots histograms and a CDF of pairwise and median similarities between data points.

        Parameters:
        ----------
        subsample : int, optional
            Number of data points to use for calculating similarities (default is 100).
        """
        similarities = []
        similarities_median = []
        
        data = self.data.copy()
        np.random.shuffle(data)
        data = data[:subsample]
        sim_matrix = self.similarity_matrix(data)
                
        for i in range(len(data)):
            for j in range(i):
                similarities.append(sim_matrix[i][j])

        for i in range(len(data)):
            similarities_median.append(np.median(sim_matrix[i]))
        
        # Sort the similarities
        sorted_similarities = np.sort(similarities)
        # Calculate the CDF values
        cdf = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)

        fig, ax = plt.subplots(3, 1, figsize= (8, 12))

        # First histogram for pairwise similarities
        ax[0].hist(similarities, bins=30, edgecolor='black')
        ax[0].set_title('Pairwise Similarities Histogram')
        ax[0].set_xlabel('Similarity')
        ax[0].set_ylabel('Frequency')

        # CDF
        ax[1].plot(sorted_similarities, cdf, marker='.', linestyle='none')
        ax[1].set_xlabel('Similarity')
        ax[1].set_ylabel('Cumulative Probability')
        ax[1].set_title('Pairwise Similarities CDF')
        ax[1].grid(True)

        # Second histogram for median similarities
        ax[2].hist(similarities_median, bins=30, edgecolor='black')
        ax[2].set_title('Median Pairwise Similarities Histogram')
        ax[2].set_xlabel('Median Similarity')
        ax[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # Calculate and print statistics
        mean_similarity = np.mean(similarities)
        quartiles = np.percentile(similarities, [25, 50, 75])

        print(f"Mean similarity: {mean_similarity:.2f}")
        print(f"Quartiles: [{quartiles[0]:.2f}  {quartiles[1]:.2f}  {quartiles[2]:.2f}]")
    
    
    @staticmethod
    @njit()
    def _gower_pairwise(x, y, types, ranges, weights):
        """
        Computes the Gower similarity between two data points.

        Parameters:
        ----------
        x : np.ndarray
            First data point.
        y : np.ndarray
            Second data point.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.

        Returns:
        -------
        float
            The Gower similarity between the two data points.
        """
        sim = 0.0
        delta = 0.0

        for i in range(len(types)):
            w = weights[i]
            x_val = x[i]
            y_val = y[i]
                
            if not np.isnan(x_val) and not np.isnan(y_val):
                delta += w
                if types[i] == 0 and x_val == True and y_val == True: # Binary
                    sim += w
                elif types[i] == 1 and x_val == y_val: # Categorical
                        sim += w
                elif types[i] == 2 or types[i] == 3: # Integer or Continuous
                    sim += (1.0 - abs(x_val - y_val) / ranges[i]) * w
                    if np.isnan(sim):
                        print(i, x_val, y_val, ranges[i], w)
        
        if sim/delta > 1:
            raise ValueError("gower sim > 1")

        return sim / delta if delta != 0 else 0.0

    def gower_pairwise(self, x, y, types, ranges, weights):
        """
        Wrapper function for the Gower similarity calculation.

        Parameters:
        ----------
        x : np.ndarray
            First data point.
        y : np.ndarray
            Second data point.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.

        Returns:
        -------
        float
            The Gower similarity between the two data points.
        """
        return self._gower_pairwise(x, y, types, ranges, weights)


    @staticmethod
    @njit(parallel=True)
    def _euclidean_pairwise(x, y, types, ranges, weights):
        """
        Computes the Euclidean similarity between two data points.

        Parameters:
        ----------
        x : np.ndarray
            First data point.
        y : np.ndarray
            Second data point.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.

        Returns:
        -------
        float
            The Euclidean similarity between the two data points.
        """
        eucl_dist = np.sqrt(np.sum(weights * (x - y) ** 2))
        return 1 - eucl_dist / (eucl_dist + 1)
    
    def euclidean_pairwise(self, x, y, types, ranges, weights):
        """
        Wrapper function for the Euclidean similarity calculation.

        Parameters:
        ----------
        x : np.ndarray
            First data point.
        y : np.ndarray
            Second data point.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.

        Returns:
        -------
        float
            The Euclidean similarity between the two data points.
        """
        return self._euclidean_pairwise(x, y, types, ranges, weights)


    @staticmethod
    @njit(parallel=True)
    def _similarity_matrix(X, types, ranges, weights, pairwise_func):
        """
        Computes the similarity matrix for a dataset using a given pairwise similarity function.

        Parameters:
        ----------
        X : np.ndarray
            The dataset as a numpy array.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_func : function
            The pairwise similarity function to use.

        Returns:
        -------
        np.ndarray
            The similarity matrix for the dataset.
        """
        num_rows = len(X)

        out = np.zeros((num_rows, num_rows))

        for i in prange(num_rows):
            for j in range(i):
                sim = pairwise_func(X[i], X[j], types, ranges, weights)
                out[i][j] = sim
                out[j][i] = sim

        np.fill_diagonal(out, 1.0)
        return out

    def similarity_matrix(self, X, types = None, ranges=None, weights=None):
        """
        Computes the similarity matrix for a dataset.

        Parameters:
        ----------
        X : np.ndarray or list
            The dataset as a numpy array or list.
        types : np.ndarray, optional
            Array indicating the type of each feature (default is None, which uses self.types).
        ranges : np.ndarray, optional
            Array containing the range of values for each feature (default is None, which uses self.ranges).
        weights : np.ndarray, optional
            Array containing the weights for each feature (default is None, which uses self.weights).

        Returns:
        -------
        np.ndarray
            The similarity matrix for the dataset.

        Raises:
        ------
        TypeError: If X is not of type 'numpy.ndarray' or 'list'.
        """
        if not isinstance(types, np.ndarray):
            types = self.types
        if not isinstance(ranges, np.ndarray):
            ranges = self.ranges
        if not isinstance(weights, np.ndarray):
            weights = self.weights

        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X must be of type 'numpy.ndarray' or 'list'")
        
        return self._similarity_matrix(X, types, ranges, weights, self.pairwise_similarity)
        



    @staticmethod
    @njit(parallel=True)
    def _Leader(data, s_min, types, ranges, weights, pairwise_function):
        """
        Implements Hartigan's Leader algorithm to cluster data points based on similarity.

        Parameters:
        ----------
        data : np.ndarray
            Array containing the dataset.
        s_min : float
            Minimum similarity threshold for forming clusters.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_function : function
            The pairwise similarity function to use.

        Returns:
        -------
        leaders : list
            Indices of the cluster leaders.
        cluster : np.ndarray
            Array indicating the cluster assignment for each data point.
        avg_leader_sim : list
            Average similarity of data points to their respective cluster leaders.
        cluster_sizes : list
            Sizes of the clusters.
        """
        # Number of data points
        N = len(data)

        # Initialize cluster assignments, leaders, and related statistics
        cluster = np.zeros(N, dtype=np.int32)  # Cluster assignments for each data point
        cluster[0] = 0  # The first data point is assigned to the first cluster
        leaders = [0]  # Indices of cluster leaders
        avg_leader_sim = [1.0]  # Average similarity to cluster leaders
        cluster_sizes = [1]  # Sizes of the clusters

        # Iterate over each data point starting from the second one
        for i in range(1, N):
            found_leader = False  # Flag to check if a suitable leader is found

            # Iterate over existing leaders
            for k in range(len(leaders)):
                leader_idx = leaders[k]  # Index of the current leader
                sim_ik = pairwise_function(data[i], data[leader_idx], types, ranges, weights)
                
                # If similarity is above the threshold, assign to this cluster
                if sim_ik >= s_min:
                    cluster[i] = k  # Assign to the cluster of the leader
                    
                    # Update average similarity and cluster size
                    avg_leader_sim[k] = (sim_ik + avg_leader_sim[k]*cluster_sizes[k]) / (cluster_sizes[k]+1)
                    cluster_sizes[k] += 1

                    found_leader = True  # Suitable leader found
                    break

            # If no suitable leader found, create a new cluster
            if not found_leader:
                leaders.append(i)  # Add the current data point as a new leader

                # Initialize the new average similarity and cluster size
                avg_leader_sim.append(1.0)
                cluster_sizes.append(1)

                cluster[i] = len(leaders) - 1   # Assign to the new cluster

        return leaders, cluster, avg_leader_sim, cluster_sizes
    

    def Leader(self, s_min, data = None, verbose = 1):
        """
        Applies Hartigan's Leader algorithm to the dataset to form clusters based on a similarity threshold.

        Parameters:
        ----------
        s_min : float
            Minimum similarity threshold for forming clusters.
        data : np.ndarray, optional
            Array containing the dataset (default is None, which uses self.data).

        Returns:
        -------
        int
            Number of clusters.
        list
            Indices of the cluster leaders.
        np.ndarray
            Cluster assignment for each data point.
        """
        if data == None:
            data = self.data  # Use class data if no specific data is provided
        self.leader_similarities = None  # Reset leader similarities
        self.s_min = s_min

        start_time = time.time()
        
        # Call the static method to perform clustering
        self.leaders, self.cluster, self.avg_leader_sim, self.cluster_sizes = self._Leader(data, s_min, self.types, self.ranges, self.weights, self.pairwise_similarity)
        
        elapsed_time = time.time() - start_time
        if verbose != 0:
            self.summary_of_statistics("Leader", elapsed_time)  # Print summary statistics
        return len(self.leaders), self.leaders, self.cluster
        

    @staticmethod
    @njit(parallel=True)
    def _Leader2(data, s_min, types, ranges, weights, pairwise_function):
        """
        Implement Jerónimo's modification of Hartigan's Leader algorithm to cluster data points.

        Parameters:
        ----------
        data : np.ndarray
            Array containing the dataset.
        s_min : float
            Minimum similarity threshold for forming clusters.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_function : function
            The pairwise similarity function to use.

        Returns:
        -------
        leaders : list
            Indices of the cluster leaders.
        cluster : np.ndarray
            Array indicating the cluster assignment for each data point.
        avg_leader_sim : list
            Average similarity of data points to their respective cluster leaders.
        cluster_sizes : list
            Sizes of the clusters.
        """
        # Number of data points
        N = len(data)

        # Initialize cluster assignments, leaders, and related statistics
        cluster = np.zeros(N, dtype=np.int32)  # Cluster assignments for each data point
        cluster[0] = 0  # The first data point is assigned to the first cluster
        leaders = [0]  # Indices of cluster leaders
        avg_leader_sim = [1.0]  # Average similarity to cluster leaders
        cluster_sizes = [1]  # Sizes of the clusters

        # Iterate over each data point starting from the second one
        for i in range(1,N):
            # Find the most similar leader for the current data point.
            max_sim, max_leader_idx, max_k = _find_most_similar_leader(i, leaders, data, types, ranges, weights, pairwise_function)
            
            # Assign to the cluster if similarity is above the threshold (s_min)
            if max_sim >= s_min:
                # Assign the current data point to the cluster of the leader with the highest similarity
                cluster[i] = cluster[max_leader_idx] 

                # Update average similarity and cluster size
                avg_leader_sim[max_k] = (max_sim + avg_leader_sim[max_k]*cluster_sizes[max_k]) / (cluster_sizes[max_k]+1)
                cluster_sizes[max_k] += 1

            # Create a new cluster if no suitable leader is found
            else:
                leaders.append(i)  # Add the current data point as a new leader
                # Initialize the new average similarity and cluster size
                avg_leader_sim.append(1.0)
                cluster_sizes.append(1)
                cluster[i] = len(leaders) - 1   # Assign to the new cluster
                
                # Re-evaluate existing clusters to ensure optimal clustering
                for j in range(i):  # Iterate over all previously processed data points
                    if j not in leaders:  # Skip the leaders
                        k = cluster[j]  # Current cluster assignment of data point j

                        # Calculate similarity between current data point i (new leader) and data point j
                        sim_ij = pairwise_function(data[i], data[j], types, ranges, weights)

                        # Calculate similarity between data point j and its current cluster leader
                        sim_j_leader =  pairwise_function(data[j], data[leaders[k]], types, ranges, weights)

                        # Check if data point j is more similar to the new leader (data point i)
                        if sim_ij > sim_j_leader:
                            # Reassign data point j to the new cluster
                            cluster[j] = len(leaders) - 1

                            # Update the average similarity of the current and new cluster
                            avg_leader_sim[k] = (avg_leader_sim[k]*cluster_sizes[k]-sim_ij) / (cluster_sizes[k]-1)  # Current cluster
                            avg_leader_sim[-1] = (sim_ij + avg_leader_sim[-1]*cluster_sizes[-1]) / (cluster_sizes[-1]+1)  # New cluster

                            # Adjust the sizes of the clusters
                            cluster_sizes[k] -= 1
                            cluster_sizes[-1] += 1

        return leaders, cluster, avg_leader_sim, cluster_sizes 
    

    def Leader2(self, s_min, data = None, verbose=1):
        """
        Applies Jerónimo's modification of Hartigan's Leader algorithm to the dataset to form clusters based on a similarity threshold.

        Parameters:
        ----------
        s_min : float
            Minimum similarity threshold for forming clusters.
        data : np.ndarray, optional
            Array containing the dataset (default is None, which uses self.data).

        Returns:
        -------
        int
            Number of clusters.
        list
            Indices of the cluster leaders.
        np.ndarray
            Cluster assignment for each data point.
        """
        if data == None:
            data = self.data  # Use class data if no specific data is provided
        self.leader_similarities = None  # Reset leader similarities
        self.s_min = s_min

        start_time = time.time()
        
        # Call the static method to perform clustering
        self.leaders, self.cluster, self.avg_leader_sim, self.cluster_sizes = self._Leader2(data, s_min, self.types, self.ranges, self.weights, self.pairwise_similarity)
        
        elapsed_time = time.time() - start_time
        if verbose != 0:
            self.summary_of_statistics("Leader2", elapsed_time)  # Print summary statistics
        return len(self.leaders), self.leaders, self.cluster
        

    @staticmethod
    @njit(parallel=False)
    def _Leader_Medoid(data, s_min, types, ranges, weights, pairwise_function, parallel=False):
        """
        Implement the Hartigan's Leader algorithm with medoid adjustment to cluster data points.

        Parameters:
        ----------
        data : np.ndarray
            Array containing the dataset.
        s_min : float
            Minimum similarity threshold for forming clusters.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_function : function
            The pairwise similarity function to use.

        Returns:
        -------
        leaders : list
            Indices of the cluster leaders.
        cluster : np.ndarray
            Array indicating the cluster assignment for each data point.
        avg_leader_sim : list
            Average similarity of data points to their respective cluster leaders.
        cluster_sizes : list
            Sizes of the clusters.
        """
        N = len(data)  # Number of data points

        # Initialize cluster assignments, leaders, average similarities, and related statistics
        cluster = np.zeros(N, dtype=np.int32)  # Cluster assignments for each data point
        cluster[0] = 0  # The first data point is assigned to the first cluster
        avg_sim = np.ones(N, dtype=np.float64)  # Average similarity of each data point to its cluster
        avg_sim[0] = 1.0   # Similarity of the first data point to itself
        leaders = [0]  # Indices of cluster leaders
        avg_leader_sim = [1.0]  # Average similarity to cluster leaders
        cluster_sizes = [1]  # Sizes of the clusters

        # Iterate over each data point starting from the second one
        for i in range(1, N):
            # Find the most similar leader for the current data point.
            max_sim, max_leader_idx, max_k = _find_most_similar_leader(i, leaders, data, types, ranges, weights, pairwise_function)

            # Assign to the cluster if similarity is above the threshold (s_min)
            if max_sim >= s_min:
                # Assign the current data point to the cluster of the leader with the highest similarity
                cluster[i] = cluster[max_leader_idx] 
                
                # Update the cluster leaders and statistics in parallel or sequentially
                if parallel:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)
                else:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)

            # If no suitable leader found, create a new cluster
            else:
                leaders.append(i)  # Add the current data point as a new leader
                # Initialize the new average similarity and cluster size
                avg_leader_sim.append(1.0)
                cluster_sizes.append(1)
                cluster[i] = len(leaders) - 1   # Assign to the new cluster

        return leaders, cluster, avg_leader_sim, cluster_sizes 


    def Leader_Medoid(self, s_min, data = None, parallel=False, verbose=1):
        """
        Wrapper function to execute the _Leader_Medoid algorithm and measure its performance.

        Parameters:
        ----------
        s_min : float
            Minimum similarity threshold for forming clusters.
        data : np.ndarray, optional
            Array containing the dataset (default is None, which uses self.data).

        Returns:
        -------
        int
            Number of clusters.
        list
            Indices of the cluster leaders.
        np.ndarray
            Cluster assignment for each data point.
        """
        if data == None:
            data = self.data  # Use class data if no specific data is provided
        self.leader_similarities = None  # Reset leader similarities
        self.s_min = s_min

        start_time = time.time()
        
        # Call the static method to perform clustering
        self.leaders, self.cluster, self.avg_leader_sim, self.cluster_sizes = self._Leader_Medoid(data, s_min, self.types, self.ranges, self.weights, self.pairwise_similarity, parallel=parallel)
        
        elapsed_time = time.time() - start_time
        if verbose != 0:
            self.summary_of_statistics("Leader Medoid", elapsed_time)
        return len(self.leaders), self.leaders, self.cluster


    @staticmethod
    @njit(parallel=True)
    def _Leader2_Medoid(data, s_min, types, ranges, weights, pairwise_function, parallel=False, merge=False):
        """
        Implement the Jerónimo's Leader algorithm with medoid adjustment to cluster data points.

        Parameters:
        ----------
        data : np.ndarray
            Array containing the dataset.
        s_min : float
            Minimum similarity threshold for forming clusters.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_function : function
            The pairwise similarity function to use.

        Returns:
        -------
        leaders : list
            Indices of the cluster leaders.
        cluster : np.ndarray
            Array indicating the cluster assignment for each data point.
        avg_leader_sim : list
            Average similarity of data points to their respective cluster leaders.
        cluster_sizes : list
            Sizes of the clusters.
        """
        N = len(data)  # Number of data points

        # Initialize cluster assignments, leaders, average similarities, and related statistics
        cluster = np.zeros(N, dtype=np.int32)  # Cluster assignments for each data point
        cluster[0] = 0  # The first data point is assigned to the first cluster
        avg_sim = np.ones(N, dtype=np.float64)  # Average similarity of each data point to its cluster
        avg_sim[0] = 1.0   # Similarity of the first data point to itself
        leaders = [0]  # Indices of cluster leaders
        avg_leader_sim = [1.0]  # Average similarity to cluster leaders
        cluster_sizes = [1]  # Sizes of the clusters

        # Iterate over each data point starting from the second one
        for i in range(1, N):
            # Find the most similar leader for the current data point.
            max_sim, max_leader_idx, max_k = _find_most_similar_leader(i, leaders, data, types, ranges, weights, pairwise_function)

            # Assign to the cluster if similarity is above the threshold (s_min)
            if max_sim >= s_min:
                # Assign the current data point to the cluster of the leader with the highest similarity
                cluster[i] = cluster[max_leader_idx] 
                
                # Update the cluster leaders and statistics in parallel or sequentially
                if parallel:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)
                else:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)

            # If no suitable leader found, create a new cluster
            else:
                new_leader = i  # Create a new leader with the current data point
                new_k = len(leaders)  # Create a new cluster index
                leaders.append(new_leader)  # Add the current data point as a new leader
                # Initialize the new average similarity and cluster size
                avg_leader_sim.append(1.0)
                cluster_sizes.append(1)
                cluster[new_leader] = new_k  # Assign to the new cluster
                # Re-evaluate existing clusters to ensure optimal clustering
                for j in range(i):  # Iterate over all previously processed data points
                    if j not in leaders:  # Skip the leaders 
                        old_k = cluster[j]
                        old_leader = leaders[old_k]

                        # Calculate the similarity between the new leader and the current data point
                        sim_newl_j = pairwise_function(data[new_leader], data[j], types, ranges, weights)
                        # Calculate the similarity between the current data point and its existing leader
                        sim_oldl_j = pairwise_function(data[j], data[old_leader], types, ranges, weights)

                        # If the new leader is more similar to the data point than the old leader
                        if sim_newl_j > sim_oldl_j:
                            # Re-assign the data point to the cluster of the new leader
                            cluster[j] = new_k
                            
                            # Update the cluster leaders and statistics in parallel or sequentially
                            if parallel:
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(j, new_leader, sim_newl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)
                                
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_parallel(j, old_leader, sim_oldl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)
                            else:
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(j, new_leader, sim_newl_j,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)
                                
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_sequential(j, old_leader, sim_oldl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)

        if merge:
            leaders, cluster, avg_leader_sim, cluster_sizes = _merge_clusters(s_min, leaders, cluster, avg_leader_sim, cluster_sizes, avg_sim,
                                                                          data, types, ranges, weights, pairwise_function)

        return leaders, cluster, avg_leader_sim, cluster_sizes 


    def Leader2_Medoid(self, s_min, data = None, parallel=False, merge = False, verbose=1):
        if data == None:
            data = self.data
        
        self.leader_similarities = None
        self.s_min = s_min
        
        start_time = time.time()
        
        self.leaders, self.cluster, self.avg_leader_sim, self.cluster_sizes = self._Leader2_Medoid(data, s_min, self.types, self.ranges, self.weights, self.pairwise_similarity, parallel, merge)
        
        elapsed_time = time.time() - start_time
        if verbose != 0:
            self.summary_of_statistics("Leader2 Medoid", elapsed_time)
        return len(self.leaders), self.leaders, self.cluster



    @staticmethod
    @njit(parallel=True)
    def _Leader3_Medoid(data, s_min, types, ranges, weights, pairwise_function, parallel=False, merge=False, second_pass=False):
        """
        Implement the Jerónimo's Leader algorithm with medoid adjustment to cluster data points.

        Parameters:
        ----------
        data : np.ndarray
            Array containing the dataset.
        s_min : float
            Minimum similarity threshold for forming clusters.
        types : np.ndarray
            Array indicating the type of each feature.
        ranges : np.ndarray
            Array containing the range of values for each feature.
        weights : np.ndarray
            Array containing the weights for each feature.
        pairwise_function : function
            The pairwise similarity function to use.

        Returns:
        -------
        leaders : list
            Indices of the cluster leaders.
        cluster : np.ndarray
            Array indicating the cluster assignment for each data point.
        avg_leader_sim : list
            Average similarity of data points to their respective cluster leaders.
        cluster_sizes : list
            Sizes of the clusters.
        """
        N = len(data)  # Number of data points

        # Initialize cluster assignments, leaders, average similarities, and related statistics
        cluster = np.zeros(N, dtype=np.int32)  # Cluster assignments for each data point
        cluster[0] = 0  # The first data point is assigned to the first cluster
        avg_sim = np.ones(N, dtype=np.float64)  # Average similarity of each data point to its cluster
        avg_sim[0] = 1.0   # Similarity of the first data point to itself
        leaders = [0]  # Indices of cluster leaders
        avg_leader_sim = [1.0]  # Average similarity to cluster leaders
        cluster_sizes = [1]  # Sizes of the clusters

        # Iterate over each data point starting from the second one
        for i in range(1, N):
            # Find the most similar leader for the current data point.
            max_sim, max_leader_idx, max_k = _find_most_similar_leader(i, leaders, data, types, ranges, weights, pairwise_function)

            # Assign to the cluster if similarity is above the threshold (s_min)
            if max_sim >= s_min:
                # Assign the current data point to the cluster of the leader with the highest similarity
                cluster[i] = cluster[max_leader_idx] 
                
                # Update the cluster leaders and statistics in parallel or sequentially
                if parallel:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)
                else:
                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(i, max_leader_idx, max_sim,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)

                # If Leader updated to medoid
                if max_leader_idx != leaders[max_k]:
                    # Re-evaluate existing clusters to ensure optimal clustering
                    for j in range(i):  # Iterate over all previously processed data points
                        if j not in leaders:  # Skip the leaders 
                            old_k = cluster[j]
                            old_leader = leaders[old_k]

                            new_k = max_k
                            new_leader = leaders[new_k]

                            # Calculate the similarity between the new leader and the current data point
                            sim_newl_j = pairwise_function(data[new_leader], data[j], types, ranges, weights)
                            # Calculate the similarity between the current data point and its existing leader
                            sim_oldl_j = pairwise_function(data[j], data[old_leader], types, ranges, weights)

                            # If the new leader is more similar to the data point than the old leader
                            if sim_newl_j > sim_oldl_j:
                                # Re-assign the data point to the cluster of the new leader
                                cluster[j] = new_k
                                
                                # Update the cluster leaders and statistics in parallel or sequentially
                                if parallel:
                                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(j, new_leader, sim_newl_j,
                                                                                            leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                            data, types, ranges, weights, pairwise_function)
                                    
                                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_parallel(j, old_leader, sim_oldl_j,
                                                                                            leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                            data, types, ranges, weights, pairwise_function)
                                else:
                                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(j, new_leader, sim_newl_j,
                                                                                leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                data, types, ranges, weights, pairwise_function)
                                    
                                    leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_sequential(j, old_leader, sim_oldl_j,
                                                                                            leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                            data, types, ranges, weights, pairwise_function)



            # If no suitable leader found, create a new cluster
            else:
                new_leader = i  # Create a new leader with the current data point
                new_k = len(leaders)  # Create a new cluster index
                leaders.append(new_leader)  # Add the current data point as a new leader
                # Initialize the new average similarity and cluster size
                avg_leader_sim.append(1.0)
                cluster_sizes.append(1)
                cluster[new_leader] = new_k  # Assign to the new cluster
                # Re-evaluate existing clusters to ensure optimal clustering
                for j in range(i):  # Iterate over all previously processed data points
                    if j not in leaders:  # Skip the leaders 
                        old_k = cluster[j]
                        old_leader = leaders[old_k]

                        # Calculate the similarity between the new leader and the current data point
                        sim_newl_j = pairwise_function(data[new_leader], data[j], types, ranges, weights)
                        # Calculate the similarity between the current data point and its existing leader
                        sim_oldl_j = pairwise_function(data[j], data[old_leader], types, ranges, weights)

                        # If the new leader is more similar to the data point than the old leader
                        if sim_newl_j > sim_oldl_j:
                            # Re-assign the data point to the cluster of the new leader
                            cluster[j] = new_k
                            
                            # Update the cluster leaders and statistics in parallel or sequentially
                            if parallel:
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_parallel(j, new_leader, sim_newl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)
                                
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_parallel(j, old_leader, sim_oldl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)
                            else:
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _find_medoid_sequential(j, new_leader, sim_newl_j,
                                                                               leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                               data, types, ranges, weights, pairwise_function)
                                
                                leaders, avg_sim, avg_leader_sim, cluster_sizes = _remove_find_medoid_sequential(j, old_leader, sim_oldl_j,
                                                                                        leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes,
                                                                                        data, types, ranges, weights, pairwise_function)

        if merge:
            leaders, cluster, avg_leader_sim, cluster_sizes = _merge_clusters(s_min, leaders, cluster, avg_leader_sim, cluster_sizes, avg_sim,
                                                                          data, types, ranges, weights, pairwise_function)

        if second_pass:
            leaders, cluster, avg_leader_sim, cluster_sizes = _Reassigner_Leader(leaders, cluster, avg_leader_sim, cluster_sizes,
                                                                          data, types, ranges, weights, pairwise_function)

        return leaders, cluster, avg_leader_sim, cluster_sizes 


    def Leader3_Medoid(self, s_min, data = None, parallel=False, merge = False, second_pass=False, verbose=1):
        if data == None:
            data = self.data
        
        self.leader_similarities = None
        self.s_min = s_min
        
        start_time = time.time()
        
        self.leaders, self.cluster, self.avg_leader_sim, self.cluster_sizes = self._Leader3_Medoid(data, s_min, self.types, self.ranges, self.weights, self.pairwise_similarity, parallel, merge, second_pass)
        
        elapsed_time = time.time() - start_time
        if verbose != 0:
            self.summary_of_statistics("Leader3 Medoid", elapsed_time)
        return len(self.leaders), self.leaders, self.cluster   




    def plot_graph(self, figsize=(16, 8), edges_strat='top_n', n=-1):
        data = self.data
        cluster_sizes = self.cluster_sizes
        avg_sim = self.avg_leader_sim
        leaders = self.leaders

        # Create a complete graph
        G = nx.Graph()
        edges_list = []

        # Generate edges with weights based on similarity
        for i in range(len(leaders)):
            for j in range(i):
                weight = self.pairwise_similarity(data[leaders[i]], data[leaders[j]], self.types, self.ranges, self.weights)
                edges_list.append((i, j, weight))
            
            G.add_node(i)

        # Sort edges by weight and select top_n if required
        if edges_strat == 'top_n':
            edges_list.sort(key=lambda x: x[2], reverse=True)  # Sort edges by weight in descending order
            if n == -1 or n > len(edges_list):
                n = len(leaders)
            edges_list = edges_list[:n]
        elif edges_strat == 'all':
            pass

        # Add edges to the graph
        G.add_weighted_edges_from(edges_list)

        if edges_strat == 'top_n' and n == len(leaders):
            # Use a circular layout for equal distribution of nodes
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Create a mask to exclude nodes with cluster_size = 1 from avg_sim calculation
        cluster_size_mask = np.array(cluster_sizes) != 1
        max_avg_sim_excluding_size1 = np.max(np.array(avg_sim)[cluster_size_mask])

        # cmap_nodes = plt.cm.summer
        cmap_nodes = plt.cm.YlOrRd
        cmap_nodes_truncated = mcolors.ListedColormap(cmap_nodes(np.linspace(0.1, 0.9, cmap_nodes.N)))
        norm_nodes = mcolors.Normalize(vmin=self.s_min, vmax=max_avg_sim_excluding_size1)
        
        # Assign colors to nodes
        node_colors = []
        for i in range(len(cluster_sizes)):
            if cluster_sizes[i] == 1:
                node_colors.append('cornflowerblue')  
            else:
                node_colors.append(cmap_nodes_truncated(norm_nodes(avg_sim[i])))

        labels = {k: f"Cluster {k+1}\n{cluster_sizes[k]}" for k in range(len(cluster_sizes))}
        
        # Get edge weights
        edge_weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
        
        # Map the edge weights to colors using a blue colormap
        cmap_edges = plt.cm.summer.reversed()
        # cmap_nodes_truncated = mcolors.ListedColormap(cmap_nodes(np.linspace(0.1, 0.9, cmap_nodes.N)))
        norm_edges = mcolors.Normalize(vmin=np.min(edge_weights), vmax=max(self.s_min, np.max(edge_weights)))
        edge_colors = cmap_edges(norm_edges(edge_weights))
        
        plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_size=5000, node_color=node_colors)
        nx.draw_networkx_labels(G, pos, labels, font_size=14)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=3)
        
        # Add a colorbar to show the mapping from edge weight to color
        sm_edges = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
        sm_edges.set_array([])
        # Explicitly specify the axis for the colorbar
        cbar_edges = plt.colorbar(sm_edges, ax=plt.gca(), label='Similarity between leaders')

        # Add a colorbar to show the mapping from edge weight to color
        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes_truncated, norm=norm_nodes)
        sm_nodes.set_array([])
        # Explicitly specify the axis for the colorbar
        cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), label='Average leader similarity')
        
        plt.axis('off')
        plt.show()


    def plot_clustering_summary(self):
        data = self.data
        cluster_sizes = self.cluster_sizes
        avg_sim = self.avg_leader_sim
        leaders = self.leaders

        # Bars for the bar chart
        bars = [f"Cluster {k + 1}" for k in range(len(cluster_sizes))]

        # Color mapping for the bar chart
        cmap_bars = plt.cm.YlOrRd
        cmap_bars_truncated = mcolors.ListedColormap(cmap_bars(np.linspace(0.1, 0.9, cmap_bars.N)))
        
        # Normalize bars based on average similarity, excluding clusters with size 1
        max_avg_sim_excluding_size1 = np.max(np.array(avg_sim)[np.array(cluster_sizes) > 1])
        norm_bars = mcolors.Normalize(vmin=self.s_min, vmax=max_avg_sim_excluding_size1)

        # Assign colors to bars, use cornflowerblue for bars with size 1
        bars_colors = []
        for i, size in enumerate(cluster_sizes):
            if size == 1:
                bars_colors.append('cornflowerblue')
            else:
                bars_colors.append(cmap_bars_truncated(norm_bars(avg_sim[i])))

        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the bar chart
        ax[0].bar(bars, cluster_sizes, color=bars_colors)
        ax[0].set_xticklabels(bars, rotation=45, ha='right')
        ax[0].set_ylabel('Cluster Lengths')
        ax[0].set_title('Cluster Lengths with leaders Average Similarity Colors')

        # Create a colorbar for the bar chart
        sm = plt.cm.ScalarMappable(cmap=cmap_bars_truncated, norm=norm_bars)
        sm.set_array([])  # Only needed for older versions of Matplotlib
        cbar = plt.colorbar(sm, ax=ax[0])
        cbar.set_label('Leaders Average Similarity')

        # Calculate the inter-cluster similarity matrix
        inter_cluster_sim = self.similarity_matrix(data[leaders])

        # Mask the diagonal
        mask = np.eye(inter_cluster_sim.shape[0], dtype=bool)

        # Find the min and max similarity values excluding the diagonal
        min_sim_non_diag = np.min(inter_cluster_sim[~mask])
        max_sim_non_diag = np.max(inter_cluster_sim[~mask])

        # Normalize based on the actual min and max values in the similarities
        norm_heatmap = mcolors.Normalize(vmin=min_sim_non_diag, vmax=max_sim_non_diag)

        # Plot the heatmap
        heatmap = ax[1].imshow(inter_cluster_sim, cmap=plt.cm.summer.reversed(), norm=norm_heatmap)
        ax[1].set_xlabel('Clusters')
        ax[1].set_ylabel('Clusters')
        ax[1].set_title('Similarity between leaders Heatmap')
        ticks = np.arange(len(leaders)) + 1
        ax[1].set_xticks(np.arange(len(leaders)))
        ax[1].set_yticks(np.arange(len(leaders)))
        ax[1].set_xticklabels(ticks)
        ax[1].set_yticklabels(ticks)


        # Add colorbar for the heatmap
        cbar_heatmap = fig.colorbar(heatmap, ax=ax[1], orientation='vertical')
        cbar_heatmap.set_label('Similarity')

        # Make the diagonal filled with cornflowerblue
        for i in range(inter_cluster_sim.shape[0]):
            ax[1].add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, color='cornflowerblue', lw=0))

        plt.tight_layout()
        plt.show()


    def summary_of_statistics(self, algorithm, elapsed_time):
        print(algorithm.center(40, " "))
        print()
        print("SUMMARY OF STATISTICS".center(40, "="))
        print()
        print(f"Algorithm executed in {elapsed_time:.2f} seconds")
        print()
        print(f"{len(self.leaders)} clusters have been found:")
        print()
        for k in range(len(self.leaders)):
            print(f" Cluster {k + 1}".ljust(15) + f"Leader: {self.leaders[k]}")
            print("".ljust(15) + f"Size: {self.cluster_sizes[k]}")
            print("".ljust(15) + f"Average similarity: {self.avg_leader_sim[k]:.4f}")
            print("-" * 40)


    def cluster_boxplots(self):
        data = self.data
        cluster = self.cluster
        leaders = self.leaders
        types = self.types
        ranges = self.ranges
        weights = self.weights
        
        # Dictionary to store similarity scores for each leader
        leader_similarities = {leader: [] for leader in leaders}
        
        # Compute similarity scores
        for leader in leaders:
            for i in range(len(data)):
                if cluster[leader] == cluster[i] and leader != i:
                    sim = self.pairwise_similarity(data[leader], data[i], types, ranges, weights)
                    leader_similarities[leader].append(sim)
        
        # Prepare data for boxplot
        boxplot_data = [leader_similarities[leader] for leader in leaders]
        
        # Plot the boxplots
        plt.figure(figsize=(10, 6))
        plt.boxplot(boxplot_data, labels=[f"Cluster {i+1}" for i in range(len(leaders))])
        plt.ylabel('Similarity Scores with the leader')
        plt.title('Boxplot of Similarity Scores for Each Leader')
        plt.show()

    
    def simplified_silhouette(self):
        leaders = self.leaders
        data = self.data
        cluster = self.cluster
        s = np.zeros(len(data))

        for i in range(len(data)):
            b_i = 0
            for leader in leaders:
                if cluster[leader] == cluster[i]:
                    a_i = self.pairwise_similarity(data[leader], data[i], self.types, self.ranges, self.weights)
                else:
                    b_i = max(b_i, self.pairwise_similarity(data[leader], data[i], self.types, self.ranges, self.weights))
            s[i] =(a_i - b_i) / max(a_i, b_i)

        return s


    def plot_silhouette(self):
        silhouette_scores = self.simplified_silhouette()
        cluster = self.cluster
        leaders = self.leaders
        silhouette_plot = {}

        # Organize silhouette scores by cluster
        for k in range(len(leaders)):
            silhouette_plot[k] = [silhouette_scores[i] for i in range(len(cluster)) if cluster[i] == k]

        # Sort silhouette_plot by average silhouette score for each cluster
        silhouette_plot = {k: sorted(silhouette_scores) for k, silhouette_scores in silhouette_plot.items()}
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compute y_lower for plotting each cluster
        y_lower = 10
        for k in sorted(silhouette_plot.keys(), reverse=True):  # Iterate from highest cluster index to lowest
            silhouette_scores = silhouette_plot[k]
            size_cluster_k = len(silhouette_scores)
            y_upper = y_lower + size_cluster_k

            color = plt.cm.nipy_spectral(float(k) / len(silhouette_plot))
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, silhouette_scores,
                            facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_k, f'Cluster {k + 1}')

            y_lower = y_upper + 10

        ax.set_title('Silhouette plot')
        ax.set_xlabel('Silhouette coefficient values')
        ax.set_ylabel('Cluster label')

        # The vertical line for average silhouette score of all the values
        silhouette_avg = np.mean([score for scores in silhouette_plot.values() for score in scores])
        ax.axvline(x=silhouette_avg, color='red', linestyle='--')        

        ax.set_yticks([])

        if ax.get_xlim()[0] > -0.05:
            ax.set_xlim(left=-0.1)  # Adjust this value as needed

        plt.show()

        for k in silhouette_plot.keys():
            print(f"Cluster {k + 1} has average silhouette width: {np.mean(silhouette_plot[k]):.2f}")
        
        print(f"For the entire dataset, the average silhouette width is: {silhouette_avg:.2f}")


    def cluster_classes_comparison(self, classes, percentage=True):
        """
        Creates a stacked bar plot with one bar for each cluster. Each segment of the bar represents
        the percentage (or count) of each class within the cluster.

        Parameters:
        classes (list or array): The ground truth class labels.
        percentage (bool): If True, plot percentages; if False, plot absolute values. Default is True.

        Returns:
        None
        """
        if len(self.cluster) != len(classes):
            raise ValueError("The length of clusters and classes must be the same.")
        
        # Create a DataFrame for easier manipulation
        df = pd.DataFrame({'Cluster': self.cluster, 'Class': classes})
        
        # Count the occurrences of each class within each cluster
        count_df = df.groupby(['Cluster', 'Class']).size().unstack(fill_value=0)
        
        if percentage:
            # Calculate the percentage of each class within each cluster
            plot_df = count_df.div(count_df.sum(axis=1), axis=0) * 100
            y_label = 'Percentage'
            title = 'Class Distribution in Each Cluster (Percentage)'
        else:
            # Use absolute values
            plot_df = count_df
            y_label = 'Count'
            title = 'Class Distribution in Each Cluster (Count)'
        
        # Plotting
        clusters = plot_df.index
        classes = plot_df.columns
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot stacked bars for each cluster
        bottom = np.zeros(len(clusters))
        for class_name in classes:
            ax.bar(clusters, plot_df[class_name], bottom=bottom, label=f'Class {class_name}')
            bottom += plot_df[class_name]
        
        # Set labels and title
        ax.set_xlabel('Clusters')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xticks(clusters)
        ax.set_xticklabels([f'Cluster {cluster}' for cluster in clusters])
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Show plot
        plt.tight_layout()
        plt.show()



@njit()
def _find_most_similar_leader(item, leaders, data, types, ranges, weights, pairwise_function):
    max_sim = -1  # Maximum similarity found for current data point
    max_leader_idx = -1  # Index of the leader with the maximum similarity
    max_k = -1  # Cluster index of the leader with the maximum similarity

    # Find the most similar leader for the current data point
    for k in range(len(leaders)):
        leader_idx = leaders[k]
        sim_ik = pairwise_function(data[item], data[leader_idx], types, ranges, weights)

        if sim_ik > max_sim:
            max_sim = sim_ik
            max_leader_idx = leader_idx
            max_k = k
        
    return max_sim, max_leader_idx, max_k



@njit(parallel=True)
def _find_medoid_parallel(item, leader_idx, sim_item_leader, leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes, data, types, ranges, weights, pairwise_function):
    k = cluster[leader_idx]
    # Update the average similarity for the cluster leader
    avg_leader_sim[k] = (sim_item_leader + avg_leader_sim[k]*cluster_sizes[k]) / (cluster_sizes[k]+1)
    avg_sim[leader_idx] = avg_leader_sim[k]
    
    # Initialize the average similarity for the new data point
    avg_sim[item] = 1 + sim_item_leader
    
    # Iterate over all previous data points to update their average similarities
    for j in prange(item):
        # Check if the data point belongs to the same cluster as the current leader and is not the leader itself
        if cluster[j] == cluster[leader_idx] and j != leader_idx:
            # Calculate the similarity between the current data point and another data point in the cluster
            sim_ij = pairwise_function(data[item], data[j], types, ranges, weights)
            # Update the average similarity for the other data point
            avg_sim[j] = (sim_ij + avg_sim[j] * cluster_sizes[k]) /(cluster_sizes[k] +1)

            # Check if this data point should be the new leader (medoid) based on its average similarity
            if avg_sim[j] > avg_leader_sim[k]:
                # Update the leader to the current data point
                leaders[k] = j
                avg_leader_sim[k] = avg_sim[j]

            # Update the average similarity for the new data point
            avg_sim[item] += sim_ij

    # Increment the size of the cluster as a new data point has been added
    cluster_sizes[k] += 1 
    # Finalize the average similarity for the new data point
    avg_sim[item] /= cluster_sizes[k]
    
    # Update leader if current data point has a higher average similarity (therefore is the medoid)
    if avg_sim[item] > avg_leader_sim[k]:
        leaders[k] = item
        avg_leader_sim[k] = avg_sim[item]

    
    return leaders, avg_sim, avg_leader_sim, cluster_sizes  


@njit(parallel=True)
def _remove_find_medoid_parallel(item, leader_idx, sim_item_leader, leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes, data, types, ranges, weights, pairwise_function):
    k = cluster[leader_idx]
    # Update the average similarity for the cluster leader
    avg_leader_sim[k] = (avg_leader_sim[k]*cluster_sizes[k] - sim_item_leader) / (cluster_sizes[k]-1)
    avg_sim[leader_idx] = avg_leader_sim[k]
    
    # Iterate over all previous data points to update their average similarities
    for j in prange(item):
        # Check if the data point belongs to the same cluster as the current leader and is not the leader itself
        if cluster[j] == cluster[leader_idx] and j != leader_idx:
            # Calculate the similarity between the current data point and another data point in the cluster
            sim_ij = pairwise_function(data[item], data[j], types, ranges, weights)
            # Update the average similarity for the other data point
            avg_sim[j] = (avg_sim[j] * cluster_sizes[k] - sim_ij) /(cluster_sizes[k] - 1)

            # Check if this data point should be the new leader (medoid) based on its average similarity
            if avg_sim[j] > avg_leader_sim[k]:
                # Update the leader to the current data point
                leaders[k] = j
                avg_leader_sim[k] = avg_sim[j]

    # Increment the size of the cluster as a new data point has been added
    cluster_sizes[k] -= 1 
    
    return leaders, avg_sim, avg_leader_sim, cluster_sizes


@njit(parallel=False)
def _find_medoid_sequential(item, leader_idx, sim_item_leader, leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes, data, types, ranges, weights, pairwise_function):
    k = cluster[leader_idx]
    # Update the average similarity for the cluster leader
    avg_leader_sim[k] = (sim_item_leader + avg_leader_sim[k]*cluster_sizes[k]) / (cluster_sizes[k]+1)
    avg_sim[leader_idx] = avg_leader_sim[k]
    
    # Initialize the average similarity for the new data point
    avg_sim[item] += sim_item_leader
    
    # Iterate over all previous data points to update their average similarities
    for j in range(item):
        # Check if the data point belongs to the same cluster as the current leader and is not the leader itself
        if cluster[j] == cluster[leader_idx] and j != leader_idx:
            # Calculate the similarity between the current data point and another data point in the cluster
            sim_ij = pairwise_function(data[item], data[j], types, ranges, weights)
            # Update the average similarity for the other data point
            avg_sim[j] = (sim_ij + avg_sim[j] * cluster_sizes[k]) /(cluster_sizes[k] +1)

            # Check if this data point should be the new leader (medoid) based on its average similarity
            if avg_sim[j] > avg_leader_sim[k]:
                # Update the leader to the current data point
                leaders[k] = j
                avg_leader_sim[k] = avg_sim[j]

            # Update the average similarity for the new data point
            avg_sim[item] += sim_ij

    # Increment the size of the cluster as a new data point has been added
    cluster_sizes[k] += 1 
    # Finalize the average similarity for the new data point
    avg_sim[item] /= cluster_sizes[k]
    
    # Update leader if current data point has a higher average similarity (therefore is the medoid)
    if avg_sim[item] > avg_leader_sim[k]:
        leaders[k] = item
        avg_leader_sim[k] = avg_sim[item]

    
    return leaders, avg_sim, avg_leader_sim, cluster_sizes  



@njit(parallel=False)
def _remove_find_medoid_sequential(item, leader_idx, sim_item_leader, leaders, cluster, avg_sim, avg_leader_sim, cluster_sizes, data, types, ranges, weights, pairwise_function):
    k = cluster[leader_idx]
    # Update the average similarity for the cluster leader
    avg_leader_sim[k] = (avg_leader_sim[k]*cluster_sizes[k] - sim_item_leader) / (cluster_sizes[k]-1)
    avg_sim[leader_idx] = avg_leader_sim[k]
    
    # Iterate over all previous data points to update their average similarities
    for j in range(item):
        # Check if the data point belongs to the same cluster as the current leader and is not the leader itself
        if cluster[j] == cluster[leader_idx] and j != leader_idx:
            # Calculate the similarity between the current data point and another data point in the cluster
            sim_ij = pairwise_function(data[item], data[j], types, ranges, weights)
            # Update the average similarity for the other data point
            avg_sim[j] = (avg_sim[j] * cluster_sizes[k] - sim_ij) /(cluster_sizes[k] - 1)

            # Check if this data point should be the new leader (medoid) based on its average similarity
            if avg_sim[j] > avg_leader_sim[k]:
                # Update the leader to the current data point
                leaders[k] = j
                avg_leader_sim[k] = avg_sim[j]

    # Increment the size of the cluster as a new data point has been added
    cluster_sizes[k] -= 1 
    
    return leaders, avg_sim, avg_leader_sim, cluster_sizes  



@njit(parallel=False)
def _merge_clusters(s_min, leaders, cluster, avg_leader_sim, cluster_sizes, avg_sim, data, types, ranges, weights, pairwise_function):
    """
    Merge clusters based on pairwise similarity between cluster leaders and update cluster assignments.

    Parameters:
    -----------
    s_min : float
        Minimum similarity threshold for merging clusters.
    leaders : list
        List of indices representing current cluster leaders.
    cluster : numpy.ndarray
        Array of integers representing cluster assignments for each data point.
    avg_sim : numpy.ndarray
        Array of floats representing average similarity of each data points to their clusters.
    cluster_sizes : list
        List holding the sizes of each cluster.
    avg_leader_sim : list
        List holding the average similarity of data points to their respective cluster leaders.
    data : numpy.ndarray
        Array of data points.
    types : list
        List of types or data characteristics needed for pairwise_function.
    ranges : list
        List of ranges or data scales needed for pairwise_function.
    weights : list
        List of weights for different features or dimensions in the data points.
    pairwise_function : function
        Function to compute pairwise similarity between data points.

    Returns:
    --------
    leaders : list
        Updated list of indices representing cluster leaders.
    cluster : numpy.ndarray
        Updated array of integers representing cluster assignments for each data point.
    avg_leader_sim : list
        Updated list of average similarity of data points to their respective cluster leaders.
    cluster_sizes : list
        Updated list holding the sizes of each cluster.
    """

    N = len(data)
    K = len(leaders)

    # Iterate over the leaders
    for i in range(K):
        for j in range(i):
            leader_i = leaders[i]
            leader_j = leaders[j]

            if leader_i != -1 and leader_j != -1:
                # Check if the similarity between the leaders is above the threshold
                if pairwise_function(data[leader_i], data[leader_j], types, ranges, weights) > s_min:
                    # Find a new leader that maximizes the average similarity to the merged cluster
                    new_leader, new_k = _merge_medoid(N, i, j, leaders, cluster, cluster_sizes, data, types, ranges, weights, pairwise_function)

                    # Update clusters and statistics
                    for l in range(N):
                        if (cluster[l] == cluster[leader_i] or cluster[l] == cluster[leader_j]) and l != leader_i and l != leader_j:
                            if cluster[l] != new_k:
                                sim_l_new_leader = pairwise_function(data[l], data[new_leader], types, ranges, weights)
                                avg_sim[new_leader] = (sim_l_new_leader + avg_sim[new_leader] * cluster_sizes[new_k]) / (cluster_sizes[new_k]+1)
                                cluster_sizes[new_k] += 1
                                cluster[l] = new_k
                    
                    

                    # Update leaders and cluster sizes after merging
                    if cluster[leader_i] == new_k:
                        sim_j_new_leader = pairwise_function(data[leader_j], data[new_leader], types, ranges, weights)
                        avg_sim[new_leader] = (sim_j_new_leader + avg_sim[new_leader] * cluster_sizes[new_k]) / (cluster_sizes[new_k]+1)
                        cluster_sizes[new_k] += 1
                        cluster[leader_j] = new_k

                        leaders[i] = new_leader
                        avg_leader_sim[i] = avg_sim[new_leader]

                        leaders[j] = -1
                        cluster_sizes[j] = -1
                        avg_leader_sim[j] = -1


                    elif cluster[leader_j] == new_k:
                        sim_i_new_leader = pairwise_function(data[leader_i], data[new_leader], types, ranges, weights)
                        avg_sim[new_leader] = (sim_i_new_leader + avg_sim[new_leader] * cluster_sizes[new_k]) / (cluster_sizes[new_k]+1)
                        cluster_sizes[new_k] += 1
                        cluster[leader_i] = new_k

                        leaders[j] = new_leader
                        avg_leader_sim[j] = avg_sim[new_leader]

                        leaders[i] = -1
                        cluster_sizes[i] = -1
                        avg_leader_sim[i] = -1
    
    i = 0
    while i  < len(leaders):
        if leaders[i] == -1:
            leaders.remove(-1)
            cluster_sizes.remove(-1)
            avg_leader_sim.remove(-1)
            for m in range(N):
                cluster[m] = cluster[m] if cluster[m] < i else cluster[m] - 1
        else:
            i += 1
        
    
    return leaders, cluster, avg_leader_sim, cluster_sizes



@njit()
def _merge_medoid(N, i, j, leaders, cluster, cluster_sizes, data, types, ranges, weights, pairwise_function):
    max_sim = -1
    new_leader = -1

    leader_i = leaders[i]
    leader_j = leaders[j]

    for l in range(N):
        if cluster[l] == cluster[leader_i] or cluster[l] == cluster[leader_j]:
            sim_il = pairwise_function(data[leader_i], data[l], types, ranges, weights)
            sim_jl = pairwise_function(data[leader_j], data[l], types, ranges, weights)

            avg_similarity = (sim_il * cluster_sizes[i] + sim_jl * cluster_sizes[j]) / (cluster_sizes[i] + cluster_sizes[j])

            if avg_similarity > max_sim:
                max_sim = avg_similarity
                new_leader = l
    
    new_k = cluster[new_leader]

    return new_leader, new_k



@njit()
def _Reassigner_Leader(leaders, cluster, avg_leader_sim, cluster_sizes, data, types, ranges, weights, pairwise_function):
    # Number of data points
    N = len(data)

    # Re-Initialize cluster assignments, leaders, and related statistics
    avg_leader_sim = [1.0]*len(avg_leader_sim)  # Average similarity to cluster leaders
    cluster_sizes = [1]*len(cluster_sizes)  # Sizes of the clusters

    # Iterate over each data point starting from the second one
    for i in range(1,N):
        # Find the most similar leader for the current data point.
        max_sim, max_leader_idx, max_k = _find_most_similar_leader(i, leaders, data, types, ranges, weights, pairwise_function)
        
        # Assign the current data point to the cluster of the leader with the highest similarity
        cluster[i] = cluster[max_leader_idx] 

        if i not in leaders:
            # Update average similarity and cluster size
            avg_leader_sim[max_k] = (max_sim + avg_leader_sim[max_k]*cluster_sizes[max_k]) / (cluster_sizes[max_k]+1)
            cluster_sizes[max_k] += 1

    return leaders, cluster, avg_leader_sim, cluster_sizes 
