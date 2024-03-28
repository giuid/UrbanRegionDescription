import hdbscan
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

def cluster_data(data, min_cluster_sizes, min_samples_values):

    # Standardize the data
    data = StandardScaler().fit_transform(data)

    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

            clusterer.fit(data)

            labels = clusterer.labels_

            results = pd.DataFrame(data, columns=['latitude', 'longitude'])
            results['cluster_id'] = labels

            results.to_csv(f'cluster_results_{min_cluster_size}_{min_samples}.csv', index=False)

            #print(f"The clustering results with min_cluster_size={min_cluster_size} and min_samples={min_samples} were saved to 'cluster_results_{min_cluster_size}_{min_samples}.csv'.")
