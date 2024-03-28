import json
import pandas as pd
from src import cluster_images, captioning, semantic_clustering, summarize

if __name__ == '__main__':
    # Load the configuration
    with open('data/config.json', 'r') as f:
        config = json.load(f)

    city = config['city']

    # Load the data
    data = pd.read_csv(config['data_path'])

    # Cluster the data
    clustered_images = cluster_images.cluster_data(data, config['min_cluster_sizes'], config['min_samples_values'])
    print("Clustered images successfully.")
    # Execute captioning
    cluster_captioned = captioning.execute_captioning(city, f'../data/{city}/{city}_with_captions.html', photos_path=f'../data/{city}/photos')
    print("Captioned images successfully.")

    # Compute similarity matrix and perform semantic clustering
    df_grouped_by_cluster = clustered_images.groupby('cluster').agg({'caption': '. '.join}).reset_index()
    similarity_func = config['similarity_func']
    sim_matrix = semantic_clustering.compute_sim_matrix(df_grouped_by_cluster, similarity_func)
    threshold = config['threshold']
    sematic_clustered = semantic_clustering.semantic_clustering(df_grouped_by_cluster, sim_matrix, threshold)
    print("Clustered semantically successfully.")
    
    # Generate summaries
    models = config['models']
    prompts = config['prompts']
    output_path = config['output_path']
    n_shots = config['n_shots']
    
    for model_id in models:
        summarized = summarize.summarize(model_id, prompts, sematic_clustered, output_path, n_shots)
        summarized.to_csv(f'{output_path}{model_id}.csv', index=False)
    print("Summarized successfully.")
    
    # Save the results    
    sematic_clustered.to_csv(f'{output_path}semantic_clustered.csv', index=False)
    clustered_images.to_csv(f'{output_path}clustered_images.csv', index=False)
    cluster_captioned.to_csv(f'{output_path}cluster_captioned.csv', index=False)
    