import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import argparse

def load_model(model_path):
    """
    Loads a pre-trained embedding model and returns it
    """
    model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model

def get_word_embedding(model, word, unknown_words):
    """
    Retrieves the embedding vector of the provided word.
    """
    try:
        return model[word]
    except KeyError:
        unknown_words.append(word)
        return []

def plot_embeddings(embeddings, words, args):
    """
    Plot of the 2D embeddings using Plotly.
    """
    # Perform clustering on the reduced embeddings
    kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create a DataFrame for the embeddings, words, and clusters
    df = pd.DataFrame(embeddings, columns=['x', 'y'])
    df['word'] = words
    df['cluster'] = cluster_labels

    # Create the scatter plot
    fig = px.scatter(df, x='x', y='y', hover_name='word', color='cluster',
                     title=args.title, template=args.theme)

    fig.update_layout(coloraxis_showscale=False)
    
    # Add text labels
    if args.labels == True:
        for i, row in df.iterrows():
            fig.add_annotation(x=row['x'], y=row['y'], text=row['word'],
                               showarrow=False, yshift=10)

    # Update traces and layout for better readability
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(hovermode='closest', showlegend=True)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_html(args.output)


def plot_embeddings_3d(embeddings, words, args):
    """
    Plot of the 3D embeddings using Plotly.
    """
    # Perform clustering on the reduced embeddings
    kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)

    # Create a DataFrame for the embeddings, words, and clusters
    df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
    df['word'] = words
    df['cluster'] = cluster_labels

    # Create the 3D scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='word', color='cluster',
                        title=args.title, template=args.theme)
    
    fig.update_layout(coloraxis_showscale=False)

    # Add text labels if required
    if args.labels == True:
        for i, row in df.iterrows():
            fig.add_trace(go.Scatter3d(x=[row['x']], y=[row['y']], z=[row['z']], 
                                       mode='text', text=[row['word']],
                                       textposition='middle center',
                                       showlegend=False))

    # Update traces and layout for better readability
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(hovermode='closest', showlegend=True)
    fig.update_layout(scene=dict(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False))
    fig.write_html(args.output)


def reduce_dimensions(embeddings, method, dims):
    """
    Reduce the dimensions of embeddings to 2D or 3D.

    Parameters:
    embeddings (array): High-dimensional embeddings.
    method (str): Dimensionality reduction method ('pca' or 'tsne').
    dims (int): number of dimensions to reduce embeddings to. 
    """
    n_samples = embeddings.shape[0]
    if method == 'pca':
        reducer = PCA(n_components=dims)
    elif method == 'tsne':
        # Adjust perplexity for small datasets
        perplexity = min(30, max(n_samples // 3, 5))
        reducer = TSNE(n_components=dims, perplexity=perplexity, random_state=0)
    else:
        raise ValueError("Invalid method: choose 'pca' or 'tsne'")

    return reducer.fit_transform(embeddings)

def main():
    parser = argparse.ArgumentParser(description='Word Embeddings Visualization Tool')
    parser.add_argument('-m', '--model', required=True, help='Path to the word embeddings model')
    parser.add_argument('-i', '--input', required=True, help='Input text file with words')
    parser.add_argument('-o', '--output', required=True, help='Output HTML file for the visualization')
    parser.add_argument('-l', '--labels', action='store_true', help='Show labels on plot (default: False)')
    parser.add_argument('-c', '--clusters', type=int, default=5, help='Number of clusters for KMeans (default: 5)')
    parser.add_argument('-d', '--dimensions', type=int, default=2, help='Number of in the plot, "2" for 2D, "3" for 3D (default: 2)')
    parser.add_argument('-r', '--reduction', choices=['pca', 'tsne'], default='tsne', help='Method for dimensionality reduction (default: tsne)')
    parser.add_argument('-t', '--title', required=False, default='Word Embeddings Visualization', help='Set the title for the plot')
    parser.add_argument('-th', '--theme', required=False, default='plotly', help='color theme for the plot: "plotly", "plotly_white" or "plotly_dark" (default: plotly)')
    args = parser.parse_args()

    model = load_model(args.model)
    embeddings = []
    unknown_words = []
    wordlist = []

    with open(args.input, 'r') as file:
        for line in file:
            line = line.strip()
            line = line.split()
            wordlist.extend(line)

    for word in wordlist:
        embed = get_word_embedding(model, word, unknown_words)
        if len(embed) > 1:
            embeddings.append(embed)
        else:
            print(f"Word '{word}' not found in model")

    words = [word for word in wordlist if word not in unknown_words]
    embeddings = np.array(embeddings)
    embeddings_d = reduce_dimensions(embeddings, args.reduction, args.dimensions)
    if args.dimensions == 3:
        plot_embeddings_3d(embeddings_d, words, args)
    else:
        plot_embeddings(embeddings_d, words, args)
  

if __name__ == "__main__":
    main()
