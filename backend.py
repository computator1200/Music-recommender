# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Text
import os

# Load the datasets
track_data = pd.read_csv('data/data.csv')
genre_data = pd.read_csv('data/data_by_genres.csv')
year_data = pd.read_csv('data/data_by_year.csv')
artist_genre_data = pd.read_csv('data/data_w_genres.csv')

# Preview the main track data
print(f"Track data shape: {track_data.shape}")
track_data.head()

# %%
# Handle missing values
track_data = track_data.fillna(0)

# Convert categorical features to strings for embedding
track_data['year'] = track_data['year'].astype(str)
track_data['key'] = track_data['key'].astype(str)
track_data['mode'] = track_data['mode'].astype(str)

# Process the artists column (converts from string representation of list to actual list)
track_data['artists'] = track_data['artists'].apply(lambda x: eval(x) if isinstance(x, str) else x)
track_data['artist_main'] = track_data['artists'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')

# Create a function to extract audio features
def extract_audio_features(row):
    return {
        'acousticness': row['acousticness'],
        'danceability': row['danceability'],
        'duration_ms': row['duration_ms'] / 1000000,  # Normalize
        'energy': row['energy'],
        'instrumentalness': row['instrumentalness'],
        'liveness': row['liveness'],
        'loudness': (row['loudness'] + 60) / 60,  # Normalize to 0-1 range (assuming range is -60 to 0)
        'speechiness': row['speechiness'],
        'tempo': row['tempo'] / 200,  # Normalize assuming max tempo of 200
        'valence': row['valence'],
        'popularity': row['popularity'] / 100,  # Normalize to 0-1
        'year': row['year'],
        'key': row['key'],
        'mode': row['mode']
    }

# Create a unique track identifier
track_data['track_id'] = track_data['id']

# Get unique artists and genres for vocabulary sizing
unique_artists = track_data['artist_main'].unique()
unique_years = track_data['year'].unique()

# Process genre data
artist_genres = {}
for _, row in artist_genre_data.iterrows():
    if isinstance(row['genres'], str) and row['genres'] != '[]':
        genres = eval(row['genres'])
        if isinstance(genres, list):
            artist_genres[row['artists']] = genres

# Map artists to their genres
def get_artist_genres(artist):
    for artist_name, genres in artist_genres.items():
        if artist in artist_name:
            return genres
    return []

track_data['genres'] = track_data['artist_main'].apply(get_artist_genres)

# Convert to TensorFlow Dataset
def create_tf_dataset(dataframe, is_training=True):
    # Extract features
    features = {
        'track_id': dataframe['track_id'].values,
        'artist': dataframe['artist_main'].values,
        'name': dataframe['name'].values,
        'year': dataframe['year'].values,
        'key': dataframe['key'].values,
        'mode': dataframe['mode'].values,
        'acousticness': dataframe['acousticness'].values.astype(np.float32),
        'danceability': dataframe['danceability'].values.astype(np.float32),
        'duration_ms': (dataframe['duration_ms'] / 1000000).values.astype(np.float32),
        'energy': dataframe['energy'].values.astype(np.float32),
        'instrumentalness': dataframe['instrumentalness'].values.astype(np.float32),
        'liveness': dataframe['liveness'].values.astype(np.float32),
        'loudness': ((dataframe['loudness'] + 60) / 60).values.astype(np.float32),
        'speechiness': dataframe['speechiness'].values.astype(np.float32),
        'tempo': (dataframe['tempo'] / 200).values.astype(np.float32),
        'valence': dataframe['valence'].values.astype(np.float32),
        'popularity': (dataframe['popularity'] / 100).values.astype(np.float32),
    }
    
    dataset = tf.data.Dataset.from_tensor_slices(features)
    
    if is_training:
        dataset = dataset.shuffle(len(dataframe))
    
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Split data into train and test
train_df = track_data.sample(frac=0.8, random_state=42)
test_df = track_data.drop(train_df.index)

# Create TensorFlow datasets
train_dataset = create_tf_dataset(train_df)
test_dataset = create_tf_dataset(test_df, is_training=False)

# %%
@tf.keras.utils.register_keras_serializable(package="Custom")
class MusicModel(tfrs.Model):
    def __init__(
        self,
        unique_artists,
        unique_years,
        embedding_dim=128
    ):
        super().__init__()
        
        # Vocabulary sizes
        self.artist_vocab_size = len(unique_artists) + 1  # +1 for OOV tokens
        self.year_vocab_size = len(unique_years) + 1
        self.key_vocab_size = 12 + 1  # 0-11 plus OOV
        self.mode_vocab_size = 2 + 1  # 0-1 plus OOV
        
        # Embedding dimensions
        self.artist_embed_dim = 64
        self.year_embed_dim = 32
        self.key_embed_dim = 8
        self.mode_embed_dim = 4
        
        # Store vocabularies for lookup layers
        self.unique_artists = unique_artists
        self.unique_years = unique_years
        
        # Create query and song models
        self.song_model = self._build_tower(embedding_dim)
        self.query_model = self._build_tower(embedding_dim)
        
        # Store embedding_dim for later use
        self.embedding_dim = embedding_dim
    
    def _build_tower(self, embedding_dim):
        # Define input specs explicitly
        inputs = {
            'track_id': tf.keras.layers.Input(shape=(), dtype=tf.string, name='track_id'),
            'artist': tf.keras.layers.Input(shape=(), dtype=tf.string, name='artist'),
            'name': tf.keras.layers.Input(shape=(), dtype=tf.string, name='name'),
            'year': tf.keras.layers.Input(shape=(), dtype=tf.string, name='year'),
            'key': tf.keras.layers.Input(shape=(), dtype=tf.string, name='key'),
            'mode': tf.keras.layers.Input(shape=(), dtype=tf.string, name='mode'),
            'acousticness': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='acousticness'),
            'danceability': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='danceability'),
            'duration_ms': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='duration_ms'),
            'energy': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='energy'),
            'instrumentalness': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='instrumentalness'),
            'liveness': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='liveness'),
            'loudness': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='loudness'),
            'speechiness': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='speechiness'),
            'tempo': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='tempo'),
            'valence': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='valence'),
            'popularity': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='popularity'),
        }
        
        # Process string inputs with lookup layers - ensure proper OOV handling
        artist_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.unique_artists, num_oov_indices=1)(inputs['artist'])
        year_lookup = tf.keras.layers.StringLookup(
            vocabulary=self.unique_years, num_oov_indices=1)(inputs['year'])
        key_lookup = tf.keras.layers.StringLookup(
            vocabulary=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], 
            num_oov_indices=1)(inputs['key'])
        mode_lookup = tf.keras.layers.StringLookup(
            vocabulary=['0', '1'], num_oov_indices=1)(inputs['mode'])
        
        # Embedding layers
        artist_embedding = tf.keras.layers.Embedding(
            self.artist_vocab_size, self.artist_embed_dim)(artist_lookup)
        year_embedding = tf.keras.layers.Embedding(
            self.year_vocab_size, self.year_embed_dim)(year_lookup)
        key_embedding = tf.keras.layers.Embedding(
            self.key_vocab_size, self.key_embed_dim)(key_lookup)
        mode_embedding = tf.keras.layers.Embedding(
            self.mode_vocab_size, self.mode_embed_dim)(mode_lookup)
        
        # Flatten embeddings
        artist_embedding = tf.keras.layers.Flatten()(artist_embedding)
        year_embedding = tf.keras.layers.Flatten()(year_embedding)
        key_embedding = tf.keras.layers.Flatten()(key_embedding)
        mode_embedding = tf.keras.layers.Flatten()(mode_embedding)
        
        # Concatenate numerical features
        numerical_features = tf.keras.layers.Concatenate()([
            inputs['acousticness'], 
            inputs['danceability'], 
            inputs['duration_ms'],
            inputs['energy'], 
            inputs['instrumentalness'], 
            inputs['liveness'],
            inputs['loudness'], 
            inputs['speechiness'], 
            inputs['tempo'],
            inputs['valence'], 
            inputs['popularity']
        ])
        
        # Concatenate all features
        all_features = tf.keras.layers.Concatenate()([
            artist_embedding, year_embedding, key_embedding, mode_embedding, numerical_features
        ])
        
        # Dense layers
        x = tf.keras.layers.Dense(256, activation='relu')(all_features)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(embedding_dim)(x)
        
        # Return the model
        return tf.keras.Model(inputs=inputs, outputs=output)
    
    def compute_loss(self, features, training=False):
        # Get embeddings
        query_embeddings = self.query_model(features)
        song_embeddings = self.song_model(features)
        
        # Compute similarity scores between queries and candidates
        scores = tf.matmul(query_embeddings, tf.transpose(song_embeddings))
        
        # Each query should match with its corresponding candidate
        # (i.e., diagonal of the similarity matrix should be high)
        labels = tf.range(tf.shape(scores)[0])
        
        # Compute standard categorical cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, scores, from_logits=True
        )
        
        # Return the mean loss
        return tf.reduce_mean(loss)
    
    def integrate_audio_latent(self, audio_latent_vector, song_features):
        """
        Integrates the audio latent vector with song features
        
        Args:
            audio_latent_vector: The latent vector from audioToLatentVector(audio)
            song_features: The song features from our dataset
            
        Returns:
            Enhanced song embedding
        """
        # Get the song embedding from existing features
        song_embedding = self.song_model(song_features)
        
        # Convert audio latent vector to appropriate tensor
        audio_latent_tensor = tf.convert_to_tensor(audio_latent_vector, dtype=tf.float32)
        
        # Fusion layer (simple concatenation with projection to embedding_dim)
        fusion_layer = tf.keras.layers.Dense(self.embedding_dim)
        fused_embedding = fusion_layer(tf.concat([song_embedding, audio_latent_tensor], axis=1))
        
        return fused_embedding

# %%
# Create and train the model
embedding_dim = 128
model = MusicModel(unique_artists, unique_years, embedding_dim)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

# Evaluate on the test set
metrics = model.evaluate(test_dataset, return_dict=True)
print(f"Evaluation metrics: {metrics}")

model.save('models/metadata_two_tower_deep_retrieval.keras')

# %%
# Create a mapping from string IDs to integer IDs
track_ids = track_data['track_id'].unique()
id_to_int = {id: i for i, id in enumerate(track_ids)}
int_to_id = {i: id for i, id in enumerate(track_ids)}

# Create an index for fast nearest neighbor lookup
index = tfrs.layers.factorized_top_k.BruteForce()

# Create the index with explicit batch processing
# First create a dataset with all songs
songs_dataset = tf.data.Dataset.from_tensor_slices({
    'track_id': track_data['track_id'].values,
    'artist': track_data['artist_main'].values,
    'name': track_data['name'].values,
    'year': track_data['year'].values,
    'key': track_data['key'].values,
    'mode': track_data['mode'].values,
    'acousticness': track_data['acousticness'].values.astype(np.float32),
    'danceability': track_data['danceability'].values.astype(np.float32),
    'duration_ms': (track_data['duration_ms'] / 1000000).values.astype(np.float32),
    'energy': track_data['energy'].values.astype(np.float32),
    'instrumentalness': track_data['instrumentalness'].values.astype(np.float32),
    'liveness': track_data['liveness'].values.astype(np.float32),
    'loudness': ((track_data['loudness'] + 60) / 60).values.astype(np.float32),
    'speechiness': track_data['speechiness'].values.astype(np.float32),
    'tempo': (track_data['tempo'] / 200).values.astype(np.float32),
    'valence': track_data['valence'].values.astype(np.float32),
    'popularity': (track_data['popularity'] / 100).values.astype(np.float32),
})

# Create a separate dataset with just the IDs
ids_dataset = tf.data.Dataset.from_tensor_slices(
    np.array([id_to_int[id] for id in track_data['track_id'].values], dtype=np.int64)
)

# Process in batches
batch_size = 1024
songs_batches = songs_dataset.batch(batch_size)
ids_batches = ids_dataset.batch(batch_size)

# Get embeddings for each batch
embeddings_batches = songs_batches.map(model.song_model)

# Index using the prepared datasets
index.index_from_dataset(
    tf.data.Dataset.zip((ids_batches, embeddings_batches))
)

# Function to get recommendations
def get_recommendations(song_id, k=10):
    try:
        # Find the song by ID
        song = track_data[track_data['track_id'] == song_id].iloc[0]
        song_id_int = id_to_int[song_id]
        
        # Create a tensor of the song's features with correct shapes
        song_features = {
            'track_id': np.array([song['track_id']]),
            'artist': np.array([song['artist_main']]),
            'name': np.array([song['name']]),
            'year': np.array([song['year']]),
            'key': np.array([song['key']]),
            'mode': np.array([song['mode']]),
            # Reshape numeric features to have shape (batch_size, 1)
            'acousticness': np.array([[song['acousticness']]], dtype=np.float32),
            'danceability': np.array([[song['danceability']]], dtype=np.float32),
            'duration_ms': np.array([[song['duration_ms'] / 1000000]], dtype=np.float32),
            'energy': np.array([[song['energy']]], dtype=np.float32),
            'instrumentalness': np.array([[song['instrumentalness']]], dtype=np.float32),
            'liveness': np.array([[song['liveness']]], dtype=np.float32),
            'loudness': np.array([[(song['loudness'] + 60) / 60]], dtype=np.float32),
            'speechiness': np.array([[song['speechiness']]], dtype=np.float32),
            'tempo': np.array([[song['tempo'] / 200]], dtype=np.float32),
            'valence': np.array([[song['valence']]], dtype=np.float32),
            'popularity': np.array([[song['popularity'] / 100]], dtype=np.float32),
        }
        
        # Remove 'track_id_int' as it's not expected by the model
        # Get the query embedding
        query_embedding = model.query_model(song_features)
        
        # Get recommendations
        scores, song_id_ints = index(query_embedding, k=k+1)
        
        # Skip the first one as it would be the query song itself
        recommended_song_id_ints = song_id_ints[0][1:]
        
        # Get the recommended songs
        recommended_songs = []
        for int_id in recommended_song_id_ints.numpy():
            original_id = int_to_id[int(int_id)]
            if original_id in track_data['track_id'].values:
                recommended_song = track_data[track_data['track_id'] == original_id].iloc[0]
                recommended_songs.append({
                    'track_id': original_id,
                    'name': recommended_song['name'],
                    'artist': recommended_song['artist_main'],
                    'year': recommended_song['year'],
                    'popularity': recommended_song['popularity']
                })
        
        return recommended_songs
    except IndexError:
        print(f"Song with ID {song_id} not found in track data")
        return []
# %%
# Assuming we have a function audioToLatentVector that returns latent vectors
def audioToLatentVector(audio_file_path):
    """
    Placeholder function for the audio to latent vector conversion.
    In a real implementation, this would use the provided audio processing functionality.
    
    Returns:
        A mock latent vector of dimension 64
    """
    # Mock implementation - in reality, this would call your audio processing function
    return np.random.normal(0, 1, 64)

# Example of how to use the audio latent vector with our model
def recommend_with_audio(audio_file_path, k=10):
    # Get the audio latent vector
    audio_latent = audioToLatentVector(audio_file_path)
    
    # For this example, we'll find a random song to use as base features
    random_song_idx = np.random.randint(0, len(track_data))
    song = track_data.iloc[random_song_idx]
    song_id_int = id_to_int[song['track_id']]
    
    # Create a tensor of the song's features
    song_features = {
        'track_id': np.array([song['track_id']]),
        'track_id_int': np.array([song_id_int], dtype=np.int64),  # Add integer ID
        'artist': np.array([song['artist_main']]),
        'name': np.array([song['name']]),
        'year': np.array([song['year']]),
        'key': np.array([song['key']]),
        'mode': np.array([song['mode']]),
        'acousticness': np.array([song['acousticness']], dtype=np.float32),
        'danceability': np.array([song['danceability']], dtype=np.float32),
        'duration_ms': np.array([song['duration_ms'] / 1000000], dtype=np.float32),
        'energy': np.array([song['energy']], dtype=np.float32),
        'instrumentalness': np.array([song['instrumentalness']], dtype=np.float32),
        'liveness': np.array([song['liveness']], dtype=np.float32),
        'loudness': np.array([(song['loudness'] + 60) / 60], dtype=np.float32),
        'speechiness': np.array([song['speechiness']], dtype=np.float32),
        'tempo': np.array([song['tempo'] / 200], dtype=np.float32),
        'valence': np.array([song['valence']], dtype=np.float32),
        'popularity': np.array([song['popularity'] / 100], dtype=np.float32),
    }
    
    # Integrate audio latent vector with song features
    enhanced_query_embedding = model.integrate_audio_latent(
        audio_latent_vector=audio_latent, 
        song_features=song_features
    )
    
    # Get recommendations based on the enhanced query
    scores, song_id_ints = index(enhanced_query_embedding, k=k)
    
    # Get the recommended songs
    recommended_songs = []
    for int_id in song_id_ints[0].numpy():
        original_id = int_to_id[int(int_id)]
        if original_id in track_data['track_id'].values:
            recommended_song = track_data[track_data['track_id'] == original_id].iloc[0]
            recommended_songs.append({
                'track_id': original_id,
                'name': recommended_song['name'],
                'artist': recommended_song['artist_main'],
                'year': recommended_song['year'],
                'popularity': recommended_song['popularity']
            })
    
    return recommended_songs

# Example usage
# audio_path = "path/to/audio/file.mp3"  # In a real scenario, this would be a path to an audio file
# audio_recommendations = recommend_with_audio(audio_path, k=5)
# print(f"Recommendations based on audio:")
# for i, rec in enumerate(audio_recommendations, 1):
#     print(f"{i}. {rec['name']} by {rec['artist']} ({rec['year']}) - Popularity: {rec['popularity']}")

# %%
# This is a placeholder for Spotify API integration
def get_spotify_track_features(track_id):
    """
    Get track features from Spotify API
    
    Args:
        track_id: Spotify track ID
        
    Returns:
        Track features dictionary
    """
    # In a real implementation, this would use the Spotify API
    # import spotipy
    # from spotipy.oauth2 import SpotifyClientCredentials
    
    # sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    #     client_id="YOUR_CLIENT_ID",
    #     client_secret="YOUR_CLIENT_SECRET"))
    
    # features = sp.audio_features(track_id)[0]
    # return features
    
    # Mock return
    return {
        'acousticness': 0.5,
        'danceability': 0.7,
        'duration_ms': 200000,
        'energy': 0.6,
        'instrumentalness': 0.1,
        'liveness': 0.2,
        'loudness': -8.0,
        'speechiness': 0.05,
        'tempo': 120.0,
        'valence': 0.8
    }

def recommend_from_spotify_track(spotify_track_id, k=10):
    """
    Get recommendations based on a Spotify track
    
    Args:
        spotify_track_id: Spotify track ID
        k: Number of recommendations to return
        
    Returns:
        List of recommended tracks
    """
    # Get track features from Spotify
    spotify_features = get_spotify_track_features(spotify_track_id)
    
    # Create a feature dictionary for the model
    features = {
        'track_id': np.array(['spotify_' + spotify_track_id]),
        'artist': np.array(['unknown']),  # We might get this from Spotify API
        'name': np.array(['unknown']),    # We might get this from Spotify API
        'year': np.array(['2020']),       # Mock value
        'key': np.array(['5']),           # Mock value
        'mode': np.array(['1']),          # Mock value
        'acousticness': np.array([spotify_features['acousticness']], dtype=np.float32),
        'danceability': np.array([spotify_features['danceability']], dtype=np.float32),
        'duration_ms': np.array([spotify_features['duration_ms'] / 1000000], dtype=np.float32),
        'energy': np.array([spotify_features['energy']], dtype=np.float32),
        'instrumentalness': np.array([spotify_features['instrumentalness']], dtype=np.float32),
        'liveness': np.array([spotify_features['liveness']], dtype=np.float32),
        'loudness': np.array([(spotify_features['loudness'] + 60) / 60], dtype=np.float32),
        'speechiness': np.array([spotify_features['speechiness']], dtype=np.float32),
        'tempo': np.array([spotify_features['tempo'] / 200], dtype=np.float32),
        'valence': np.array([spotify_features['valence']], dtype=np.float32),
        'popularity': np.array([0.5], dtype=np.float32),  # Mock value
    }
    
    # Get the query embedding
    query_embedding = model.query_model(features)
    
    # Get recommendations
    scores, song_ids = index(query_embedding, k=k)
    
    # Get the recommended songs
    recommended_songs = []
    for song_id in song_ids[0].numpy():
        song_id_str = song_id.decode('utf-8')
        if song_id_str in track_data['track_id'].values:
            recommended_song = track_data[track_data['track_id'] == song_id_str].iloc[0]
            recommended_songs.append({
                'track_id': song_id_str,
                'name': recommended_song['name'],
                'artist': recommended_song['artist_main'],
                'year': recommended_song['year'],
                'popularity': recommended_song['popularity']
            })
    
    return recommended_songs

# Example usage
# spotify_track_id = "4BJqT0PrAfrxzMOxytFOIz"  # Example Spotify track ID
# spotify_recommendations = recommend_from_spotify_track(spotify_track_id, k=5)
# print(f"Recommendations from Spotify track:")
# for i, rec in enumerate(spotify_recommendations, 1):
#     print(f"{i}. {rec['name']} by {rec['artist']} ({rec['year']}) - Popularity: {rec['popularity']}")


# Create a server loop to handle requests from the frontend

import socket
import json
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Server started.")

# create a socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(5)
logging.info("Server is listening for connections...")

def handle_client(client_socket):
    try:
        # Receive data from the client
        request = client_socket.recv(1024).decode('utf-8')
        logging.info(f"Received request: {request}")
        
        # Parse the request
        request_data = json.loads(request)
        action = request_data.get('action')
        
        if action == 'recommend':
            song_id = request_data.get('song_id')
            k = request_data.get('k', 10)
            recommendations = get_recommendations(song_id, k)
            response = {
                'status': 'success',
                'recommendations': recommendations
            }
        elif action == 'recommend_with_audio':
            audio_file_path = request_data.get('audio_file_path')
            k = request_data.get('k', 10)
            recommendations = recommend_with_audio(audio_file_path, k)
            response = {
                'status': 'success',
                'recommendations': recommendations
            }
        elif action == 'recommend_from_spotify':
            spotify_track_id = request_data.get('spotify_track_id')
            k = request_data.get('k', 10)
            recommendations = recommend_from_spotify_track(spotify_track_id, k)
            response = {
                'status': 'success',
                'recommendations': recommendations
            }
        else:
            response = {
                'status': 'error',
                'message': 'Invalid action'
            }
        
        # Send the response back to the client
        client_socket.send(json.dumps(response).encode('utf-8'))
    except Exception as e:
        logging.error(f"Error handling client: {e}")
    finally:
        client_socket.close()
        logging.info("Client connection closed.")
        logging.info("Waiting for new connections...")
        # Close the client socket
        client_socket.close()

while True:
    try:
        # Accept a new client connection
        client_socket, addr = server_socket.accept()
        logging.info(f"Accepted connection from {addr}")
        
        # Handle the client in a separate thread or process
        handle_client(client_socket)
    except KeyboardInterrupt:
        logging.info("Server shutting down.")
        break
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        # Close the server socket
        server_socket.close()
        logging.info("Server socket closed.")
        logging.info("Server stopped.")
    
