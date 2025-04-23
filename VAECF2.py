# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from rapidfuzz import process, fuzz
from tqdm import tqdm
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# -------------------------------
# 1. Optimized Data Loading
# -------------------------------
def load_data():
    """Load and preprocess data efficiently"""
    # Load data with optimized parameters
    data = pd.read_csv("./data/data.csv", low_memory=False)
    
    # Load Last.fm data with optimized parameters
    lastfm_df = pd.read_csv(
        './archive/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv',
        sep='\t',
        names=['user', 'timestamp', 'artist_id', 'artist', 'track_id', 'track'],
        quoting=3,
        on_bad_lines='skip',
        engine='c',
        dtype={
            'user': 'str',
            'timestamp': 'str',
            'artist_id': 'str',
            'artist': 'str',
            'track_id': 'str',
            'track': 'str'
        }
    )
    return data, lastfm_df

# -------------------------------
# 2. Parallel Fuzzy Matching
# -------------------------------
def fuzzy_match_wrapper(args):
    """Wrapper function for fuzzy matching that can be pickled"""
    track, valid_song_names, score_cutoff = args
    match = process.extractOne(
        track,
        valid_song_names,
        scorer=fuzz.ratio,
        score_cutoff=score_cutoff
    )
    return track, match[0] if match else None

def batch_fuzzy_match(tracks, valid_song_names, score_cutoff=85, n_workers=None):
    """Parallel fuzzy matching for a batch of tracks using dask"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Create dask dataframe for parallel processing
    df = pd.DataFrame({'track': tracks})
    ddf = dd.from_pandas(df, npartitions=n_workers)
    
    # Define the matching function
    def match_track(row):
        match = process.extractOne(
            row['track'],
            valid_song_names,
            scorer=fuzz.ratio,
            score_cutoff=score_cutoff
        )
        return match[0] if match else None
    
    # Apply matching in parallel
    with ProgressBar():
        results = ddf.apply(match_track, axis=1, meta=('x', 'object')).compute()
    
    return dict(zip(tracks, results))

# -------------------------------
# 3. Efficient Data Processing
# -------------------------------
def process_data(data, lastfm_df):
    """Process data efficiently"""
    # Prepare song names
    data['name_lower'] = data['name'].str.lower().str.strip()
    valid_song_names = list(set(data['name_lower'].values))
    
    # Filter and prepare Last.fm data
    lastfm_df = lastfm_df.dropna(subset=['track'])
    lastfm_df = lastfm_df.copy()
    lastfm_df['track_lower'] = lastfm_df['track'].str.lower().str.strip()
    
    # Get top tracks efficiently
    top_tracks = (
        lastfm_df['track_lower'].value_counts()
        .loc[lambda x: x > 5]
        .head(10000)
        .index.tolist()
    )
    lastfm_df = lastfm_df[lastfm_df['track_lower'].isin(top_tracks)]
    
    return data, lastfm_df, valid_song_names, top_tracks

# -------------------------------
# 4. Main Processing Pipeline
# -------------------------------
def main():
    # Load data
    data, lastfm_df = load_data()
    
    # Process data
    data, lastfm_df, valid_song_names, top_tracks = process_data(data, lastfm_df)
    
    # Check cache
    matched_path = "archive/lastfm_matched_fast.csv"
    if os.path.exists(matched_path):
        lastfm_df = pd.read_csv(matched_path)
        print("✅ Loaded matched results from cache.")
    else:
        print(f"⏳ Fuzzy matching {len(top_tracks)} tracks to song catalog...")
        
        # Parallel fuzzy matching using dask
        match_cache = batch_fuzzy_match(top_tracks, valid_song_names)
        
        # Apply matches
        lastfm_df['matched_name'] = lastfm_df['track_lower'].map(match_cache)
        lastfm_df = lastfm_df.dropna(subset=['matched_name'])
        
        # Save cache
        lastfm_df.to_csv(matched_path, index=False)
        print(f"✅ Saved matched dataset to: {matched_path}")
    
    # Rest of the code remains the same...
    # Filter to top songs
    top_songs = lastfm_df['matched_name'].value_counts().head(5000).index
    lastfm_df = lastfm_df[lastfm_df['matched_name'].isin(top_songs)]
    filtered_data = data[data['name_lower'].isin(top_songs)].copy()
    
    # Build mappings
    song_ids = filtered_data['id'].values
    song_names = filtered_data['name_lower'].values
    song2idx = {name: idx for idx, name in enumerate(song_names)}
    idx2song = {idx: song_id for idx, song_id in enumerate(song_ids)}
    num_items = len(song_ids)
    
    # Select active users
    user_playcounts = lastfm_df['user'].value_counts()
    eligible_users = user_playcounts[user_playcounts >= 10].index
    
    if len(eligible_users) < 100:
        raise ValueError(f"⚠️ Only {len(eligible_users)} users found. Lower threshold or increase dataset size.")
    
    selected_users = np.random.choice(eligible_users, size=100, replace=False)
    user2idx = {user: idx for idx, user in enumerate(selected_users)}
    
    # Build user-song matrix efficiently
    R = np.zeros((len(selected_users), num_items), dtype=np.float32)
    
    # Use groupby for efficient aggregation
    user_song_counts = lastfm_df.groupby(['user', 'matched_name']).size().reset_index()
    for _, row in user_song_counts.iterrows():
        if row['user'] in user2idx and row['matched_name'] in song2idx:
            u = user2idx[row['user']]
            i = song2idx[row['matched_name']]
            R[u, i] = min(row[0], 10)  # Cap at 10 instead of 5
    
    # Train/Validation split
    n_users = R.shape[0]
    n_train = int(0.8 * n_users)
    R_train = R[:n_train, :]
    R_val = R[n_train:, :]
    
    # Normalize ratings
    def normalize_ratings(R):
        mask = R > 0
        R_norm = R.copy()
        R_norm[mask] = (R[mask] - 1) / 9.0  # Changed from 4.0 to 9.0 to normalize 1-10 range
        return R_norm.astype(np.float32) 
    
    R_train_norm = normalize_ratings(R_train)
    R_val_norm = normalize_ratings(R_val)
    
    print("\n✅ Real user-song data matrix created:")
    print(f"Training shape: {R_train.shape}")
    print(f"Validation shape: {R_val.shape}")
    print(f"Sparsity: {(R > 0).mean():.2%}")
    print(f"Rating range: [{R_train_norm[R_train_norm > 0].min():.2f}, {R_train_norm.max():.2f}]")
    
    return R_train_norm, R_val_norm, num_items, idx2song, song2idx

# if __name__ == "__main__":
#     R_train_norm, R_val_norm, song2idx, idx2song = main()

# %%
print("Starting data preprocessing...")
R_train_norm, R_val_norm, num_items, idx2song, song2idx = main()

# 3. Get the number of items from the processed data
print(f"\nPreprocessing completed!")
print(f"Number of items: {num_items}")
print(f"Training data shape: {R_train_norm.shape}")
print(f"Validation data shape: {R_val_norm.shape}")


# %%
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

def kl_anneal_fn(step, cycle_len=1000, beta_max=0.1):
    """KL annealing function to gradually increase KL weight"""
    cycle = step % cycle_len
    return beta_max * tf.sigmoid(8 * (cycle / cycle_len - 0.5))



class MultVAE(tf.keras.Model):
    """Multinomial Variational Autoencoder for Collaborative Filtering"""
    def __init__(self, input_dim, latent_dim=256, hidden_dims=[512, 384], dropout_rate=0.3):
        super(MultVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        
        # Encoder - wider network for better feature extraction
        self.encoder_net = tf.keras.Sequential([
                layers.Dense(512, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(384, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Dense(256, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
            ])
            # VAE specific layers with proper initialization
        self.z_mean = layers.Dense(latent_dim, kernel_initializer='he_normal')
        self.z_log_var = layers.Dense(latent_dim, kernel_initializer='he_normal')
        
        # Decoder with symmetric architecture
        self.decoder_net = tf.keras.Sequential([
            layers.Dense(256, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(384, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(512, activation='gelu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1.5e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(input_dim, activation='linear') 
        ])
    
    def encode(self, x):
        """Encode input to latent space"""
        x = tf.cast(x, tf.float32)
        x = self.encoder_net(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick"""
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        """Decode from latent space to reconstruction"""
        return self.decoder_net(z)
    
    def call(self, x):
        """Forward pass"""
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed
    
    def train_step(self, data):
        """Custom training step with normalized multinomial loss"""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
            
        x = tf.cast(x, tf.float32)
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            reconstructed = self.decode(z)
            
            # Create mask for non-zero elements
            mask = tf.cast(tf.not_equal(x, 0), tf.float32)
            
            # Calculate normalized exposure for proper multinomial loss
            x_norm = x / tf.maximum(tf.reduce_sum(x, axis=1, keepdims=True), 1e-10)
            
            # Multinomial log loss - better for implicit feedback
            log_softmax_var = tf.nn.log_softmax(reconstructed)
            
            # Apply mask to loss calculation
            neg_ll = -tf.reduce_sum(x_norm * log_softmax_var * mask, axis=1)
            
            # Mean loss over batch
            reconstruction_loss = tf.reduce_mean(neg_ll)
            
            # KL divergence
            kl_tolerance = 0.0
            kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            kl = tf.maximum(kl, kl_tolerance)
            kl_loss = tf.reduce_mean(kl)
            
            # Adaptive KL annealing with warm-up
            beta = kl_anneal_fn(self.global_step, cycle_len=1200, beta_max=0.12)
            beta = tf.cast(beta, tf.float32) 
            total_loss = reconstruction_loss + beta * kl_loss
            
        # Get gradients and apply
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.8)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Increment global step
        self.global_step.assign_add(1)
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": beta
        }
    
    def test_step(self, data):
        """Custom test step"""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
            
        x = tf.cast(x, tf.float32)
        
        # Forward pass
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        
        # Create mask for non-zero elements
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)
        
        # Calculate normalized exposure for proper multinomial loss
        x_norm = x / tf.maximum(tf.reduce_sum(x, axis=1, keepdims=True), 1e-10)
        
        # Multinomial log loss
        log_softmax_var = tf.nn.log_softmax(reconstructed)
        neg_ll = -tf.reduce_sum(x_norm * log_softmax_var * mask, axis=1)
        reconstruction_loss = tf.reduce_mean(neg_ll)
        
        # KL divergence
        kl_tolerance = 0.0
        kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl = tf.maximum(kl, kl_tolerance)
        kl_loss = tf.reduce_mean(kl)
        

        
        # Fixed beta for validation
        beta = tf.constant(0.2, dtype=tf.float32)
        total_loss = reconstruction_loss + beta * kl_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

# Function to recommend songs using the trained model
def recommend_songs(model, user_vector, num_items, idx2song, top_n=10):
    """Generate recommendations using the trained model"""
    # Get predictions
    predictions = model(user_vector)
    
    # Get indices of top-N items
    user_ratings = predictions[0].numpy()
    already_liked = np.where(user_vector[0] > 0)[0]
    user_ratings[already_liked] = -np.inf  # Exclude already liked songs
    
    top_indices = np.argsort(user_ratings)[-top_n:][::-1]
    top_scores = user_ratings[top_indices]
    
    # Convert indices to song IDs
    recommended_songs = [idx2song[idx] for idx in top_indices]
    
    return recommended_songs, top_scores

# Training function with learning rate scheduling and warmup
def train_model(model, R_train, R_val, batch_size=64, epochs=300, patience=15):
    """Train the MultVAE model with proper scheduling and callbacks"""
    # Learning rate schedule with warmup
    def lr_schedule(epoch):
        warmup_epochs = 8
        if epoch < warmup_epochs:
            return 1e-4 * (epoch + 1) / warmup_epochs
        return 2e-4 * (0.95 ** (epoch - warmup_epochs))
    
    # Callbacks
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        min_delta=1e-4
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    # Optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.5
    )
    model.compile(optimizer=optimizer)
    
    # Training
    history = model.fit(
        R_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=R_val,
        callbacks=[early_stop, reduce_lr, lr_scheduler],
        verbose=1,
        shuffle=True
    )
    
    return history
    
   

# Example usage (based on your data)

def setup_and_train(R_train_norm, R_val_norm, num_items, idx2song, song2idx, batch_size=32, epochs=200, patience=15, verbose=1):
    """Setup and train the MultVAE model"""
    # Initialize model
    model = MultVAE(input_dim=num_items)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)
    
    # Train model
    history = model.fit(
        R_train_norm,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=R_val_norm,
        verbose=verbose
    )
    
    return model, history

# Generate recommendations for all users
def generate_all_recommendations(model, R_data, idx2song, data, top_n=10, is_train=True):
    """Generate recommendations for all users in the dataset"""
    user_count = R_data.shape[0]
    recommendations = {}
    
    for user_id in range(user_count):
        # Get user vector
        user_vector = R_data[user_id].reshape(1, -1)
        
        # Get already liked songs
        liked_indices = np.where(user_vector[0] > 0)[0]
        
        # Get recommendations
        song_ids, scores = recommend_songs(
            model=model,
            user_vector=user_vector,
            num_items=R_data.shape[1],
            idx2song=idx2song,
            top_n=top_n
        )
        
        # Get song names
        song_names = [data[data['id'] == sid]['name'].values[0] for sid in song_ids]
        
        # For display purposes
        global_user_id = user_id if is_train else (user_id + R_data.shape[0])
        
        recommendations[f"User {global_user_id}"] = {
            'song_ids': song_ids,
            'song_names': song_names,
            'predicted_ratings': scores.tolist()
        }
    
    return recommendations

# Evaluate model performance
def evaluate_model(model, R_val, top_n=10):
    """
    Evaluate model using NDCG@k and Recall@k metrics with proper ground truth handling
    
    Args:
        model: Trained model
        R_val: Validation rating matrix
        top_n: Number of top items to consider
    
    Returns:
        tuple: (recall@k, ndcg@k)
    """
    from sklearn.metrics import ndcg_score
    
    # Get predictions for all users
    predictions = model(R_val).numpy()
    
    # Initialize metrics
    total_recall = 0
    total_ndcg = 0
    num_users = R_val.shape[0]
    
    for user_idx in range(num_users):
        # Get ground truth items (items with non-zero ratings)
        ground_truth = R_val[user_idx]
        relevant_items = np.where(ground_truth > 0)[0]
        
        if len(relevant_items) == 0:
            continue
            
        # Create binary relevance array for all items
        binary_relevance = np.zeros(R_val.shape[1])
        binary_relevance[relevant_items] = 1
        
        # Get predicted scores for this user
        user_scores = predictions[user_idx]
        
        # Calculate NDCG
        # Reshape for sklearn's ndcg_score
        user_scores_2d = user_scores.reshape(1, -1)
        binary_relevance_2d = binary_relevance.reshape(1, -1)
        
        # Calculate metrics
        user_ndcg = ndcg_score(binary_relevance_2d, user_scores_2d, k=top_n)
        
        # Get top predicted items
        top_predicted = np.argsort(user_scores)[-top_n:][::-1]
        
        # Calculate recall
        hits = len(set(top_predicted) & set(relevant_items))
        user_recall = hits / min(len(relevant_items), top_n)
        
        total_recall += user_recall
        total_ndcg += user_ndcg
    
    # Calculate average metrics
    avg_recall = total_recall / num_users
    avg_ndcg = total_ndcg / num_users
    
    return avg_recall, avg_ndcg

def compute_gradient_norm(gradients):
    """
    Compute the L2 norm of the combined gradient vector.
    
    Args:
        gradients: List of gradient tensors/arrays
        
    Returns:
        float: L2 norm of the combined gradient vector
    """
    # Flatten all gradients and concatenate them
    flat_grads = np.concatenate([g.flatten() for g in gradients])
    
    # Compute L2 norm
    l2_norm = np.sqrt(np.sum(flat_grads ** 2))
    
    return l2_norm


# %%
# 1. First, load your data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from rapidfuzz import process, fuzz
from tqdm import tqdm
import os
import time

# 2. Load and preprocess data
 # Add your data loading step here

# 3. Run the preprocessing pipeline
# Add your preprocessing code here to create R_train_norm, R_val_norm, num_items, idx2song, song2idx
gradients = [np.array([[1, 2], [3, 4]]), np.array([5, 6, 7])]
norm = compute_gradient_norm(gradients)
print(f"L2 norm of combined gradients: {norm:.4f}")
# 4. Now run the training with progress display
print("\nStarting model training...")
vae, history = setup_and_train(
    R_train_norm=R_train_norm,
    R_val_norm=R_val_norm, 
    num_items=num_items,
    idx2song=idx2song,
    song2idx=song2idx,
    batch_size=32,
    epochs=200,
    patience=15,
    verbose=1
)

# 5. After training, display results
print("\nTraining completed!")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Check latent space statistics
z_mean = vae.encode(R_train_norm)[0]
z_log_var = vae.encode(R_train_norm)[1]
z_std = np.exp(0.5 * z_log_var)

print("\nLatent Space Statistics:")
print(f"Mean μ range: [{np.min(z_mean):.3f}, {np.max(z_mean):.3f}]")
print(f"Mean σ range: [{np.min(z_std):.3f}, {np.max(z_std):.3f}]")
print(f"Average μ: {np.mean(z_mean):.3f} ± {np.std(z_mean):.3f}")
print(f"Average σ: {np.mean(z_std):.3f} ± {np.std(z_std):.3f}")

data = pd.read_csv("./data/data.csv", low_memory=False)



# 6. Generate recommendations


# print("\nGenerating recommendations...")
# # Generate recommendations for 10 users
# num_users_to_show = 10
# train_recommendations = generate_all_recommendations(
#     vae, R_train_norm, data=data, idx2song=idx2song, top_n=10, is_train=True
# )

# print("\nShowing recommendations for 10 users:")
# for user_idx, (user, recs) in enumerate(list(train_recommendations.items())[:num_users_to_show]):
#     print(f"\nUser {user_idx+1} ({user}) Recommendations:")
#     for i, (song_name, score) in enumerate(zip(recs['song_names'], recs['predicted_ratings']), 1):
#         print(f"{i}. {song_name} (Score: {score:.2f})")

# # 7. Evaluate the model
# print("\nEvaluating model performance...")
# recall, ndcg = evaluate_model(vae, R_val_norm, top_n=10)
# print(f"Recall@10: {recall:.4f}")
# print(f"NDCG@10: {ndcg:.4f}")

# # %%

# import matplotlib.pyplot as plt

# def plot_training_history(history):
#     # Set figure style
#     plt.style.use('default')  # Use default style instead of seaborn
    
#     # Create a figure with subplots
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
#     # Plot total loss
#     ax1.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
#     ax1.plot(history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
#     ax1.set_title('Total Loss Over Time', fontsize=12, pad=10)
#     ax1.set_xlabel('Epoch', fontsize=10)
#     ax1.set_ylabel('Loss', fontsize=10)
#     ax1.legend(fontsize=10)
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot reconstruction loss
#     ax2.plot(history.history['reconstruction_loss'], 'b-', 
#              label='Training Reconstruction Loss', linewidth=2)
#     ax2.plot(history.history['val_reconstruction_loss'], 'r--',
#              label='Validation Reconstruction Loss', linewidth=2)
#     ax2.set_title('Reconstruction Loss Over Time', fontsize=12, pad=10)
#     ax2.set_xlabel('Epoch', fontsize=10)
#     ax2.set_ylabel('Reconstruction Loss', fontsize=10)
#     ax2.legend(fontsize=10)
#     ax2.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot KL loss
#     ax3.plot(history.history['kl_loss'], 'b-', label='Training KL Loss', linewidth=2)
#     ax3.plot(history.history['val_kl_loss'], 'r--', label='Validation KL Loss', linewidth=2)
#     ax3.set_title('KL Divergence Loss Over Time', fontsize=12, pad=10)
#     ax3.set_xlabel('Epoch', fontsize=10)
#     ax3.set_ylabel('KL Loss', fontsize=10)
#     ax3.legend(fontsize=10)
#     ax3.grid(True, linestyle='--', alpha=0.7)
    
#     # Adjust layout and display
#     plt.tight_layout()
#     plt.show()
    
#     # Print final loss values
#     print("\nFinal Loss Values:")
#     print(f"Training Loss: {history.history['loss'][-1]:.4f}")
#     print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
#     print(f"Training Reconstruction Loss: {history.history['reconstruction_loss'][-1]:.4f}")
#     print(f"Validation Reconstruction Loss: {history.history['val_reconstruction_loss'][-1]:.4f}")
#     print(f"Training KL Loss: {history.history['kl_loss'][-1]:.4f}")
#     print(f"Validation KL Loss: {history.history['val_kl_loss'][-1]:.4f}")

# # Create a log-scale version of the plot
# def plot_training_history_log_scale(history):
#     plt.style.use('default')
    
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
#     # Plot total loss
#     ax1.semilogy(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
#     ax1.semilogy(history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
#     ax1.set_title('Total Loss Over Time (Log Scale)', fontsize=12, pad=10)
#     ax1.set_xlabel('Epoch', fontsize=10)
#     ax1.set_ylabel('Loss (log scale)', fontsize=10)
#     ax1.legend(fontsize=10)
#     ax1.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot reconstruction loss
#     ax2.semilogy(history.history['reconstruction_loss'], 'b-',
#                  label='Training Reconstruction Loss', linewidth=2)
#     ax2.semilogy(history.history['val_reconstruction_loss'], 'r--',
#                  label='Validation Reconstruction Loss', linewidth=2)
#     ax2.set_title('Reconstruction Loss Over Time (Log Scale)', fontsize=12, pad=10)
#     ax2.set_xlabel('Epoch', fontsize=10)
#     ax2.set_ylabel('Reconstruction Loss (log scale)', fontsize=10)
#     ax2.legend(fontsize=10)
#     ax2.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot KL loss
#     ax3.semilogy(history.history['kl_loss'], 'b-', label='Training KL Loss', linewidth=2)
#     ax3.semilogy(history.history['val_kl_loss'], 'r--', label='Validation KL Loss', linewidth=2)
#     ax3.set_title('KL Divergence Loss Over Time (Log Scale)', fontsize=12, pad=10)
#     ax3.set_xlabel('Epoch', fontsize=10)
#     ax3.set_ylabel('KL Loss (log scale)', fontsize=10)
#     ax3.legend(fontsize=10)
#     ax3.grid(True, linestyle='--', alpha=0.7)
    
#     plt.tight_layout()
#     plt.show()

# # Plot both regular and log-scale versions
# print("Regular Scale:")
# plot_training_history(history)
# print("\nLogarithmic Scale:")
# plot_training_history_log_scale(history)

# # Optional: Save the plots
# plt.savefig('vae_training_history.png', dpi=300, bbox_inches='tight')




