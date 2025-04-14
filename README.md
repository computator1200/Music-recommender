# Music recommender system
A music recommender system focused on providing accurate and personalised recommendations.

## Techniques
- **Collaborative filtering**: Uses user-item interactions to recommend items based on similar users' preferences. Implemented using the Autoencoder architecture to reconstruct the user-item interaction matrix.

- **Content-based filtering**: Recommends items based on the features of the items themselves. Implemented using the Two-Tower Deep Retrieval model to embed songs into a latent space and compute the K-nearest neighbours for a query.

## Technologies used

- **Python**: The primary programming language used for the implementation.

- **TensorFlow**: A deep learning framework used for building and training the models.

- **TensorFlow Recommenders**: A library built on top of TensorFlow for building recommender systems.
- **Pandas**: A data manipulation library used for data preprocessing and analysis.

- **NumPy**: A library for numerical computations in Python, used for handling arrays and matrices.

- **Streamlit**: A web application framework used for building interactive frontend for the recommender system.
