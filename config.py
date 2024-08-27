
import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Data file paths
RATINGS_FILE = os.path.join(RAW_DATA_DIR, 'ratings.csv')
BOOKS_FILE = os.path.join(RAW_DATA_DIR, 'books.csv')

# Processed data file paths
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
USER_ITEM_MATRIX_FILE = os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.csv')

# Model parameters
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# Evaluation metrics
TOP_K = 10  # For precision@k and recall@k

# Flask app settings (if you decide to create a web interface)
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000