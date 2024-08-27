# src/data/preprocess_main.py

import pandas as pd
import numpy as np
from scipy import sparse
from config import (BOOKS_RATING_FILE, BOOKS_DATA_FILE, TRAIN_DATA_FILE, 
                    TEST_DATA_FILE, USER_ITEM_MATRIX_FILE, 
                    USER_ENCODER_FILE, BOOK_ENCODER_FILE)
from src.data.preprocessor import preprocess_data, create_user_item_matrix

def main():
    # Load raw data
    books_rating_df = pd.read_csv(BOOKS_RATING_FILE)
    books_data_df = pd.read_csv(BOOKS_DATA_FILE)
    
    # Preprocess data
    train_data, test_data, user_encoder, book_encoder = preprocess_data(books_rating_df, books_data_df)
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(train_data)
    
    # Save processed data
    train_data.to_csv(TRAIN_DATA_FILE, index=False)
    test_data.to_csv(TEST_DATA_FILE, index=False)
    
    # Save user-item matrix in a sparse format
    sparse.save_npz(USER_ITEM_MATRIX_FILE, user_item_matrix)
    
    print("Data preprocessing completed.")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"User-Item matrix shape: {user_item_matrix.shape}")
    print(f"User-Item matrix non-zero entries: {user_item_matrix.nnz}")

    # Save encoders
    pd.to_pickle(user_encoder, USER_ENCODER_FILE)
    pd.to_pickle(book_encoder, BOOK_ENCODER_FILE)

if __name__ == "__main__":
    main()