import torch
import pandas as pd
from scipy import sparse
from src.models.collaborative_filtering import CollaborativeFiltering
from config import (TRAIN_DATA_FILE, USER_ITEM_MATRIX_FILE, 
                    USER_ENCODER_FILE, BOOK_ENCODER_FILE, 
                    EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)

def load_data():
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    user_item_matrix = sparse.load_npz(USER_ITEM_MATRIX_FILE)
    user_encoder = pd.read_pickle(USER_ENCODER_FILE)
    book_encoder = pd.read_pickle(BOOK_ENCODER_FILE)
    
    return train_data, user_item_matrix, user_encoder, book_encoder

def main():
    train_data, user_item_matrix, user_encoder, book_encoder = load_data()
    
    num_users = len(user_encoder.classes_)
    num_items = len(book_encoder.classes_)
    
    model = CollaborativeFiltering(num_users, num_items, EMBEDDING_DIM, LEARNING_RATE)
    
    user_ids = torch.LongTensor(train_data['user_id_encoded'].values)
    item_ids = torch.LongTensor(train_data['book_id_encoded'].values)
    ratings = torch.FloatTensor(train_data['review/score'].values)
    
    model.train(user_ids, item_ids, ratings, NUM_EPOCHS)
    
    # TODO: Add model evaluation and saving

if __name__ == "__main__":
    main()