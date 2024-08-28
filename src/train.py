import torch
import pandas as pd
from scipy import sparse
from models.collaborative_filtering import CollaborativeFiltering
from evaluation.metrics import evaluate_model
import sys
import os

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from config import (TRAIN_DATA_FILE, TEST_DATA_FILE, USER_ITEM_MATRIX_FILE, 
                    USER_ENCODER_FILE, BOOK_ENCODER_FILE, 
                    EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)

def load_data():
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    test_data = pd.read_csv(TEST_DATA_FILE)
    user_item_matrix = sparse.load_npz(USER_ITEM_MATRIX_FILE)
    user_encoder = pd.read_pickle(USER_ENCODER_FILE)
    book_encoder = pd.read_pickle(BOOK_ENCODER_FILE)
    
    return train_data, test_data, user_item_matrix, user_encoder, book_encoder

def main():
    train_data, test_data, user_item_matrix, user_encoder, book_encoder = load_data()
    
    num_users = len(user_encoder.classes_)
    num_items = len(book_encoder.classes_)
    
    model = CollaborativeFiltering(num_users, num_items, EMBEDDING_DIM, LEARNING_RATE)
    
    train_user_ids = torch.LongTensor(train_data['user_id_encoded'].values)
    train_item_ids = torch.LongTensor(train_data['book_id_encoded'].values)
    train_ratings = torch.FloatTensor(train_data['review/score'].values)
    
    test_user_ids = torch.LongTensor(test_data['user_id_encoded'].values)
    test_item_ids = torch.LongTensor(test_data['book_id_encoded'].values)
    test_ratings = torch.FloatTensor(test_data['review/score'].values)
    
    for epoch in range(NUM_EPOCHS):
        model.train(train_user_ids, train_item_ids, train_ratings, 1)  # Train for 1 epoch
        
        # Evaluate on test set
        metrics = evaluate_model(model, test_user_ids, test_item_ids, test_ratings)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value:.4f}')
        print()

    # Final evaluation
    final_metrics = evaluate_model(model, test_user_ids, test_item_ids, test_ratings)
    print("Final Test Metrics:")
    for metric_name, metric_value in final_metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')

if __name__ == "__main__":
    main()