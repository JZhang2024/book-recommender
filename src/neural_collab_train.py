import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from scipy import sparse
from .models.neural_collaborative_filtering import NeuralCollaborativeFiltering
from .evaluation.metrics import evaluate_model
import sys
import os
from tqdm import tqdm

# Add the project root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from config import (TRAIN_DATA_FILE, TEST_DATA_FILE, USER_ITEM_MATRIX_FILE, 
                    USER_ENCODER_FILE, BOOK_ENCODER_FILE, 
                    EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
def load_data():
    print("Loading data...")
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    test_data = pd.read_csv(TEST_DATA_FILE)
    
    # Reduce dataset size for testing
    train_data = train_data.sample(frac=0.05, random_state=42)
    test_data = test_data.sample(frac=0.05, random_state=42)
    
    user_item_matrix = sparse.load_npz(USER_ITEM_MATRIX_FILE)
    user_encoder = pd.read_pickle(USER_ENCODER_FILE)
    book_encoder = pd.read_pickle(BOOK_ENCODER_FILE)
    print("Data loaded successfully.")

    return train_data, test_data, user_item_matrix, user_encoder, book_encoder

def train_model(model, train_loader, test_loader, device, num_epochs, patience=5):
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        # Training
        model.model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_pbar:
            user_ids, item_ids, ratings = [x.to(device) for x in batch]
            loss = model.train_step(user_ids, item_ids, ratings)
            train_loss += loss
            train_pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        train_loss /= len(train_loader)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Evaluation
        print("Evaluating...")
        metrics = evaluate_model(model, test_loader, device)
        val_loss = metrics['MSE']
        
        print("Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            print("Saving best model...")
            torch.save(model.model.state_dict(), 'best_ncf_model.pth')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print("Loading best model...")
    model.model.load_state_dict(torch.load('best_ncf_model.pth'))
    return model

def main():
    train_data, test_data, user_item_matrix, user_encoder, book_encoder = load_data()

    num_users = len(user_encoder.classes_)
    num_items = len(book_encoder.classes_)
    print(f"Number of users: {num_users}, Number of items: {num_items}")

    print("Preparing tensors...")
    train_user_ids = torch.LongTensor(train_data['user_id_encoded'].values)
    train_item_ids = torch.LongTensor(train_data['book_id_encoded'].values)
    train_ratings = torch.FloatTensor(train_data['review/score'].values)

    test_user_ids = torch.LongTensor(test_data['user_id_encoded'].values)
    test_item_ids = torch.LongTensor(test_data['book_id_encoded'].values)
    test_ratings = torch.FloatTensor(test_data['review/score'].values)
    print("Tensors prepared.")

    # Create datasets
    train_dataset = TensorDataset(train_user_ids, train_item_ids, train_ratings)
    test_dataset = TensorDataset(test_user_ids, test_item_ids, test_ratings)
    print("Datasets created.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("DataLoaders created.")

    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = NeuralCollaborativeFiltering(num_users, num_items, EMBEDDING_DIM, [128, 64, 32, 16], LEARNING_RATE).to(device)
    print("Model initialized.")

    print("Starting training loop...")
    model = train_model(model, train_loader, test_loader, device, NUM_EPOCHS)

    print("Training completed. Performing final evaluation...")
    final_metrics = evaluate_model(model, test_loader, device)
    print("Final Test Metrics:")
    for metric_name, metric_value in final_metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')
    
    return model

if __name__ == "__main__":
    main()