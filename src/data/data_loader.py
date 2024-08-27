import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class BookDataset(Dataset):
    def __init__(self, ratings_file, books_file):
        self.ratings = pd.read_csv(ratings_file)
        self.books = pd.read_csv(books_file)
        
        # Merge ratings with book info
        self.data = pd.merge(self.ratings, self.books, on='Title')
        
        # Create user and book mappings
        self.user_ids = self.data['User_id'].unique()
        self.book_ids = self.data['Id'].unique()
        
        self.user_to_index = {id: i for i, id in enumerate(self.user_ids)}
        self.book_to_index = {id: i for i, id in enumerate(self.book_ids)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'user': torch.tensor(self.user_to_index[row['User_id']], dtype=torch.long),
            'book': torch.tensor(self.book_to_index[row['Id']], dtype=torch.long),
            'rating': torch.tensor(row['review/score'], dtype=torch.float),
            'title': row['Title'],
            'authors': row['authors'],
            'price': torch.tensor(row['Price'], dtype=torch.float)
        }

def load_data(ratings_file, books_file, batch_size=64):
    dataset = BookDataset(ratings_file, books_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)