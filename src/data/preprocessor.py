# src/data/preprocessor.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def preprocess_data(books_rating_df, books_data_df, test_size=0.2):
    # Merge ratings with book info
    data = pd.merge(books_rating_df, books_data_df, on='Title')
    
    # Encode categorical variables
    le_user = LabelEncoder()
    le_book = LabelEncoder()
    
    data['user_id_encoded'] = le_user.fit_transform(data['User_id'])
    data['book_id_encoded'] = le_book.fit_transform(data['Id'])
    
    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    return train_data, test_data, le_user, le_book

def create_user_item_matrix(data):
    users = data['user_id_encoded']
    items = data['book_id_encoded']
    ratings = data['review/score']
    shape = (users.max() + 1, items.max() + 1)
    
    return csr_matrix((ratings, (users, items)), shape=shape)
