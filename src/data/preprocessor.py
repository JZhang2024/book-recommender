import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(ratings_df, books_df, test_size=0.2):
    # Merge ratings with book info
    data = pd.merge(ratings_df, books_df, on='book_id')
    
    # Encode categorical variables
    le_user = LabelEncoder()
    le_book = LabelEncoder()
    
    data['user_id'] = le_user.fit_transform(data['user_id'])
    data['book_id'] = le_book.fit_transform(data['book_id'])
    
    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    return train_data, test_data, le_user, le_book

def create_user_item_matrix(data):
    return data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)