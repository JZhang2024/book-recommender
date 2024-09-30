from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import torch
import numpy as np
from src.models.neural_collaborative_filtering import NeuralCollaborativeFiltering

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///book_recommender.db'
db = SQLAlchemy(app)

# Load the trained model
model = NeuralCollaborativeFiltering.load_model('path/to/your/trained_model.pth')
model.eval()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    author = db.Column(db.String(100))
    isbn = db.Column(db.String(13), unique=True)

class ReadingList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)

@app.route('/add_to_reading_list', methods=['POST'])
def add_to_reading_list():
    data = request.json
    user_id = data['user_id']
    book_id = data['book_id']
    
    reading_list_item = ReadingList(user_id=user_id, book_id=book_id)
    db.session.add(reading_list_item)
    db.session.commit()
    
    return jsonify({"message": "Book added to reading list"}), 200

@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    
    # Get the user's reading list
    reading_list = ReadingList.query.filter_by(user_id=user_id).all()
    
    # Use the model to generate recommendations
    # This is a placeholder - you'll need to implement the actual recommendation logic
    recommendations = generate_recommendations(user_id, reading_list)
    
    return jsonify(recommendations)

def generate_recommendations(user_id, reading_list, model, book_encoder, top_n=10):
    # Convert reading list to model input format
    book_ids = [item.book_id for item in reading_list]
    encoded_book_ids = [book_encoder.transform([book_id])[0] for book_id in book_ids]
    
    # Generate candidate set (all books not in reading list)
    all_book_ids = set(book_encoder.classes_)
    candidate_book_ids = list(all_book_ids - set(book_ids))
    encoded_candidate_ids = [book_encoder.transform([book_id])[0] for book_id in candidate_book_ids]
    
    # Prepare input for the model
    user_ids = torch.LongTensor([user_id] * len(encoded_candidate_ids))
    item_ids = torch.LongTensor(encoded_candidate_ids)
    
    # Get predictions from the model
    with torch.no_grad():
        predictions = model.predict(user_ids, item_ids)
    
    # Sort predictions and get top N
    top_n_indices = np.argsort(predictions.numpy())[-top_n:][::-1]
    top_n_book_ids = [candidate_book_ids[i] for i in top_n_indices]
    
    # Fetch book details from the database
    recommended_books = Book.query.filter(Book.id.in_(top_n_book_ids)).all()
    
    return recommended_books

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)