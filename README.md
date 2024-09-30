# Book Recommendation System

This project implements a collaborative filtering-based book recommendation system using PyTorch. It processes and analyzes the Amazon Books Reviews dataset to provide personalized book recommendations.

## Dataset

The project uses the [Amazon Books Reviews dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data) from Kaggle. This extensive dataset contains millions of book reviews, providing a rich source of information for our recommendation system.

## Project Structure
```
book_recommender/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py
│   │   └── neural_collaborative_filtering.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── preprocess_main.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── matrix_factor_train.py
│   └── neural_collab_train.py
├── data/
│   ├── raw/
│   └── processed/
├── app.py
├── config.py
├── environment.yml
└── README.md
```

## Features

- Data preprocessing and encoding
- Implementation of Matrix Factorization and Neural Collaborative Filtering models using PyTorch
- Training loops with batch processing for both models
- Evaluation metrics including MSE, RMSE, MAE, NDCG@10, Precision@k, and Recall@k
- Support for both CPU and GPU training
- Flask web application for serving recommendations

## Installation

1. Clone this repository:
```
git clone https://github.com/YourUsername/book-recommender.git
cd book-recommender
```

2. Set up the Conda environment:
```
conda env create -f environment.yml
conda activate book_recommender
```

3. Download the dataset from Kaggle and place the CSV files in the `data/raw/` directory.

## Usage

1. Update the `config.py` file with your desired parameters and file paths.

2. Run the preprocessing script to prepare the data:
```
python src/data/preprocess_main.py
```

3. Train the models:
   - For Matrix Factorization:
     ```
     python src/matrix_factor_train.py
     ```
   - For Neural Collaborative Filtering:
     ```
     python src/neural_collab_train.py
     ```

4. Run the Flask application:
```
python app.py
```

## Models

The project implements two collaborative filtering models:

1. Matrix Factorization (`src/models/collaborative_filtering.py`): A simple matrix factorization model that learns latent representations of users and items to predict ratings.

2. Neural Collaborative Filtering (`src/models/neural_collaborative_filtering.py`): A more advanced model that uses neural networks to learn non-linear interactions between users and items.

## Evaluation

The models are evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Normalized Discounted Cumulative Gain (NDCG@10)
- Precision@k
- Recall@k

Evaluation code can be found in `src/evaluation/metrics.py`.

## Web Application

The project includes a Flask web application (`app.py`) that provides an API for adding books to a user's reading list and getting personalized recommendations.

## Future Work

- Implement more advanced models (e.g., transformers-based recommenders)
- Add cross-validation for more robust evaluation
- Develop a user interface for interactive recommendations
- Optimize for larger datasets and faster training
- Implement A/B testing to compare different recommendation algorithms
- Add support for content-based filtering using book metadata