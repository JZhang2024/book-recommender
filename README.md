# Book Recommendation System

This project implements a collaborative filtering-based book recommendation system using PyTorch. It processes and analyzes the Amazon Books Reviews dataset to provide personalized book recommendations.

## Dataset

The project uses the [Amazon Books Reviews dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data) from Kaggle. This extensive dataset contains millions of book reviews, providing a rich source of information for our recommendation system.

## Project Structure
```
book_recommender/
├── src/
│   ├── models/
│   │   ├── init.py
│   │   └── collaborative_filtering.py
|   ├── data/
│   │   ├── init.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── preprocess_main.py
│   ├── evaluation/
│   │   ├── init.py
│   │   └── metrics.py
│   └── train.py
├── data/
│   ├── raw/
│   └── processed/
├── config.py
└── requirements.txt
```

## Features

- Data preprocessing and encoding
- Implementation of a Matrix Factorization model using PyTorch
- Training loop with batch processing
- Evaluation metrics including MSE, RMSE, MAE, and NDCG@10
- Support for both CPU and GPU training

## Installation

1. Clone this repository:
```
git clone https://github.com/JZhang2024/book-recommender.git
cd book-recommender
```
1. Install the required packages:
```
pip install -r requirements.txt
```
1. Download the dataset from Kaggle and place the CSV files in the `data/raw/` directory.

## Usage

1. Update the `config.py` file with your desired parameters and file paths.

2. Run the preprocessing script to prepare the data:
```
python src/preprocess_main.py
```
3. Train the model:
```
python src/train.py
```

## Model

The current implementation uses a simple Matrix Factorization model, which can be found in `src/models/collaborative_filtering.py`. This model learns latent representations of users and items to predict ratings.

## Evaluation

The model is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Normalized Discounted Cumulative Gain (NDCG@10)

Evaluation code can be found in `src/evaluation/metrics.py`.

## Future Work

- Implement more advanced models (e.g., Neural Collaborative Filtering)
- Add cross-validation for more robust evaluation
- Develop a user interface for interactive recommendations
- Optimize for larger datasets and faster training