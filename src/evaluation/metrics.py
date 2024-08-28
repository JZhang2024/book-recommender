import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

def calculate_mse(predictions, targets):
    return mean_squared_error(targets, predictions)

def calculate_rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets, predictions))

def calculate_mae(predictions, targets):
    return mean_absolute_error(targets, predictions)

def calculate_ndcg(predictions, targets, k=10):
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(predictions, targets, k):
        assert len(predictions) == len(targets)
        predicted_ranking = np.argsort(predictions)[::-1]
        ideal_ranking = np.argsort(targets)[::-1]
        dcg = dcg_at_k(targets[predicted_ranking], k)
        idcg = dcg_at_k(targets[ideal_ranking], k)
        return dcg / idcg if idcg > 0 else 0

    return np.mean([ndcg_at_k(p, t, k) for p, t in zip(predictions, targets)])

def evaluate_model(model, user_ids, item_ids, ratings):
    model.model.eval()
    with torch.no_grad():
        predictions = model.predict(user_ids, item_ids).numpy()
    
    mse = calculate_mse(predictions, ratings.numpy())
    rmse = calculate_rmse(predictions, ratings.numpy())
    mae = calculate_mae(predictions, ratings.numpy())
    ndcg = calculate_ndcg(predictions, ratings.numpy())
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NDCG@10': ndcg
    }