import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from tqdm import tqdm

def calculate_mse(predictions, targets):
    return mean_squared_error(targets, predictions)

def calculate_rmse(predictions, targets):
    return np.sqrt(mean_squared_error(targets, predictions))

def calculate_mae(predictions, targets):
    return mean_absolute_error(targets, predictions)

def calculate_ndcg(predictions, targets, k=10):
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.

    predictions = np.array(predictions)
    targets = np.array(targets)

    assert predictions.shape == targets.shape

    # Sort predictions and targets in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]

    # Calculate DCG and IDCG
    dcg = dcg_at_k(sorted_targets, k)
    idcg = dcg_at_k(np.sort(targets)[::-1], k)

    # Avoid division by zero
    if idcg == 0:
        return 0.

    return dcg / idcg

def evaluate_model(model, dataloader, device):
    model.model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            user_ids, item_ids, ratings = [x.to(device) for x in batch]
            batch_predictions = model.predict(user_ids, item_ids)
            predictions.extend(batch_predictions.cpu().numpy())
            targets.extend(ratings.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = calculate_mse(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    mae = calculate_mae(predictions, targets)
    ndcg = calculate_ndcg(predictions, targets)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NDCG@10': ndcg
    }