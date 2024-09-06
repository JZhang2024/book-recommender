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

    ndcgs = []
    for user_preds, user_targets in zip(predictions, targets):
        # Sort predictions and targets in descending order of predictions
        sorted_indices = np.argsort(user_preds)[::-1]
        user_preds = user_preds[sorted_indices]
        user_targets = user_targets[sorted_indices]

        dcg = dcg_at_k(user_targets, k)
        idcg = dcg_at_k(np.sort(user_targets)[::-1], k)

        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return np.mean(ndcgs)

def precision_at_k(predictions, targets, k=10):
    # Sort predictions in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    # Get top k predictions
    top_k_indices = sorted_indices[:k]
    # Count number of relevant items in top k
    num_relevant = np.sum(targets[top_k_indices] > 0)
    return num_relevant / k

def recall_at_k(predictions, targets, k=10):
    # Sort predictions in descending order
    sorted_indices = np.argsort(predictions)[::-1]
    # Get top k predictions
    top_k_indices = sorted_indices[:k]
    # Count number of relevant items in top k
    num_relevant_in_top_k = np.sum(targets[top_k_indices] > 0)
    # Count total number of relevant items
    total_relevant = np.sum(targets > 0)
    return num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0

def evaluate_model(model, dataloader, device, k=10):
    model.model.eval()
    all_predictions = []
    all_targets = []
    user_item_pairs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            user_ids, item_ids, ratings = [x.to(device) for x in batch]
            batch_predictions = model.predict(user_ids, item_ids)
            all_predictions.extend(batch_predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
            user_item_pairs.extend(zip(user_ids.cpu().numpy(), item_ids.cpu().numpy()))

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    mse = calculate_mse(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    mae = calculate_mae(predictions, targets)

    # Group predictions and targets by user
    user_predictions = {}
    user_targets = {}
    for (user, item), pred, target in zip(user_item_pairs, predictions, targets):
        if user not in user_predictions:
            user_predictions[user] = []
            user_targets[user] = []
        user_predictions[user].append(pred)
        user_targets[user].append(target)

    # Calculate NDCG, Precision, and Recall for each user and take the mean
    ndcgs = []
    precisions = []
    recalls = []
    for user in user_predictions:
        user_preds = np.array(user_predictions[user])
        user_targs = np.array(user_targets[user])
        ndcgs.append(calculate_ndcg([user_preds], [user_targs], k)[0])
        precisions.append(precision_at_k(user_preds, user_targs, k))
        recalls.append(recall_at_k(user_preds, user_targs, k))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NDCG@{}'.format(k): np.mean(ndcgs),
        'Precision@{}'.format(k): np.mean(precisions),
        'Recall@{}'.format(k): np.mean(recalls)
    }