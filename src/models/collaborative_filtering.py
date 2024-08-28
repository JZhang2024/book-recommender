import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot_product = torch.sum(user_embeds * item_embeds, dim=1)
        return dot_product + user_bias + item_bias

class CollaborativeFiltering:
    def __init__(self, num_users, num_items, embedding_dim=100, learning_rate=0.001):
        self.model = MatrixFactorization(num_users, num_items, embedding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train_epoch(self, train_loader, device):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            user_ids, item_ids, ratings = [x.to(device) for x in batch]
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def predict(self, user_ids, item_ids):
        self.model.eval()
        with torch.no_grad():
            return self.model(user_ids, item_ids)