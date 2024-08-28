import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100):
        """
        Initialize the Matrix Factorization model.
        
        Args:
            num_users (int): Number of users in the dataset
            num_items (int): Number of items in the dataset
            embedding_dim (int): Dimension of the embedding vectors
        """
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        """
        Forward pass of the model.
        
        Args:
            user_ids (torch.Tensor): Tensor of user IDs
            item_ids (torch.Tensor): Tensor of item IDs
        
        Returns:
            torch.Tensor: Predicted ratings
        """
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        
        dot_product = torch.sum(user_embeds * item_embeds, dim=1)
        return dot_product + user_bias + item_bias

class CollaborativeFiltering:
    def __init__(self, num_users, num_items, embedding_dim=100, learning_rate=0.001):
        """
        Initialize the Collaborative Filtering model.
        
        Args:
            num_users (int): Number of users in the dataset
            num_items (int): Number of items in the dataset
            embedding_dim (int): Dimension of the embedding vectors
            learning_rate (float): Learning rate for the optimizer
        """
        self.model = MatrixFactorization(num_users, num_items, embedding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, user_ids, item_ids, ratings, num_epochs):
        """
        Train the Collaborative Filtering model.
        Args:
            user_ids (torch.Tensor): Tensor of user IDs
            item_ids (torch.Tensor): Tensor of item IDs
            ratings (torch.Tensor): Tensor of actual ratings
            num_epochs (int): Number of training epochs
        """
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, user_ids, item_ids):
        """
        Make predictions using the trained model.
        
        Args:
            user_ids (torch.Tensor): Tensor of user IDs
            item_ids (torch.Tensor): Tensor of item IDs
        
        Returns:
            torch.Tensor: Predicted ratings
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(user_ids, item_ids)
