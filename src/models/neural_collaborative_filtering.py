import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32, 16]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        
        self.output_layer = nn.Linear(layers[-1], 1)
        self.activation = nn.ReLU()

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        
        for layer in self.fc_layers:
            vector = self.activation(layer(vector))
        
        output = self.output_layer(vector)
        output = torch.sigmoid(output.squeeze(-1))  # Sigmoid activation
        return output * 5.0  # Scale to 0-5 range

class NeuralCollaborativeFiltering:
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32, 16], learning_rate=0.001):
        self.model = NCF(num_users, num_items, embedding_dim, layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_step(self, user_ids, item_ids, ratings):
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(user_ids, item_ids)
        loss = self.criterion(predictions, ratings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, train_loader, device):
        total_loss = 0
        for batch in train_loader:
            user_ids, item_ids, ratings = [x.to(device) for x in batch]
            loss = self.train_step(user_ids, item_ids, ratings)
            total_loss += loss
        return total_loss / len(train_loader)

    def predict(self, user_ids, item_ids):
        self.model.eval()
        with torch.no_grad():
            return self.model(user_ids, item_ids)

    def to(self, device):
        self.model = self.model.to(device)
        return self