import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.where(input > 0, torch.tensor(1), torch.tensor(0))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# use ste to set the binary function
def binary_ste(input):
    return BinarySTE.apply(input)

class MLPWithMask(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hideen_dim_2, output_dim):
        super(MLPWithMask, self).__init__()
        # mask
        self.mask_1 = nn.Parameter(torch.ones(input_dim))
        self.mask_2 = nn.Parameter(torch.ones(hidden_dim_1))
        self.mask_3 = nn.Parameter(torch.ones(hidden_dim_2))
        # MLP
        self.feature_extraction_layer_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.PReLU()
        )
        
        self.feature_extraction_layer_2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.PReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_2, output_dim)
        )

    def forward(self, x):
        mask_1 = self.mask_1
        # binary_mask_1 = binary_ste(self.mask_2)
        # binary_mask_2 = binary_ste(self.mask_3)
        masked_x = x * mask_1
        x = self.feature_extraction_layer_1(masked_x)
        binary_mask_1 = binary_ste(x)
        masked_x_1 = x * binary_mask_1
        x = self.feature_extraction_layer_2(masked_x_1)
        binary_mask_2 = binary_ste(x)
        masked_x_2 = x * binary_mask_2
        x = self.classifier(masked_x_2)
        return x

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model, loss function and optimizer
# set input_dim and output_dim according to your dataset
input_dim = input_dim
hidden_dim_1 = 64
hidden_dim_2 = 16
output_dim = output_dim
model = MLPWithMask(input_dim, hidden_dim_1, hidden_dim_2, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

# train model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # set l1_weight by yourself
            l1_loss = l1_weight * torch.sum(torch.abs(model.mask_1))
            total_loss = loss+l1_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")
        
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
                precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average = 'macro')
                recall = recall_score(y_test_tensor.numpy(), predicted.numpy(), average = 'macro')
                f1score = f1_score(y_test_tensor.numpy(), predicted.numpy(), average = 'macro')
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Val Loss: {val_loss.item():.4f}, "
                      f"Accuracy: {accuracy:.4f}",
                      f"Precision: {precision:.4f}",
                      f"Recall: {recall:.4f}",
                      f"F1score: {f1score:.4f}"
                )

# train and test
import time

start_time = time.time()
train_model(model, train_loader, criterion, optimizer, epochs)
end_time = time.time()

train_time = end_time - start_time
