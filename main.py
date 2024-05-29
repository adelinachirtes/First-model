import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

# Create DataLoader for each dataset
batch_size = 105
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define models
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        outputs = self.linear(x)
        return self.sigmoid(outputs)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return self.sigmoid(out)


# Custom Decision Tree implementation (simplified)
class DecisionTree:
    def __init__(self, input_dim, output_dim, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.output_dim = output_dim

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        proba = self.model.predict_proba(X)
        # Convert probabilities to one-hot encoding
        one_hot_output = np.zeros((proba.shape[0], self.output_dim))
        one_hot_output[np.arange(proba.shape[0]), proba.argmax(axis=1)] = 1
        return torch.tensor(one_hot_output, dtype=torch.float32)

# Define your ensemble
class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = []
        for model in self.models:
            if isinstance(model, DecisionTree):
                output = model.predict(x.numpy())
                outputs.append(output)
            else:
                outputs.append(model(x))
        return torch.mean(torch.stack(outputs), dim=0)

    def parameters(self):
        for model in self.models:
            if isinstance(model, torch.nn.Module):
                yield from model.parameters()

# Initialize your classifiers
num_classes = len(train_dataset.classes)
input_dim = 128 * 128 * 3  # Assuming input images are 128x128x3
hidden_dim = 64

lr = LogisticRegression(input_dim, num_classes)
mlp = MLP(input_dim, hidden_dim, num_classes)
dt = DecisionTree(input_dim, num_classes)

# Initialize your ensemble
ensemble = Ensemble([lr, mlp, dt])

# Define your loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)

# Logging with TensorBoard
writer = SummaryWriter()

# Training loop
epochs = 20
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    # Training phase
    ensemble.train()
    epoch_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
        if isinstance(ensemble.models[2], DecisionTree) and epoch == 0 and i == 0:
            dt.fit(inputs.numpy(), labels.numpy())
        outputs = ensemble(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation phase
    ensemble.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            outputs = ensemble(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        valid_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Validation Accuracy: {valid_accuracy}%, Loss: {epoch_loss}')

    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break

    # Save the model state after each epoch
    torch.save(ensemble.state_dict(), f'model_epoch_{epoch}.pth')

writer.close()


# Evaluation
def evaluate(loader):
    ensemble.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            outputs = ensemble(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Calculate metrics
train_accuracy = evaluate(train_loader)
test_accuracy = evaluate(test_loader)

print(f'Train Accuracy: {train_accuracy}%')
print(f'Test Accuracy: {test_accuracy}%')

# Cross Validation
kf = KFold(n_splits=5)
for fold, (train_index, test_index) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold + 1}')
    train_subset = Subset(train_dataset, train_index)
    test_subset = Subset(train_dataset, test_index)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Repeat the training and evaluation steps as before
    for epoch in range(epochs):
        ensemble.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
            if isinstance(ensemble.models[2], DecisionTree) and epoch == 0 and i == 0:
                dt.fit(inputs.numpy(), labels.numpy())
            outputs = ensemble(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        ensemble.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the input
                outputs = ensemble(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_accuracy = 100 * correct / total
            print(f'Fold {fold + 1}, Epoch {epoch + 1}, Validation Accuracy: {valid_accuracy}%, Loss: {epoch_loss}')

# Logging metrics with TensorBoard
writer.add_scalar('Loss/train', epoch_loss, epoch)
writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

writer.close()
