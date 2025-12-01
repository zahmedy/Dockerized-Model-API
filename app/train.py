import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import TinyBinaryClassifier
import mlflow
import mlflow.pytorch as pytorch

mlflow.set_experiment("tiny-binary-classifier")

# Create fake data
X = torch.randn(500, 4)     # 500 samples, each 4 features
y = (torch.rand(500) > 0.5).float().unsqueeze(1) # 0/1 labels

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss , optimizer
epochs = 10
lr=0.01
model = TinyBinaryClassifier()
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr)

with mlflow.start_run():
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    # Training loop 
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            mlflow.log_metric("loss", loss.item(), step=epoch)

    #Post training
    pytorch.log_model(model, "model")

# save model
torch.save(model.state_dict(), "tiny_model.pt")
print("Model saved to tiny_model.pt")


