import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import TinyBinaryClassifier

# Create fake data
X = torch.randn(500, 4)     # 500 samples, each 4 features
y = (torch.rand(500) > 0.5).float().unsqueeze(1) # 0/1 labels

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss , optimizer
model = TinyBinaryClassifier()
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop 
for epoch in range(10):
    for batch_X, batch_y in loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# save model
torch.save(model.state_dict(), "tiny_model.pt")
print("Model saved to tiny_model.pt")


