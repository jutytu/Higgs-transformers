import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


df = pd.read_hdf("events_a1a1.h5", key="df")
df['R1'] = np.sqrt((df['x_reco_R1'])**2+(df['y_reco_R1'])**2+(df['z_reco_R1'])**2)
df['R2'] = np.sqrt((df['x_reco_R2'])**2+(df['y_reco_R2'])**2+(df['z_reco_R2'])**2)

features_X = ['px_tau_1', 'py_tau_1', 'pz_tau_1', 'energy_tau_1',
              'px_tau_2', 'py_tau_2', 'pz_tau_2', 'energy_tau_2',
              'px_met', 'py_met', 'pz_met',
              'phi_tau_1', 'eta_tau_1', 'phi_tau_2', 'eta_tau_2']
features_y = ['px_gentau_1', 'py_gentau_1', 'pz_gentau_1',
              'px_gentau_2', 'py_gentau_2', 'pz_gentau_2']

X = df[features_X]
y = df[features_y]

X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, in_, out_, neurons=200):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=in_, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=out_)
        )

    def forward(self, x):
        return self.stack(x)


model = Model(in_=15, out_=6)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0005)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = 60
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_preds = model(batch_X)
        loss = loss_fn(y_preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_train_loss = epoch_loss / len(train_loader)
    train_losses.append(average_train_loss)
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_preds = model(batch_X)
            loss = loss_fn(y_preds, batch_y)
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    if epoch % 10 == 0:
        print(f"epoch {epoch}: train loss {average_train_loss}, val loss {average_val_loss}")

plt.figure()
plt.plot(range(2, epochs + 1), train_losses[1:], label='training loss')
plt.plot(range(2, epochs + 1), val_losses[1:], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


model.eval()
test_preds = model(X_test)
test_loss = loss_fn(test_preds, y_test)

relative_errors = torch.abs(y_test - test_preds) / torch.abs(y_test)
abs_errors = torch.abs(y_test - test_preds)

print(test_loss)
print(torch.mean(relative_errors).item())
print(torch.mean(abs_errors).item())

plt.figure()
plt.hist(np.sqrt((test_preds[:, 0].detach().numpy())**2+
         (test_preds[:, 1].detach().numpy())**2), bins=70, label='test_preds', alpha=0.5)
plt.hist(np.sqrt((y_test[:, 0])**2+
                 (y_test[:, 1])**2), bins=70, label='test_gen', alpha=0.5)
plt.legend()
plt.title('p_t tau_1')
plt.xlabel('p_t tau_1')
plt.ylabel('counts')
plt.show()

plt.figure()
plt.hist(np.sqrt((X_test[:, 0])**2+
                 (X_test[:, 1])**2), bins=70, label='test_raw', alpha=0.5)
plt.hist(np.sqrt((y_test[:, 0])**2+
                 (y_test[:, 1])**2), bins=70, label='test_gen', alpha=0.5)
plt.legend()
plt.title('p_t tau_1')
plt.xlabel('p_t tau_1')
plt.ylabel('counts')
plt.show()


plt.figure()
plt.hist(np.sqrt((test_preds[:, 3].detach().numpy())**2+
         (test_preds[:, 4].detach().numpy())**2), bins=70, label='test_preds', alpha=0.5)
plt.hist(np.sqrt((y_test[:, 3])**2+
                 (y_test[:, 4])**2), bins=70, label='test_gen', alpha=0.5)
plt.legend()
plt.title('p_t tau_2')
plt.xlabel('p_t tau_2')
plt.ylabel('counts')
plt.show()

plt.figure()
plt.hist(np.sqrt((X_test[:, 4])**2+
                 (X_test[:, 5])**2), bins=70, label='test_raw', alpha=0.5)
plt.hist(np.sqrt((y_test[:, 3])**2+
                 (y_test[:, 4])**2), bins=70, label='test_gen', alpha=0.5)
plt.legend()
plt.title('p_t tau_2')
plt.xlabel('p_t tau_2')
plt.ylabel('counts')
plt.show()

