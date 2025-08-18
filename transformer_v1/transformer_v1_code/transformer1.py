# added mask embedding and masking
# first working transformer

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_hdf("events_a1a1_new.h5", key="df")
df = df.sample(frac=1).reset_index(drop=True)
i_train = int(len(df)*0.8)
i_valid = int(len(df)*0.9)
df_train = df.iloc[0:i_train, :]
df_valid = df.iloc[i_train:i_valid, :]
df_test = df.iloc[i_valid:, :]

def input_vectors(df): 
    features1 = ['pt_tau_1', 'phi_tau_1', 'eta_tau_1']
    features2 = ['pt_tau_2', 'phi_tau_2', 'eta_tau_2']
    features_met = ['pt_met', 'phi_met']
    
    features_11 = ['pt_h1_pi1_reco', 'phi_h1_pi1_reco', 'eta_h1_pi1_reco']
    features_12 = ['pt_h1_pi2_reco', 'phi_h1_pi2_reco', 'eta_h1_pi2_reco']
    features_13 = ['pt_h1_pi3_reco', 'phi_h1_pi3_reco', 'eta_h1_pi3_reco']
    features_21 = ['pt_h2_pi1_reco', 'phi_h2_pi1_reco', 'eta_h2_pi1_reco']
    features_22 = ['pt_h2_pi2_reco', 'phi_h2_pi2_reco', 'eta_h2_pi2_reco']
    features_23 = ['pt_h2_pi3_reco', 'phi_h2_pi3_reco', 'eta_h2_pi3_reco']
    
    features_gen1 = ['pt_gentau_1']
    features_gen2 = ['pt_gentau_2']
    
    df1 = df[features1].copy()
    df2 = df[features2].copy()
    df_met = df[features_met].copy()
    df1['pt_tau_1'] = np.log(df1['pt_tau_1'])
    df2['pt_tau_2'] = np.log(df2['pt_tau_2'])
    df_met['pt_met'] = np.log(df_met['pt_met'])
    df_met['eta_met'] = 0
    X1 = torch.tensor(df1.values, dtype=torch.float32)
    X2 = torch.tensor(df2.values, dtype=torch.float32)
    X_met = torch.tensor(df_met.values, dtype=torch.float32)
    
    df11 = df[features_11].copy()
    df12 = df[features_12].copy()
    df13 = df[features_13].copy()
    df21 = df[features_21].copy()
    df22 = df[features_22].copy()
    df23 = df[features_23].copy()
    df11.iloc[:, 0] = np.log(df11.iloc[:, 0])
    df12.iloc[:, 0] = np.log(df12.iloc[:, 0])
    df13.iloc[:, 0] = np.log(df13.iloc[:, 0])
    df21.iloc[:, 0] = np.log(df21.iloc[:, 0])
    df22.iloc[:, 0] = np.log(df22.iloc[:, 0])
    df23.iloc[:, 0] = np.log(df23.iloc[:, 0])
    X11 = torch.tensor(df11.values, dtype=torch.float32)
    X12 = torch.tensor(df12.values, dtype=torch.float32)  
    X13 = torch.tensor(df13.values, dtype=torch.float32)
    X21 = torch.tensor(df21.values, dtype=torch.float32)
    X22 = torch.tensor(df22.values, dtype=torch.float32)
    X23 = torch.tensor(df23.values, dtype=torch.float32)
    
    
    df_gen1 = df[features_gen1].copy()
    df_gen2 = df[features_gen2].copy()
    df_gen1['pt_gentau_1'] = np.log(df_gen1['pt_gentau_1'])
    df_gen2['pt_gentau_2'] = np.log(df_gen2['pt_gentau_2'])   
    y_gen1 = torch.tensor(df_gen1.values, dtype=torch.float32)
    y_gen2 = torch.tensor(df_gen2.values, dtype=torch.float32)
    
    tau_matrix = torch.stack([X1, X2, X_met], dim=1) 
    decay_matrix = torch.stack([X11, X12, X13, X21, X22, X23], dim=1)
    y_target = torch.cat((y_gen1, y_gen2), dim=1)
    
       
    return tau_matrix, decay_matrix, y_target


def compute_interaction_mask(tau_matrix):
    pT = np.exp(tau_matrix[:, :, 0])
    phi = tau_matrix[:, :, 1]
    eta = tau_matrix[:, :, 2]  

    delta_eta = eta[:, :, None] - eta[:, None, :]  
    delta_phi = phi[:, :, None] - phi[:, None, :]  
    delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi
    print('zero')
    deltaR = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-8)  
    min_pT = torch.min(pT[:, :, None], pT[:, None, :])  
    sum_pT = pT[:, :, None] + pT[:, None, :]  

    kT = min_pT * deltaR  
    z = min_pT / (sum_pT + 1e-8)

    mass2 = 2 * pT[:, :, None] * pT[:, None, :] * (torch.cosh(delta_eta) - torch.cos(delta_phi))
    mass2 = torch.clamp(mass2, min=1e-8)  

    interaction_mask = torch.stack([
        torch.log(deltaR), 
        torch.log(kT),  
        torch.log(z ), 
        torch.log(mass2) 
    ], dim=-1) 


    interaction_mask = torch.nan_to_num(interaction_mask)
    return interaction_mask


class EmbeddingDecay(nn.Module):
    def __init__(self, input_dim=3, output_dim=8, hidden_dim=512):
        super().__init__()
        self.embedding1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding4 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding5 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding6 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.embedding1(x[:,0,:]), self.embedding2(x[:,1,:]), self.embedding3(x[:,2,:]), self.embedding4(x[:,3,:]), self.embedding5(x[:,4,:]), self.embedding6(x[:,5,:])
    
    
class EmbeddingTau(nn.Module):
    def __init__(self, input_dim=3, output_dim=8, hidden_dim=512):
        super().__init__()
        self.embedding1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.embedding3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.mask_embedding =  nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
    
    def forward(self, x):
        mask = compute_interaction_mask(x)
        mask = self.mask_embedding(mask)
        return self.embedding1(x[:, 0, :]), self.embedding2(x[:, 1, :]), self.embedding3(x[:, 2, :]), mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            mask = torch.stack([mask]*self.num_heads, dim=1)
            indices = torch.arange(self.num_heads)
            mask_new = mask[:, indices, :, :, indices]
            mask_new = mask_new.transpose(0, 1)
            attn_scores = attn_scores + mask_new
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    


class CrossAttentionModule(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttentionModule, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, y):
        N = x.shape[0]
        value_len, key_len, query_len = y.shape[1], y.shape[1], x.shape[1]

        values = self.values(y)
        keys = self.keys(y)
        queries = self.queries(x)

        # Split into heads
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        # Attention calculation
        energy = torch.einsum('nqhd,nkhd->nqk', [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum('nqk,nvhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)


class TauTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, ff_hidden_dim=64, dropout=0.1):
        super().__init__()

        self.embedding_tau = EmbeddingTau(input_dim=3, output_dim=embed_dim)
        self.embedding_additional = EmbeddingDecay(input_dim=3, output_dim=embed_dim)

        self.mha_tau = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mha_additional = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.cross_attention = CrossAttentionModule(embed_dim, num_heads)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 3, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 2)
        )


    def forward(self, tau_matrix, decay_matrix):
        X1_emb, X2_emb, X_met_emb, mask = self.embedding_tau(tau_matrix)
        X11_emb, X12_emb, X13_emb, X21_emb, X22_emb, X23_emb = self.embedding_additional(decay_matrix)

        X_tau = torch.cat((X1_emb.unsqueeze(1), X2_emb.unsqueeze(1), X_met_emb.unsqueeze(1)), dim=1)
        X_prod = torch.cat((X11_emb.unsqueeze(1), X12_emb.unsqueeze(1), X13_emb.unsqueeze(1), X21_emb.unsqueeze(1), X22_emb.unsqueeze(1), X23_emb.unsqueeze(1)), dim=1)
        X_tau = self.mha_tau(X_tau, X_tau, X_tau, mask=mask)  
        X_prod = self.mha_additional(X_prod, X_prod, X_prod)  
        
        X_tau_adjusted = self.cross_attention(X_tau, X_prod)

        output = self.regressor(X_tau_adjusted.view(X_tau_adjusted.shape[0], -1))

        return output


tau_matrix, decay_matrix, y_target = input_vectors(df_train)
tau_matrix_v, decay_matrix_v, y_target_v = input_vectors(df_valid)

print('vectors generated')

model = TauTransformer(embed_dim=128, num_heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fn = nn.MSELoss()
print(y_target)

epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(tau_matrix, decay_matrix)
    loss = loss_fn(y_pred, y_target)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        y_pred_v = model(tau_matrix_v, decay_matrix_v)
        val_loss = loss_fn(y_pred_v, y_target_v)
        val_losses.append(val_loss)
    
    print(f"epoch {epoch+1}, loss: {loss.item():.4f}, valid: {val_loss.item():.4f}")
    
plt.figure()
plt.hist(y_target[:, 0].detach().numpy(), bins=50, alpha=0.5)
plt.hist(y_pred[:, 0].detach().numpy(), bins=50, alpha=0.5)
plt.title('training preds')
plt.show()

tau_matrix, decay_matrix, y_target = input_vectors(df_test)
model.eval()
test_preds = model(tau_matrix, decay_matrix)

plt.figure()
plt.hist(y_target[:, 0].detach().numpy(), bins=50, alpha=0.5)
plt.hist(test_preds[:, 0].detach().numpy(), bins=50, alpha=0.5)
plt.title('test preds')
plt.show()

plt.figure()
plt.hist(y_target[:, 0].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 0].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.ylabel('counts')
plt.xlabel('pt1')
plt.title('pt1 regressed vs generated')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 1].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 1].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.ylabel('counts')
plt.xlabel('pt2')
plt.title('pt2 regressed vs generated')
plt.legend()
plt.show()

# plt.figure()
# plt.plot(range(2, epochs + 1), losses[1:], label='training loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

import torch.nn.functional as F
# Assuming y_target and test_preds have shape [N, 2] where column 0 is pt1 and column 1 is pt2
mse_pt1 = F.mse_loss(test_preds[:, 0], y_target[:, 0]).item()
mse_pt2 = F.mse_loss(test_preds[:, 1], y_target[:, 1]).item()

print(f"MSE for pt1: {mse_pt1:.4f}")
print(f"MSE for pt2: {mse_pt2:.4f}")

plt.figure()
plt.plot(range(epochs), train_losses, label='training loss')
plt.plot(range(epochs), val_losses, label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
