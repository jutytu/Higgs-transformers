# boosting pions and adding angles

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset
import vector


def normalize_columns(df, exclude_cols=[]):
    df_norm = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = df[col] / max(abs(max_val), abs(min_val))
            else:
                df_norm[col] = 0
    return df_norm


def compute_angle(boosted_pion, tau):
    dot_product = boosted_pion.px * tau.px + boosted_pion.py * tau.py + boosted_pion.pz * tau.pz
    pion_mag = np.sqrt(boosted_pion.px**2 + boosted_pion.py**2 + boosted_pion.pz**2)
    tau_mag = np.sqrt(tau.px**2 + tau.py**2 + tau.pz**2)
    cos_theta = np.clip(dot_product / (pion_mag * tau_mag), -1.0, 1.0)
    return cos_theta


def input_vectors(df): 
    features1 = ['px_tau_1', 'py_tau_1', 'pz_tau_1', 'energy_tau_1']
    features2 = ['px_tau_2', 'py_tau_2', 'pz_tau_2', 'energy_tau_2']
    f_PV = ['PVx', 'PVy', 'PVz']
    f_SV1 = ['SVx_tau_1', 'SVy_tau_1', 'SVz_tau_1']
    f_SV2 = ['SVx_tau_2', 'SVy_tau_2', 'SVz_tau_2']
    
    features_met = ['px_met', 'py_met', 'pz_met', 'energy_met']
    features_jet1 = ['px_jet_1', 'py_jet_1', 'pz_jet_1', 'energy_jet_1']
    features_jet2 = ['px_jet_2', 'py_jet_2', 'pz_jet_2', 'energy_jet_2']
    features_jet3 = ['px_jet_3', 'py_jet_3', 'pz_jet_3', 'energy_jet_3']
    
    features_11 = ['px_h1_pi1_reco', 'py_h1_pi1_reco', 'pz_h1_pi1_reco', 'energy_h1_pi1_reco', 'charge_pos_h1_pi1_reco', 'charge_neg_h1_pi1_reco', 'efrac_h1_pi1_reco']
    features_12 = ['px_h1_pi2_reco', 'py_h1_pi2_reco', 'pz_h1_pi2_reco', 'energy_h1_pi2_reco', 'charge_pos_h1_pi2_reco', 'charge_neg_h1_pi2_reco', 'efrac_h1_pi2_reco']
    features_13 = ['px_h1_pi3_reco', 'py_h1_pi3_reco', 'pz_h1_pi3_reco', 'energy_h1_pi3_reco', 'charge_pos_h1_pi3_reco', 'charge_neg_h1_pi3_reco', 'efrac_h1_pi3_reco']
    features_21 = ['px_h2_pi1_reco', 'py_h2_pi1_reco', 'pz_h2_pi1_reco', 'energy_h2_pi1_reco', 'charge_pos_h2_pi1_reco', 'charge_neg_h2_pi1_reco', 'efrac_h2_pi1_reco']
    features_22 = ['px_h2_pi2_reco', 'py_h2_pi2_reco', 'pz_h2_pi2_reco', 'energy_h2_pi2_reco', 'charge_pos_h2_pi2_reco', 'charge_neg_h2_pi2_reco', 'efrac_h2_pi2_reco']
    features_23 = ['px_h2_pi3_reco', 'py_h2_pi3_reco', 'pz_h2_pi3_reco', 'energy_h2_pi3_reco', 'charge_pos_h2_pi3_reco', 'charge_neg_h2_pi3_reco', 'efrac_h2_pi3_reco']
    
    features_gen1 = ['px_nu1', 'py_nu1', 'pz_nu1']
    features_gen2 = ['px_nu2', 'py_nu2', 'pz_nu2']
    
    df1 = df[features1].copy()
    df2 = df[features2].copy()
    
    df_met = df[features_met].copy()
    df_jet1 = df[features_jet1].copy()
    df_jet2 = df[features_jet2].copy()
    df_jet3 = df[features_jet3].copy()
    df1[['Vx_1', 'Vy_1', 'Vz_1']] = df[f_SV1].subtract(df[f_PV].values, axis=0)
    df2[['Vx_2', 'Vy_2', 'Vz_2']] = df[f_SV2].subtract(df[f_PV].values, axis=0)
    
    tau1 = vector.array({
    "px": df1["px_tau_1"],
    "py": df1["py_tau_1"],
    "pz": df1["pz_tau_1"],
    "E": df1['energy_tau_1'],
    })
    
    tau2 = vector.array({
    "px": df2["px_tau_2"],
    "py": df2["py_tau_2"],
    "pz": df2["pz_tau_2"],
    "E": df2['energy_tau_2'],
    })
    
    tau = tau1+tau2

    # df1 = normalize_columns(df1)
    # df2 = normalize_columns(df2)
    X1 = torch.tensor(df1.values, dtype=torch.float32)
    X2 = torch.tensor(df2.values, dtype=torch.float32)
    
    # df_met = normalize_columns(df_met)
    # df_jet1 = normalize_columns(df_jet1)
    # df_jet2 = normalize_columns(df_jet2)
    # df_jet3 = normalize_columns(df_jet3)
    X_met = torch.tensor(df_met.values, dtype=torch.float32)
    X_jet1 = torch.tensor(df_jet1.values, dtype=torch.float32)
    X_jet2 = torch.tensor(df_jet2.values, dtype=torch.float32)
    X_jet3 = torch.tensor(df_jet3.values, dtype=torch.float32)
    
    df11 = df[features_11].copy()
    df12 = df[features_12].copy()
    df13 = df[features_13].copy()
    df21 = df[features_21].copy()
    df22 = df[features_22].copy()
    df23 = df[features_23].copy()
    
    pi11 = vector.array({
    "px": df11["px_h1_pi1_reco"],
    "py": df11["py_h1_pi1_reco"],
    "pz": df11["pz_h1_pi1_reco"],
    "E": df11["energy_h1_pi1_reco"],
    })
    
    pi12 = vector.array({
    "px": df12["px_h1_pi2_reco"],
    "py": df12["py_h1_pi2_reco"],
    "pz": df12["pz_h1_pi2_reco"],
    "E": df12["energy_h1_pi2_reco"],
    })

    pi13 =  vector.array({
    "px": df13["px_h1_pi3_reco"],
    "py": df13["py_h1_pi3_reco"],
    "pz": df13["pz_h1_pi3_reco"],
    "E": df13["energy_h1_pi3_reco"],
    })

    pi21 = vector.array({
    "px": df21["px_h2_pi1_reco"],
    "py": df21["py_h2_pi1_reco"],
    "pz": df21["pz_h2_pi1_reco"],
    "E": df21["energy_h2_pi1_reco"],
    })

    pi22 = vector.array({
    "px": df22["px_h2_pi2_reco"],
    "py": df22["py_h2_pi2_reco"],
    "pz": df22["pz_h2_pi2_reco"],
    "E": df22["energy_h2_pi2_reco"],
    })

    pi23 = vector.array({
    "px": df23["px_h2_pi3_reco"],
    "py": df23["py_h2_pi3_reco"],
    "pz": df23["pz_h2_pi3_reco"],
    "E": df23["energy_h2_pi3_reco"],
    })
    
    pi11 = pi11.boost(-tau.to_beta3())
    pi12 = pi12.boost(-tau.to_beta3())
    pi13 = pi13.boost(-tau.to_beta3())
    pi21 = pi21.boost(-tau.to_beta3())
    pi22 = pi22.boost(-tau.to_beta3())
    pi23 = pi23.boost(-tau.to_beta3())
    
    df11['px_h1_pi1_reco'], df11['py_h1_pi1_reco'], df11['pz_h1_pi1_reco'], df11['energy_h1_pi1_reco'] = pi11.px, pi11.py, pi11.pz, pi11.E
    df12['px_h1_pi2_reco'], df12['py_h1_pi2_reco'], df12['pz_h1_pi2_reco'], df12['energy_h1_pi2_reco'] = pi12.px, pi12.py, pi12.pz, pi12.E
    df13['px_h1_pi3_reco'], df13['py_h1_pi3_reco'], df13['pz_h1_pi3_reco'], df13['energy_h1_pi3_reco'] = pi13.px, pi13.py, pi13.pz, pi13.E

    df21['px_h2_pi1_reco'], df21['py_h2_pi1_reco'], df21['pz_h2_pi1_reco'], df21['energy_h2_pi1_reco'] = pi21.px, pi21.py, pi21.pz, pi21.E
    df22['px_h2_pi2_reco'], df22['py_h2_pi2_reco'], df22['pz_h2_pi2_reco'], df22['energy_h2_pi2_reco'] = pi22.px, pi22.py, pi22.pz, pi22.E
    df23['px_h2_pi3_reco'], df23['py_h2_pi3_reco'], df23['pz_h2_pi3_reco'], df23['energy_h2_pi3_reco'] = pi23.px, pi23.py, pi23.pz, pi23.E
    
    
    # df11 = normalize_columns(df11, exclude_cols=['charge_pos_h1_pi1_reco', 'charge_neg_h1_pi1_reco'])
    # df12 = normalize_columns(df12, exclude_cols=['charge_pos_h1_pi2_reco', 'charge_neg_h1_pi2_reco'])
    # df13 = normalize_columns(df13, exclude_cols=['charge_pos_h1_pi3_reco', 'charge_neg_h1_pi3_reco'])
    # df21 = normalize_columns(df21, exclude_cols=['charge_pos_h2_pi1_reco', 'charge_neg_h2_pi1_reco'])
    # df22 = normalize_columns(df22, exclude_cols=['charge_pos_h2_pi2_reco', 'charge_neg_h2_pi2_reco'])
    # df23 = normalize_columns(df23, exclude_cols=['charge_pos_h2_pi3_reco', 'charge_neg_h2_pi3_reco'])
    
    df11['id1'], df12['id1'], df13['id1'] = 1, 1, 1
    df21['id1'], df22['id1'], df23['id1'] = 0, 0, 0
    df11['id2'], df12['id2'], df13['id2'] = 0, 0, 0
    df21['id2'], df22['id2'], df23['id2'] = 1, 1, 1
    
    df11["angle"] = compute_angle(pi11, tau1)
    df12["angle"] = compute_angle(pi12, tau1)
    df13["angle"] = compute_angle(pi13, tau1)

    df21["angle"] = compute_angle(pi21, tau2)
    df22["angle"] = compute_angle(pi22, tau2)
    df23["angle"] = compute_angle(pi23, tau2)

    X11 = torch.tensor(df11.values, dtype=torch.float32)
    X12 = torch.tensor(df12.values, dtype=torch.float32)  
    X13 = torch.tensor(df13.values, dtype=torch.float32)
    X21 = torch.tensor(df21.values, dtype=torch.float32)
    X22 = torch.tensor(df22.values, dtype=torch.float32)
    X23 = torch.tensor(df23.values, dtype=torch.float32)
    
    df_gen1 = df[features_gen1].copy()
    df_gen2 = df[features_gen2].copy()
    y_gen1 = torch.tensor(df_gen1.values, dtype=torch.float32)
    y_gen2 = torch.tensor(df_gen2.values, dtype=torch.float32)
    
    tau_matrix = torch.stack([X1, X2], dim=1)  
    other_matrix = torch.stack([X_met, X_jet1, X_jet2, X_jet3], dim=1)                         # here
    decay_matrix = torch.stack([X11, X12, X13, X21, X22, X23], dim=1)  #
    y_target = torch.cat((y_gen1, y_gen2), dim=1)
    y_target = torch.squeeze(y_target, dim=-1)

    return tau_matrix, other_matrix, decay_matrix, y_target


def eta_phi_target(df):
    df_ang = df[['eta_gentau_1', 'phi_gentau_1', 'eta_gentau_2', 'phi_gentau_2']]
    return torch.tensor(df_ang.values, dtype=torch.float32)


def compute_interaction_mask(tau_matrix):
    pT = np.sqrt((tau_matrix[:, :, 0])**2+(tau_matrix[:, :, 1])**2)
    phi = np.arctan2(tau_matrix[:, :, 1], tau_matrix[:, :, 0])
    p = np.sqrt(tau_matrix[:, :, 0]**2 + tau_matrix[:, :, 1]**2 + tau_matrix[:, :, 2]**2)
    eta = eta = 0.5 * np.log((p + tau_matrix[:, :, 2]) / (p - tau_matrix[:, :, 2])) 
    
    # pT = tau_matrix[:, :, 0]
    # phi = tau_matrix[:, :, 2]
    # eta = tau_matrix[:, :, 1]  
    
    delta_eta = eta[:, :, None] - eta[:, None, :]  
    delta_phi = phi[:, :, None] - phi[:, None, :]  
    delta_phi = torch.remainder(delta_phi + torch.pi, 2 * torch.pi) - torch.pi
    
    deltaR = torch.sqrt(delta_eta**2 + delta_phi**2 + 1e-8)  
    min_pT = torch.min(pT[:, :, None], pT[:, None, :])  
    sum_pT = pT[:, :, None] + pT[:, None, :]  
    
    kT = min_pT * deltaR  
    z = min_pT / (sum_pT + 1e-8)
    
    mass2 = 2 * pT[:, :, None] * pT[:, None, :] * (torch.cosh(delta_eta) - torch.cos(delta_phi))
    mass2 = torch.clamp(mass2, min=1e-8) 
    z = torch.clamp(z, min=1e-8)
    kT = torch.clamp(kT, min=1e-8)
    
    interaction_mask = torch.stack([
        torch.log(deltaR), 
        torch.log(kT),  
        torch.log(z), 
        torch.log(mass2) 
    ], dim=-1) 
    
    
    interaction_mask = torch.nan_to_num(interaction_mask)
    return interaction_mask

    


class EmbeddingDecay(nn.Module):
    def __init__(self, input_dim=10, output_dim=8, hidden_dim=512):    # here for more decay info
        super().__init__()
        self.embedding1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding4 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding5 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding6 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):                                         # here
        return self.embedding1(x[:,0,:]), self.embedding2(x[:,1,:]), self.embedding3(x[:,2,:]), self.embedding4(x[:,3,:]), self.embedding5(x[:,4,:]), self.embedding6(x[:,5,:])
    
class EmbeddingOther(nn.Module):
    def __init__(self, input_dim=4, output_dim=8, hidden_dim=512):
        super().__init__()

        self.embedding3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding4 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding5 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.embedding6 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

        self.mask_embedding =  nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 16)
        )
        
    def forward(self, x):
        return self.embedding3(x[:, 0, :]), self.embedding4(x[:, 1, :]), self.embedding5(x[:, 2, :]), self.embedding6(x[:, 3, :])  # here

class EmbeddingTau(nn.Module):
    def __init__(self, input_dim=7, output_dim=8, hidden_dim=512):
        super().__init__()
        self.embedding1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        self.embedding2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        self.mask_embedding =  nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 16)
        )
    
    def forward(self, x):
        mask = compute_interaction_mask(x)
        mask = self.mask_embedding(mask)
        return self.embedding1(x[:, 0, :]), self.embedding2(x[:, 1, :]), mask  # here

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model) 
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = torch.stack([mask]*self.num_heads, dim=1)
            indices = torch.arange(self.num_heads)
            mask_new = mask[:, indices, :, :, indices]
            mask_new = mask_new.transpose(0, 1)
            attn_scores = attn_scores + mask_new
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        x = V
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        output = self.W_o(self.combine_heads(attn_output))
        return x + output
    


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

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum('nqhd,nkhd->nqk', [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum('nqk,nvhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        return x + self.fc_out(out)


class TauTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, ff_hidden_dim=64, dropout=0.1):
        super().__init__()

        self.embedding_tau = EmbeddingTau(input_dim=7, output_dim=embed_dim)          # here for decay info
        self.embedding_additional = EmbeddingDecay(input_dim=10, output_dim=embed_dim)
        self.embedding_other = EmbeddingOther(input_dim=4, output_dim=embed_dim)

        self.mha_tau = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mha_other = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mha_additional = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        self.cross_attention = CrossAttentionModule(embed_dim, num_heads)
        self.cross_attention2 = CrossAttentionModule(embed_dim, num_heads)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, ff_hidden_dim),                            # here!! - num of particles!! not features
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 6)
        )


    def forward(self, tau_matrix, other_matrix, decay_matrix):
        X1_emb, X2_emb, mask = self.embedding_tau(tau_matrix)      # here
        X_met_emb, X_jet1_emb, X_jet2_emb, X_jet3_emb = self.embedding_other(other_matrix)
        X11_emb, X12_emb, X13_emb, X21_emb, X22_emb, X23_emb = self.embedding_additional(decay_matrix)

        X_tau = torch.cat((X1_emb.unsqueeze(1), X2_emb.unsqueeze(1)), dim=1)    # here
        X_other = torch.cat((X_met_emb.unsqueeze(1), X_jet1_emb.unsqueeze(1), X_jet2_emb.unsqueeze(1), X_jet3_emb.unsqueeze(1)), dim=1)
        X_prod = torch.cat((X11_emb.unsqueeze(1), X12_emb.unsqueeze(1), X13_emb.unsqueeze(1), X21_emb.unsqueeze(1), X22_emb.unsqueeze(1), X23_emb.unsqueeze(1)), dim=1)
        X_tau = self.mha_tau(X_tau, X_tau, X_tau, mask=mask)  
        X_other = self.mha_other(X_other, X_other, X_other) 
        X_prod = self.mha_additional(X_prod, X_prod, X_prod)  
        X_tau_adjusted = self.cross_attention(X_tau, X_prod)
        X_tau_adjusted = self.cross_attention2(X_tau_adjusted, X_other)

        output = self.regressor(X_tau_adjusted.view(X_tau_adjusted.shape[0], -1))

        return output
    

# class WeightedMSELoss(nn.Module):
#     def __init__(self):
#         super(WeightedMSELoss, self).__init__()

#     def forward(self, preds, targets):
#         pt_pred1, pt_pred2 = torch.sqrt(preds[:, 0]**2+preds[:, 1]**2), torch.sqrt(preds[:, 3]**2+preds[:, 4]**2)
#         pt_tgt1, pt_tgt2 = torch.sqrt(targets[:, 0]**2+targets[:, 1]**2), torch.sqrt(targets[:, 3]**2+targets[:, 4]**2)
#         # eta_1, phi_1, eta_2, phi_2 = angles[:, 0], angles[:, 1], angles[:, 2], angles[:, 3]

#         # m_inv2 = 2*torch.exp(pt_pred1)*torch.exp(pt_pred2)*(torch.cosh(eta_1-eta_2)-torch.cos(phi_1-phi_2))
#         # m_inv = torch.where(m_inv2 >= 0, torch.sqrt(m_inv2), torch.tensor(0.0))
#         # mse_pt1 = (pt_pred1 - pt_tgt1) ** 2
#         # mse_pt2 = (pt_pred2 - pt_tgt2) ** 2
        
#         #loss = 0.0001*(pt_pred1 / pt_pred2 - pt_tgt1 / pt_tgt2).unsqueeze(1)**2*(preds - targets)**2
#         loss = torch.cat(((preds - targets) ** 2, (pt_pred1 / pt_pred2 - pt_tgt1 / pt_tgt2).unsqueeze(1) ** 2), dim=1)
#         return loss.mean()    

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
    




df = pd.read_hdf("events_a1a1_Mar30_2025.h5", key="df")
df = df.sample(frac=1).reset_index(drop=True)
i_train = int(len(df)*0.8)
i_valid = int(len(df)*0.9)
df_train = df.iloc[0:i_train, :]
df_valid = df.iloc[i_train:i_valid, :]
df_test = df.iloc[i_valid:, :]

print(len(df))


angles_train = eta_phi_target(df_train)
print(angles_train.shape)
tau_matrix, other_matrix, decay_matrix, y_target = input_vectors(df_train)
angles_valid = eta_phi_target(df_valid)
tau_matrix_v, other_matrix_v, decay_matrix_v, y_target_v = input_vectors(df_valid)
print(tau_matrix.shape)


print('vectors generated')

model = TauTransformer(embed_dim=128, num_heads=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0006)
loss_fn = LogCoshLoss()

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

batch_size = 100
train_dataset = TensorDataset(tau_matrix, other_matrix, decay_matrix, y_target, angles_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(tau_matrix_v, other_matrix_v, decay_matrix_v, y_target_v, angles_valid)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"num_workers: {train_loader.num_workers}")

epochs = 30
train_losses = []
val_losses = []

print(len(train_loader))

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for tau_matrix, other_matrix, decay_matrix, y_target, angles_train in train_loader:
        optimizer.zero_grad()
        y_preds = model(tau_matrix, other_matrix, decay_matrix)
        loss = loss_fn(y_preds, y_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_train_loss = epoch_loss / len(train_loader)
    train_losses.append(average_train_loss)
        
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for tau_matrix_v, other_matrix_v, decay_matrix_v, y_target_v, angles_valid in val_loader:
            y_preds_v = model(tau_matrix_v, other_matrix_v, decay_matrix_v)
            loss = loss_fn(y_preds_v, y_target_v)
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    if epoch % 1 == 0:
        print(f"epoch {epoch}: train loss {average_train_loss}, val loss {average_val_loss}")


torch.save(model, 'presentation_testing.pth')


tau_matrix, other_matrix, decay_matrix, y_target = input_vectors(df)
model.eval()
test_preds = model(tau_matrix, other_matrix, decay_matrix)

plt.figure()
plt.hist(y_target[:, 0].detach().numpy(), bins=100, alpha=0.5)
plt.hist(test_preds[:, 0].detach().numpy(), bins=100, alpha=0.5)
plt.title('test preds - neutrino px1')
plt.show()

plt.figure()
plt.hist(y_target[:, 1].detach().numpy(), bins=100, alpha=0.5)
plt.hist(test_preds[:, 1].detach().numpy(), bins=100, alpha=0.5)
plt.title('test preds - neutrino py1')
plt.show()


plt.figure()
plt.hist(y_target[:, 2].detach().numpy(), bins=100, alpha=0.5)
plt.hist(test_preds[:, 2].detach().numpy(), bins=100, alpha=0.5)
plt.title('test preds - neutrino pz1')
plt.show()

plt.figure()
plt.hist(y_target[:, 0].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 0].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino px1')
plt.ylabel('counts')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 3].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 3].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino px2')
plt.ylabel('counts')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 1].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 1].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino py1')
plt.ylabel('counts')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 2].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 2].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino pz1')
plt.ylabel('counts')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 4].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 4].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino py2')
plt.ylabel('counts')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_target[:, 5].detach().numpy(), bins=100, alpha=0.5, label='generated')
plt.hist(test_preds[:, 5].detach().numpy(), bins=100, alpha=0.5, label='regressed')
plt.title('test preds - neutrino pz2')
plt.ylabel('counts')
plt.legend()
plt.show()

# m_tau = 1.777  
# test_preds_np = test_preds.detach().numpy()
# px1, py1, pz1 = tau_matrix[:, 0, 0]+test_preds_np[:, 0], tau_matrix[:, 0, 1]+test_preds_np[:, 1], tau_matrix[:, 0, 2]+test_preds_np[:, 2]
# px2, py2, pz2 = tau_matrix[:, 1, 0]+test_preds_np[:, 3], tau_matrix[:, 1, 1]+test_preds_np[:, 4], tau_matrix[:, 1, 2]+test_preds_np[:, 5]
# E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m_tau**2)
# E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m_tau**2)
# m_inv = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)

# plt.figure()
# plt.hist(m_inv, bins=100, alpha=0.5)
# plt.xlabel('inv mass')
# plt.show()


# angles_test = eta_phi_target(df_test)
# eta_1, phi_1, eta_2, phi_2 = angles_test[:, 0], angles_test[:, 1], angles_test[:, 2], angles_test[:, 3]
# m_inv2 = 2*torch.exp(test_preds[:, 0])*torch.exp(test_preds[:, 1])*(torch.cosh(eta_1-eta_2)-torch.cos(phi_1-phi_2))
# plt.figure()
# plt.hist(torch.where(m_inv2 >= 0, torch.sqrt(m_inv2), torch.tensor(0.0)).detach().numpy(), bins=100, alpha=0.5)
# plt.title('test preds - inv mass')
# plt.show()

plt.figure()
plt.plot(range(epochs), train_losses, label='training loss')
plt.plot(range(epochs), val_losses, label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

import torch.nn.functional as F

# mse_px1 = F.mse_loss(test_preds[:, 0]+df['px_tau_1'].values, y_target[:, 0]+df['px_gentau_1'].values).item()
# mse_py1 = F.mse_loss(test_preds[:, 1]+df['py_tau_1'].values, y_target[:, 1]+df['py_gentau_1'].values).item()
# mse_pz1 = F.mse_loss(test_preds[:, 2]+df['pz_tau_1'].values, y_target[:, 2]+df['pz_gentau_1'].values).item()
# mse_px2 = F.mse_loss(test_preds[:, 3]+df['px_tau_2'].values, y_target[:, 3]+df['px_gentau_2'].values).item()
# mse_py2 = F.mse_loss(test_preds[:, 4]+df['py_tau_2'].values, y_target[:, 4]+df['py_gentau_2'].values).item()
# mse_pz2 = F.mse_loss(test_preds[:, 5]+df['pz_tau_2'].values, y_target[:, 5]+df['pz_gentau_2'].values).item()

dtype=torch.float32
px_tau_1 = torch.tensor(df['px_tau_1'].values, dtype=dtype)
py_tau_1 = torch.tensor(df['py_tau_1'].values, dtype=dtype)
pz_tau_1 = torch.tensor(df['pz_tau_1'].values, dtype=dtype)
px_tau_2 = torch.tensor(df['px_tau_2'].values, dtype=dtype)
py_tau_2 = torch.tensor(df['py_tau_2'].values, dtype=dtype)
pz_tau_2 = torch.tensor(df['pz_tau_2'].values, dtype=dtype)

px_gentau_1 = torch.tensor(df['px_gentau_1'].values, dtype=dtype)
py_gentau_1 = torch.tensor(df['py_gentau_1'].values, dtype=dtype)
pz_gentau_1 = torch.tensor(df['pz_gentau_1'].values, dtype=dtype)
px_gentau_2 = torch.tensor(df['px_gentau_2'].values, dtype=dtype)
py_gentau_2 = torch.tensor(df['py_gentau_2'].values, dtype=dtype)
pz_gentau_2 = torch.tensor(df['pz_gentau_2'].values, dtype=dtype)

# Compute MSEs
mse_px1 = F.mse_loss(test_preds[:, 0] + px_tau_1, y_target[:, 0] + px_gentau_1).item()
mse_py1 = F.mse_loss(test_preds[:, 1] + py_tau_1, y_target[:, 1] + py_gentau_1).item()
mse_pz1 = F.mse_loss(test_preds[:, 2] + pz_tau_1, y_target[:, 2] + pz_gentau_1).item()
mse_px2 = F.mse_loss(test_preds[:, 3] + px_tau_2, y_target[:, 3] + px_gentau_2).item()
mse_py2 = F.mse_loss(test_preds[:, 4] + py_tau_2, y_target[:, 4] + py_gentau_2).item()
mse_pz2 = F.mse_loss(test_preds[:, 5] + pz_tau_2, y_target[:, 5] + pz_gentau_2).item()



print(f"MSE for px1 - full momenta!: {mse_px1:.4f}")
print(f"MSE for px2: {mse_px2:.4f}")
print(f"MSE for py1: {mse_py1:.4f}")
print(f"MSE for py2: {mse_py2:.4f}")
print(f"MSE for pz1: {mse_pz1:.4f}")
print(f"MSE for pz2: {mse_pz2:.4f}")
