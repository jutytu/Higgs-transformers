import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_hdf("events_a1a1.h5", key="df")


def hist(data, bin_num):
    counts, bin_edges = np.histogram(data, bins=bin_num)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    errors = (bin_centers[1]-bin_centers[0])/2
    plt.errorbar(bin_centers, counts, xerr=errors, fmt='.')
    plt.xlabel(f'{data.name}')
    plt.ylabel('counts')
    
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df['pt_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df['pt_tau_2'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/pt_tau_1_pt_tau_2.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df[df['pt_tau_1'] < 150]['pt_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df[df['pt_tau_2'] < 100]['pt_tau_2'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/pt_tau_1_pt_tau_2_closeup.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df['eta_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df['eta_tau_2'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/eta_tau.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df['phi_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df['phi_tau_2'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/phi_tau.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
hist(df['pt_jet_1'], 70)
plt.subplot(1, 3, 2)
hist(df['pt_jet_2'], 70)
plt.subplot(1, 3, 3)
hist(df['pt_jet_3'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/pt_jets.png'
plt.savefig(file_path)
plt.show()

plt.figure()
hist(df['pt_met'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/pt_met.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df['charge_pos_tau_1']-df['charge_neg_tau_1'], 2)
plt.xlabel('tau 1 charge')
plt.subplot(1, 2, 2)
hist(df['charge_pos_tau_2']-df['charge_neg_tau_2'], 2)
plt.xlabel('tau 2 charge')
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/charges_tau.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df[df['pt_tau_1']<200]['pt_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df[df['pt_gentau_1']<200]['pt_gentau_1'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/pt_gentau.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
hist(df[df['eta_tau_1']<200]['eta_tau_1'], 70)
plt.subplot(1, 2, 2)
hist(df[df['eta_gentau_1']<200]['eta_gentau_1'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/eta_gentau.png'
plt.savefig(file_path)
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
hist(df['px_met'], 70)
plt.subplot(1, 3, 2)
hist(df['py_met'], 70)
plt.subplot(1, 3, 3)
hist(df['phi_met'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/phi_met.png'
plt.savefig(file_path)
plt.show()

plt.figure()
hist(df['covxy_met'], 70)
file_path = 'C:/Users/Ja/OneDrive/Dokumenty/Materiały/DA/Project Higgs_transformer/Stage/events_a1a1_figures/covxy_met.png'
plt.savefig(file_path)
plt.show()
