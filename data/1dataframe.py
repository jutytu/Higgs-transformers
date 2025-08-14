import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

df = pd.read_hdf("events_a1a1.h5", key="df")
print(df.columns)
# df.to_csv('a1a1.csv')

plt.figure()
plt.hist(df['phicp_reco_gentau'], bins=50, label='no weights')
plt.xlabel('phi_CP')
plt.ylabel('counts')
plt.legend()
plt.show()

def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def sin_fit(phi_cp, weights):
    data = phi_cp
    counts, bin_edges = np.histogram(data, bins=50, density=False, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

    initial_guess = [max(counts), 1, 0, np.mean(counts)]
    params, covariance = curve_fit(sine_function, bin_centers, counts, p0=initial_guess)

    A_fit, B_fit, C_fit, D_fit = params
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 300)
    y_fit = sine_function(x_fit, A_fit, B_fit, C_fit, D_fit)  

    plt.hist(data, bins=50, alpha=0.5, label=f'{weights.name}', weights=weights)
    plt.plot(x_fit, y_fit, 'r-', label=f'fit with peak at {(np.pi/2-C_fit)/B_fit}')
    plt.legend()
    plt.xlabel('phi_CP')
    plt.ylabel('counts')
    plt.show()
    
sin_fit(df['phicp_reco_gentau'], df['spinner_wt_cp_even'])
sin_fit(df['phicp_reco_gentau'], df['spinner_wt_cp_odd'])  
sin_fit(df['phicp_reco_gentau'], df['spinner_wt_cp_mm'])