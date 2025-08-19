# Full tau momentum regression to construct CP-sensitive variables (using transformer-based models)

This repository contains the work from my research internship in the CMS group at my university.  
The project focused on using transformer-based architectures to improve the reconstruction of tau lepton kinematics in Higgs boson decays. The motivation comes from studying the CP properties of the Higgs through the acoplanarity angle ($\phi_{CP}$), which is sensitive to the Higgs–tau–tau coupling.  

For the theoretical background and data description, see:
- [theory_overview.md](theory/theory_overview.md)  
- [data.md](data/data.md)  


## Project idea and workflow

The general idea was to test whether transformer architectures can outperform a simple feed-forward neural network in recovering missing kinematic information (especially neutrinos) and in building CP-sensitive observables.  

The workflow proceeded through consecutive models:

1. [Neural Network baseline](nn/nn.md)  
   - Feed-forward model for tau momentum regression.  
   - Serves as a baseline for comparison.  

2. [Transformer v1](transformer_v1/transformer_v1.md)  
   - First transformer adaptation for tau momentum regression.  
   - Introduced tau and pion decay matrices as inputs.  

3. [Transformer v2](transformer_v2/transformer_v2.md)  
   - Added vertex info, jets, pion charges/energies, and px/py/pz targets.  
   - Model also regresses neutrino momenta.  

4. [Transformer v3](transformer_v3/transformer_v3.md)  
   - Final version during the internship.  
   - Directly regresses full tau momenta.  
   - $\phi_{CP}$ reconstructed from full 3D information.

Many other versions of the transformer model (around 30) were created during the internship, but their results were not significantly different from the ones obtained using the main versions presented here. Due to their number, they are not included in this repository.

## Repository structure

```bash
Higgs-transformers/
├── data/
│   └── data.md
├── eda/             # exploratory plots
├── nn/
│   ├── nn_code/
│   ├── nn_results/
│   └── nn.md
├── theory/
│   └── theory_overview.md
├── transformer_v1/
│   ├── transformer_v1_code/
│   ├── transformer_v1_results/
│   └── transformer_v1.md
├── transformer_v2/
│   ├── transformer_v2_code/
│   ├── transformer_v2_results/
│   └── transformer_v2.md
├── transformer_v3/
│   ├── transformer_v3_code/
│   ├── transformer_v3_results/
│   └── transformer_v3.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Notes

This work was completed as part of my internship project, thus with a limited time. While the transformer models showed significant improvement over the neural network baseline, there is still room for refinement (e.g. better feature engineering, training strategies, or more advanced transformer variants).





