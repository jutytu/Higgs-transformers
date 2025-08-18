# Data overview

This folder contains the datasets used in the Higgs-transformers project.  
The data originates from the CMS experiment at CERN, and specifically concerns Higgs boson decays into two tau leptons ($H \to \tau^+ \tau^-$).


## Files in this folder

There are three CSV files, which are splits of a single original dataset. The split was necessary because the original file was too large to upload directly to GitHub.

Additionally:
- `dataframe.py` - script used to convert the original `.h5` files into CSV format for easier handling and sharing.

Each row in the dataset corresponds to a collision event recorded by the CMS detector, with features describing the kinematics and properties of the particles involved.


## What the data contains

Key types of variables include:

- Kinematic variables  
  - Tau momenta in Cartesian coordinates (`px`, `py`, `pz`)  
  - Momenta in angular form (`pt`, `eta`, `phi`)  
  - Invariant mass of decay products  

- Charge and particle ID 
  - Tau charge (`±1`)  
  - Decay mode identifiers (e.g., 1-prong, 3-prong)  

- Decay product information  
  - Momenta of visible pions and leptons  
  - Missing transverse energy (MET), associated with neutrinos  

These variables allow for the reconstruction of tau decay planes, polarimetric vectors, and ultimately the extraction of the CP properties of the Higgs boson.


### Additional data used later

In later stages of the project (for improved transformer models), extended datasets were used. These included additional columns with more detailed tau decay product information. These extended files are not included here due to size constraints.
