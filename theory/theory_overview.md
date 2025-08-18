# Theoretical overview
 
This project studies how the Higgs boson decays into two tau leptons and uses Machine Learning to recover information that is otherwise hidden from direct measurement.


## CP Symmetry and the Higgs boson

- CP symmetry combines two fundamental transformations:
  - C (charge conjugation): swap particles with their antiparticles.  
  - P (parity): flip spatial coordinates.  

In the Standard Model, the Higgs boson is predicted to be CP-even (it looks the same under CP). If the Higgs had a CP-odd or mixed CP nature, it would signal new physics beyond the Standard Model — potentially helping to explain the matter–antimatter asymmetry in the universe.  


## Decay planes and polarimetric vectors

- Each tau decay defines a decay plane, spanned by the directions of its decay products.  
- From the momenta of visible particles, we construct a polarimetric vector, strongly correlated with the tau’s spin direction.  
- The angle between the two decay planes is called the *acoplanarity angle* ($\phi_{CP}$), which is sensitive to the CP properties of the Higgs boson.

<img width="300" height="566" alt="image" src="https://github.com/user-attachments/assets/db531059-bd17-4003-8728-1922820702d1" />

- $h^+$ and $h^-$: polarimetric vectors, representing the spin information of the two taus. The angle $\phi_{CP}$ between the planes is the key observable for determining whether the Higgs is CP-even, CP-odd, or CP-mixed.  


## CP-sensitive distributions

<img width="500" height="590" alt="image" src="https://github.com/user-attachments/assets/a3437f5d-2887-431e-9fcf-ebc5657466d5" />

- Expected acoplanarity angle distributions for different Higgs CP states:  
  - CP-even 
  - CP-odd 
  - CP-mixed 
The peak of the distribution corresponds to the CP mixing angle $\phi_{\tau\tau}$. By measuring this peak, we can deduce the CP properties of the Higgs boson.

### Using Generated Momenta
<img width="500" height="1440" alt="image" src="https://github.com/user-attachments/assets/e05cc324-d8b6-4f99-9491-47b508460c30" />

- Same distributions, but constructed using the generated tau momenta from the dataset.  
- This is the ideal reference case, as if we had access to the true tau kinematics.  
- The goal of ML models is to reproduce these distributions from detector-level data.


## Recovering true momenta

The CMS detector measures the visible decay products (like pions) but misses neutrinos, which escape undetected. This means the true tau momenta cannot be measured directly. To analyze CP symmetry, we need as close an estimate of the true momenta as possible.  

Simplifying assumption:  
In this project we assume the direction of the tau momentum is reconstructed correctly, and focus on recovering one missing component — the transverse momentum ($p_T$).  This makes the problem well-suited for machine learning regression.


## Machine Learning in this project

The project applies a neural network and transformer-based models to learn the mapping between measured quantities (visible tau momenta, missing energy, angles, etc.) and true/generated tau momenta.  

The recovered momenta are then used to:
- build decay planes  
- construct polarimetric vectors  
- extract the CP-sensitive $\phi_{CP}$ distribution  

The scripts used to perform those steps came from the CMS group I worked with, and they are not included in this repository.
Transformer-based models were found to perform significantly better than simpler neural networks in recovering CP-sensitive distributions.


## References

- CMS Collaboration, *Analysis of the CP structure of the Yukawa coupling between the Higgs boson and τ leptons in proton-proton collisions at √s = 13 TeV*.  
- ATLAS Collaboration, *Measurement of the CP properties of Higgs boson interactions with τ-leptons with the ATLAS detector*.  
- Cardini, A., *Measurement of the CP properties of the Higgs boson in its decays to τ leptons with the CMS experiment* (Dissertation).  
