# Turbulence Jupyter Cloud (TJC) analysis environment
===========================================

## Author: José Hugo Elsas, Mohamed Danish 
## Collaborator: Charles Meneveau, Alexander Szalay

Prototype python analysis environment to be used in conjunction with Sciserver science cloud environment and the Johns Hopkins Turbulence databases.

Related publications: 
   - José Hugo Elsas, Alexander Szalay and Charles Meneveau. “Geometry and Scaling Laws of Excursion and Iso-sets of Enstrophy and Dissipation in Isotropic Turbulence”. Journal of Turbulence (2018).
   
---------------------------------------------------------------------------
# Content 


### Turbulence Jupyter Cloud needs the following libraries and programs installed in your system

   - Numpy, Scipy, Matplotlib - for basic numerical and plotting functionality
   - pyFFTW for fast FFTs
   - pyMP and mpi4py for paralelism
   - YT for (advanced) 3d plotting functionality
   - mpiFFT4py for fast parallel multi-node FFTs
   - pyMorton for implementation of Morton Z-code
