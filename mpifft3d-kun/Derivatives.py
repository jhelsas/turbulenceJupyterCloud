# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import numpy as np
from mpi4py import MPI
import math
from FourierOperations import FourierOperations

class Derivatives:
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self,nx,ny,nz,nproc,my_id):
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.nproc=nproc
        self.my_id=my_id
        self.myFO=FourierOperations(nx,ny,nz,nproc,my_id)
        return
    
    def GetDPhiDXhi_C2R(self,cphi,kappa):
        cphi[kappa==(self.ny//2)]=0 # see http://math.mit.edu/~stevenj/fft-deriv.pdf
        temp=np.complex64(0.0+1.0j)*(kappa*cphi)
        DphiDXhi=self.myFO.GetIFFT3Dfield(temp)
        return DphiDXhi
    
    def GetDPhiDXhi_C2C(self,cphi,kappa):
        cphi[kappa==(self.ny//2)]=0 # see http://math.mit.edu/~stevenj/fft-deriv.pdf
        cDphiDXhi=np.complex64(0.0+1.0j)*(kappa*cphi)
        return cDphiDXhi