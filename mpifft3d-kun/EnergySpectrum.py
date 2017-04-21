# DESCRIPTION OF THE CLASS:
# Adding one more description line
# PROVIDE EXAMPLE:
    
import numpy as np
from mpi4py import MPI

class EnergySpectrum:    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
    
    def GetSpectrumFromComplexField(self,cvx,cvy,cvz,k2,lx,ly,lz,nek,nproc,my_id):
        k2[0,0,0]=1e-6
        ek=self.cal_newspec(cvx,cvy,cvz,k2,nek)
        return ek
            
    def FindWavenumber(self,lx,ly,lz,my_id):
        lz_half=lz//2
        
        kx=np.zeros((lx,ly,lz_half+1), dtype='float32')
        ky=np.zeros((lx,ly,lz_half+1), dtype='float32')
        kz=np.zeros((lx,ly,lz_half+1), dtype='float32')
        for i in range(lx):
            ky[i,:,:]=(i+ly//2+lx*my_id)%ly-ly//2
        for j in range(ly):
            kz[:,j,:]=(j+ly//2)%ly-ly//2;
        for k in range(lz_half+1):
            kx[:,:,k]=k

        k2=np.zeros((lx,ly,lz_half+1), dtype='float32')
        np.copyto(k2,kx*kx+ky*ky+kz*kz)
        k2[0,0,0]=1e-6
        return k2, kx, ky, kz
    
    def GetWavenumbers(self,nx,ny,nz,my_id):
        Nf=(ny//2)+1
        sx=slice(my_id*nx,(my_id+1)*nx)
        ky=np.fft.fftfreq(ny,1./ny).astype(int)
        kz=ky[:Nf].copy();kz[-1]*=-1
        K=np.array(np.meshgrid(ky[sx],ky,kz,indexing='ij'),dtype=int)
        # NOTE: In FFT, the x-axis is rotated towards the y-axis, so kx should also be
        k2=K[1]*K[1]+K[0]*K[0]+K[2]*K[2]
        return k2,K[1],K[0],K[2] # returns: wavenumber grids in x,y,z directions respectively
    
    def cal_newspec(self,cvx,cvy,cvz,k2,nek):
        tmp=(cvx*cvx.conj()+cvy*cvy.conj()+cvz*cvz.conj()).real
        tmp[:,:,0]=0.5*tmp[:,:,0]

        ekbins=np.linspace(0.5,nek+0.5,nek+1)
        k2rt=np.sqrt(k2)
        ekloc,bins=np.histogram(k2rt,range=(0.5,nek+0.5),bins=ekbins,weights=tmp)
        del k2rt
        del tmp

        ekloc=np.float32(ekloc)
        eksum=np.zeros(nek,dtype='float32')
        comm = MPI.COMM_WORLD
        comm.Reduce([ekloc,MPI.REAL],[eksum,MPI.REAL],op=MPI.SUM)
        
        ek=np.zeros(nek,dtype='float32')
        np.copyto(ek,eksum)
        return ek
    
    def cal_spec(self,cvx,cvy,cvz,k2,nek):
        tmp=(cvx*cvx.conj()+cvy*cvy.conj()+cvz*cvz.conj()).real
        tmp[:,:,0]=0.5*tmp[:,:,0]
        ekloc=np.zeros(nek+1,dtype='float32')
        eksum=np.zeros(nek+1,dtype='float32')

        comm = MPI.COMM_WORLD
        ks=np.arange(1,nek+1)
        for i in range(1,nek+1):
            ekloc[i]=np.sum(tmp[np.floor(np.sqrt(k2)+0.5)==i])
        comm.Reduce([ekloc,MPI.REAL],[eksum,MPI.REAL],op=MPI.SUM)

        ek=np.zeros(nek,dtype='float32')
        np.copyto(ek,eksum)
        return ek