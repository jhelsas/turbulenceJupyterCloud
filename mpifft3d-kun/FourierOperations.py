# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import pyfftw as ft 
import numpy as np
from mpi4py import MPI

class FourierOperations:
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self,nx,ny,nz,nproc,my_id):
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.nproc=nproc
        self.my_id=my_id
        return
    
    def GetWavenumbers(self):
        Nf=(self.ny//2)+1
        sx=slice(self.my_id*self.nx,(self.my_id+1)*self.nx)
        ky=np.fft.fftfreq(self.ny,1./self.ny).astype(int)
        kz=ky[:Nf].copy();kz[-1]*=-1
        K=np.array(np.meshgrid(ky[sx],ky,kz,indexing='ij'),dtype=int)
        # NOTE: In FFT, the x-axis is rotated towards the y-axis, so kx should also be
        k2=K[1]*K[1]+K[0]*K[0]+K[2]*K[2]
        return k2,K[1],K[0],K[2] # returns: wavenumber grids in x,y,z directions respectively
    
    # 3D Real to Complex FFT
    def GetFFT3Dfield(self,v):
        nz_half=self.nz//2
        cv=ft.zeros_aligned((self.nx,self.ny,nz_half+1),dtype='complex64')
        temp_v=ft.zeros_aligned((self.nx,self.ny,self.nz), dtype='float32')
        
        # The FFTW instance kills the input array, so don't pass the actual array here.
        rfftzf=ft.FFTW(temp_v,cv,direction='FFTW_FORWARD',axes=(2,),flags=('FFTW_MEASURE',))        
        fftyf =ft.FFTW(cv,cv,direction='FFTW_FORWARD',axes=(1,),flags=('FFTW_MEASURE',))
        
        rfftzf.update_arrays(v,cv)
        rfftzf.execute()
        fftyf.update_arrays(cv,cv)
        fftyf.execute()        
        self.DoMPITranspose(cv,self.nx,nz_half,self.nproc,self.my_id)
        fftyf.execute()

        alpha2=1.0/(float(self.ny)*float(self.ny)*float(self.ny))
        cv[:,:,:]=cv[:,:,:]*alpha2
        return cv
    
    # 3D Complex to Real FFT (Inverse FFT)
    def GetIFFT3Dfield(self,cv):
        nz_half=self.nz//2
        temp_cv=ft.zeros_aligned((self.nx,self.ny,nz_half+1), dtype='complex64')        
        v=ft.zeros_aligned((self.nx,self.ny,self.nz), dtype='float32')
        # The FFTW instance kills the input array, so don't pass the actual array here.
        fftyb =ft.FFTW(temp_cv,temp_cv,direction='FFTW_BACKWARD',axes=(1,),flags=('FFTW_MEASURE',))
        rfftzb=ft.FFTW(temp_cv,v,direction='FFTW_BACKWARD',axes=(2,),flags=('FFTW_MEASURE',))
        
        cvt=ft.zeros_aligned((self.nx,self.ny,nz_half+1), dtype='complex64')
        np.copyto(cvt,cv)
        
        fftyb.update_arrays(cvt,cvt)
        fftyb.execute()
        self.DoMPITranspose(cvt,self.nx,nz_half,self.nproc,self.my_id)
        
        fftyb.execute()
        rfftzb.update_arrays(cvt,v)
        rfftzb.execute()
        return v
    
    def GetFilteredField(self,field,k2,k_cutoff,fltrType):
        if fltrType == 'gaussian':
            fltrd_field=self.GaussianFilter(field,k2,k_cutoff)
        elif fltrType == 'sharp':
            fltrd_field=self.SharpFilter(field,k2,k_cutoff)
        elif fltrType == 'box':
            fltrd_field=self.BoxFilter(field,k2,k_cutoff)
        else:
            ## WARNING: Gaussian is a default filter
            fltrd_field=self.GaussianFilter(field,k2,k_cutoff)
        return fltrd_field
    
    # FILTERS: see "Sagaut, P., Large Eddy Simulation for Incompressible Flows, Ch. 2, pp.21-23"
    def GaussianFilter(self,field,k2,k_cutoff):
        fltr_width=np.pi/k_cutoff
        fltrd_field=field*np.exp(-k2*fltr_width*fltr_width/24.0)
        return fltrd_field
            
    def SharpFilter(self,field,k2,k_cutoff):
        k=np.sqrt(k2)
        fltrd_field=field.astype(field.dtype)
        fltrd_field[k>k_cutoff]=0.0
        return fltrd_field
    
    def BoxFilter(self,field,k2,k_cutoff):
        fltr_width=np.pi/k_cutoff
        k=np.sqrt(k2)
        fltrd_field=field*np.sin(k*fltr_width/2.0)/(k*fltr_width/2.0)
        return fltrd_field
    
    def ConvertDNS2Gaussian(self,field):
        gaussianField=np.zeros_like(field)
        randPhase=np.pi*np.random.random(np.shape(field))
        gaussianField[:,:,:]=(np.cos(randPhase)+1j*np.sin(randPhase))*field
        return gaussianField
    
    def DoMPITranspose(self,v,lx,lz_half,nproc,my_id):
        comm=MPI.COMM_WORLD
        isize=(lz_half+1)*lx*lx
        tmp1=np.zeros((lx,lx,lz_half+1),dtype='complex64')
        tmp2=np.zeros((lx,lx,lz_half+1),dtype='complex64')
    
        for i in range(1,nproc):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc

            j=nzp*lx
            j1=(nzp+1)*lx
            tmp1[:,0:lx,:] = v[:,j:j1,:]

            req1=comm.Isend([tmp1,MPI.COMPLEX],dest=nzp,  tag=i)
            req2=comm.Irecv([tmp2,MPI.COMPLEX],source=nzm,tag=i)
            MPI.Request.Waitall([req1,req2])

            js = nzp*lx
        
            for i in range(lx):
                for j in range(lx):
                    j1=js+j
                    v[i,j1,:] = tmp2[j,i,:]
                
        #... diagonal block transpose
        j  =my_id*lx
        j1 =(my_id+1)*lx
        tmp1[:,0:lx,:] = v[:,j:j1,:]

        js=my_id*lx
        for i in range(lx):
            for j in range(lx):
                j1=js+j
                v[i,j1,:] = tmp1[j,i,:]

        #... adjust the position of blocks which is not on diagonal line
        for i in range(1,nproc//2):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc
            j  =nzp*lx
            j1 =lx*(nzp+1)
            k  =nzm*lx
            k1 =lx*(nzm+1)
            tmp1[:,0:lx,:] = v[:,j:j1,:]
            v[:,j:j1,:] = v[:,k:k1,:]
            v[:,k:k1,:] = tmp1[:,0:lx,:]
        
    
        del tmp1
        del tmp2
        return