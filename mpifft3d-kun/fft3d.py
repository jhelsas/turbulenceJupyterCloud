# Created by Mohammad Danish, Johns Hopkins University
# Last modified: October 05, 2016
# Updated by Jose Hugo Elsas, Johns Hopkins University
# Last modified: December 07, 2016
# Based on the original code from Kun Yang, Johns Hopkins University

# This class implements 3D FFT and iFFT of a single field from physical space
# to wave-number space, and back.

# space for importing namespace
# http://stackoverflow.com/questions/6861487/importing-modules-inside-python-class
# better import outside
import pyfftw as ft 
import numpy as np
from mpi4py import MPI
#from MPITranspose import MPITranspose
# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class MPITranspose:    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self,lx,ly,lz):
        self.comm = MPI.COMM_WORLD
        self.tmp1 = np.zeros((lx,lx,1+(lz//2)),dtype='complex64')
        self.tmp2 = np.zeros((lx,lx,1+(lz//2)),dtype='complex64')
        return
    
    def DoMPITranspose(self,v,lx,lz_half,nproc,my_id):
        isize=(lz_half+1)*lx*lx
        
        for i in range(1,nproc):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc

            j=nzp*lx
            j1=(nzp+1)*lx
            self.tmp1[:,0:lx,:] = v[:,j:j1,:]

            req1=self.comm.Isend([self.tmp1,MPI.COMPLEX],dest=nzp,  tag=i)
            req2=self.comm.Irecv([self.tmp2,MPI.COMPLEX],source=nzm,tag=i)
            MPI.Request.Waitall([req1,req2])

            js = nzp*lx
        
            for i in range(lx):
                for j in range(lx):
                    j1=js+j
                    v[i,j1,:] = self.tmp2[j,i,:]
                
        #... diagonal block transpose
        j  =my_id*lx
        j1 =(my_id+1)*lx
        self.tmp1[:,0:lx,:] = v[:,j:j1,:]

        js=my_id*lx
        for i in range(lx):
            for j in range(lx):
                j1=js+j
                v[i,j1,:] = self.tmp1[j,i,:]

        #... adjust the position of blocks which is not on diagonal line
        for i in range(1,nproc//2):
            nzp=(my_id+i)%nproc
            nzm=(my_id-i+nproc)%nproc
            j  =nzp*lx
            j1 =lx*(nzp+1)
            k  =nzm*lx
            k1 =lx*(nzm+1)
            self.tmp1[:,0:lx,:] = v[:,j:j1,:]
            v[:,j:j1,:] = v[:,k:k1,:]
            v[:,k:k1,:] = self.tmp1[:,0:lx,:]

        return

class FFT3Dfield_new:
        
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self,lx,ly,lz,nproc,my_id):
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nproc = nproc
        self.my_id = my_id

        #self.transpose=MPITranspose()
        self.transpose=MPITranspose(lx,ly,lz)
        
        self.temp_v = ft.zeros_aligned((lx,ly,lz), dtype='float32')
        self.temp_cv = ft.zeros_aligned((lx,ly,1+(lz//2)), dtype='complex64')
        
        self.cv = ft.zeros_aligned((lx,ly,1+(lz//2)), dtype='complex64')
        self.v = ft.zeros_aligned((lx,ly,lz), dtype='float32')
        self.cvt = ft.zeros_aligned((lx,ly,1+(lz//2)), dtype='complex64')
        
        self.rfftzf = ft.FFTW(self.temp_v,self.cv,direction='FFTW_FORWARD',axes=(2,),flags=('FFTW_MEASURE',))
        self.fftyf = ft.FFTW(self.cv,self.cv,direction='FFTW_FORWARD',axes=(1,),flags=('FFTW_MEASURE',))

        self.ifftyb = ft.FFTW(self.temp_cv,self.temp_cv,direction='FFTW_BACKWARD',axes=(1,),flags=('FFTW_MEASURE',))
        self.irfftzb = ft.FFTW(self.temp_cv,self.v,direction='FFTW_BACKWARD',axes=(2,),flags=('FFTW_MEASURE',))

        self.alpha2 = 1.0/(float(ly)*float(ly)*float(lz))
        return
            
    def forward3Dfft(self,v,lx,ly,lz,nproc,my_id):
        self.temp_v[:,:,:] = 0.
        self.cv[:,:,:] = np.complex64(0.0+0.0j)

        np.copyto(self.v,v)

        self.rfftzf.update_arrays(self.v,self.cv)
        self.rfftzf.execute()
        
        self.fftyf.update_arrays(self.cv,self.cv)
        self.fftyf.execute() 

        self.transpose.DoMPITranspose(self.cv,lx,lz//2,nproc,my_id)
        
        self.fftyf.execute()

        self.cv[:,:,:]=self.cv[:,:,:]*(self.alpha2)

        return self.cv[:,:,:]

    def backward3Dfft(self,cv,lx,ly,lz,nproc,my_id):
        self.cvt[:,:,:] = np.complex64(0.0+0.0j)
        self.v[:,:,:] = 0.0

        np.copyto(self.cvt,cv)
        
        self.ifftyb.update_arrays(self.cvt,self.cvt)
        self.ifftyb.execute()
        
        self.transpose.DoMPITranspose(self.cvt,lx,lz//2,nproc,my_id)
        
        self.ifftyb.execute()
        self.irfftzb.update_arrays(self.cvt,self.v)        
        self.irfftzb.execute()
        return self.v[:,:,:]


class FourierCorrelation:
	def __init__(self,nx,ny,nz,Nx,Ny,Nz,comm,nproc,rank,fft,dx,dy,dz):
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.Nx = Nx
		self.Ny = Ny
		self.Nz = Nz
		self.nproc = nproc
		self.rank = rank
		self.comm = comm

		self.ner = int(nx*np.sqrt(3))
		self.rbins = np.linspace(-0.5*dx,2*np.pi*np.sqrt(3)+0.5*dx,ner+1)

		self.fft = fft
		
		self.X = np.zeros((nx,ny,nz), dtype='float32')
		self.Y = np.zeros((nx,ny,nz), dtype='float32')
		self.Z = np.zeros((nx,ny,nz), dtype='float32')
		self.r2 = np.zeros((nx,ny,nz), dtype='float32')
		self.r2rt = np.zeros((nx,ny,nz), dtype='float32')
		
		self.chi = ft.zeros_aligned((nx,ny,nz), dtype='float32')
		self.chi2 = ft.zeros_aligned((nx,ny,nz), dtype='float32')
		self.cchi = ft.zeros_aligned((nx,ny,1+(nz//2)), dtype='complex64')
		self.cchi2 = ft.zeros_aligned((nx,ny,1+(nz//2)), dtype='complex64')
		self.corr = ft.zeros_aligned((nx,ny,nz),dtype='float32')
		self.iCorr = ft.zeros_aligned((nx,ny,nz),dtype='float32')
		
		self.corrSum = np.zeros(ner,dtype='float32')
		self.corrF = np.zeros(ner,dtype='float32')  
		self.r2Sum = np.zeros(ner,dtype='float32')
		self.r2F = np.zeros(ner,dtype='float32')

		self.corrApend = np.zeros(ner,dtype='float32') 
		self.r2Apend   = np.zeros(ner,dtype='float32') 
		self.corrLoc = 0.0
		self.r2Loc = 0.0
		return

	def prepareRadialCoordinates():
		for i in range(self.nx):
			if (i+self.rank*self.nx < self.Nx//2):
				self.X[i,:,:] = (i+(self.rank)*(self.nx))*(self.dx)
			else:
				self.X[i,:,:] = Lx-(i+(self.rank)*(self.nx))*(self.dx)

		for j in range(self.ny):
			if (j < self.Ny//2):
				self.Y[:,j,:] = j*(self.dy)
			else:
				self.Y[:,j,:] = Ly-j*(self.dy)

		for k in range(self.nz):
			if (k < self.Nz//2):
				self.Z[:,:,k] = k*(self.dz)
			else:
				self.Z[:,:,k] = Lz-k*(self.dz)

		self.r2[:,:,:] = (self.X[:,:,:])**2+(self.Y[:,:,:])**2+(self.Z[:,:,:])**2
		self.r2rt[:,:,:] = np.sqrt(self.r2[:,:,:])
		return

	def twoPointCorrelation(u,v):
		self.chi[:,:,:]  = u[:,:,:]
		self.chi2[:,:,:] = v[:,:,:]

		self.cchi[:,:,:]  = self.fft.forward3Dfft(chi ,self.nx,self.ny,self.nz,
			                                           self.nproc,self.rank)
		self.cchi2[:,:,:] = self.fft.forward3Dfft(chi2,self.nx,self.ny,self.nz,
			                                           self.nproc,self.rank)

		tmp = self.cchi*(self.cchi2.conj())
		self.corr[:,:,:] = fft.backward3Dfft(tmp,self.nx,self.ny,self.nz,
			                                 self.nproc,self.rank)
		return self.corr[:,:,:]


	def radialIntegration():
		corrLoc,redges = np.histogram(self.r2rt,
			                          range=(0.5*self.dx,(self.ner+0.5)*self.dx),
			                          bins = self.rbins,weights=self.corr)

		r2Loc,r2edges = np.histogram(self.r2rt,
			                         range=(0.5*self.dx,(self.ner+0.5)*self.dx),
			                         bins = self.rbins)

		#######################################

		self.corrLoc = np.float32(corrLoc)
		self.corrSum = 0.0
		self.comm.Allreduce([corrLoc,MPI.REAL],[self.corrSum,MPI.REAL],op=MPI.SUM)
		np.copyto(self.corrF,self.corrSum)

		
		self.corrApend[:] = self.corrF[:]

		#########################

		self.r2Loc = np.float32(r2Loc)
		self.r2Sum = 0.0
		self.comm.Allreduce([r2Loc,MPI.REAL],[self.r2Sum,MPI.REAL],op=MPI.SUM)
		np.copyto(self.r2F,self.r2Sum)

		self.r2Apend[:] = self.r2F[:]

		return self.corrApend[:],r2Apend[:]
		
		#######################################
