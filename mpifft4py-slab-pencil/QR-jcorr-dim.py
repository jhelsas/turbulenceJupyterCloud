import os
import sys
import math
import pyJHTDB
import numpy as np
import pyfftw as ft 
from mpi4py import MPI
from pyJHTDB import libJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse

from mpiFFT4py.slab import R2C

Nx = isotropic1024coarse['nx']; Ny = isotropic1024coarse['ny']; Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']; Ly = isotropic1024coarse['ly']; Lz = isotropic1024coarse['lz']

######################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
if(rank==0):
    print("n_proc = "+str(nproc))
    print("rank = "+str(rank))

######################################
#      Computational Domain          #
######################################

nx=Nx//nproc; ny=Ny; nz=Nz
time = 0.0

########## FFT alocation #############

N = np.array([Nx,Ny,Nz],dtype=int)
L = np.array([Lx,Ly,Lz],dtype=float)

FFT = R2C(N, L, MPI.COMM_WORLD, "double", communication='Alltoallw')

######### Vector alocation ###########

dx = isotropic1024coarse['dx']

comm.Barrier(); t1=MPI.Wtime()

X = np.zeros(FFT.real_shape(), dtype=FFT.float)
Y = np.zeros(FFT.real_shape(), dtype=FFT.float)
Z = np.zeros(FFT.real_shape(), dtype=FFT.float)

r2 = np.zeros(FFT.real_shape(), dtype=FFT.float)

chi = np.zeros(FFT.real_shape(), dtype=FFT.float)
cchi = np.zeros(FFT.complex_shape(), dtype=FFT.complex)
corr = np.zeros(FFT.real_shape(), dtype=FFT.float)
iCorr = np.zeros(FFT.real_shape(), dtype=FFT.float)

r2Sum = np.zeros(FFT.real_shape(), dtype=FFT.float)
r2F = np.zeros(FFT.real_shape(), dtype=FFT.float)

comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    sys.stdout.write('Alocating vectors: {0:.2f} seconds\n'.format(t2-t1))
    
####### Spatial Information #########

comm.Barrier(); t1=MPI.Wtime()
for i in range(nx):
    if (i+nx*rank)<Nx//2:
        X[i,:,:] = (i+nx*rank)*isotropic1024coarse['dx']
    else:
        X[i,:,:] = isotropic1024coarse['lx']-(i+nx*rank)*isotropic1024coarse['dx']
    
for j in range(ny):
    if j<Ny//2:
        Y[:,j,:] = j*isotropic1024coarse['dy']
    else:
        Y[:,j,:] = isotropic1024coarse['ly']-j*isotropic1024coarse['dy']
    
for k in range(nz):
    if k<Nz//2:
        Z[:,:,k] = k*isotropic1024coarse['dz']
    else:
        Z[:,:,k] = isotropic1024coarse['lz']-k*isotropic1024coarse['dz']
    
r2[:,:,:] = X[:,:,:]**2+Y[:,:,:]**2+Z[:,:,:]**2

del X,Y,Z

r2rt = np.sqrt(r2)
del r2

######## Domain boundaries ##########

minrt = r2rt.min()
maxrt = r2rt.max()

minr2Gl=np.zeros(nproc,dtype=FFT.float)
maxr2Gl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minrt,MPI.DOUBLE],[minr2Gl,MPI.DOUBLE])
comm.Allgather([maxrt,MPI.DOUBLE],[maxr2Gl,MPI.DOUBLE])

minrt = minr2Gl.min()
maxrt = maxr2Gl.max()

comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    sys.stdout.write('Preparing the real domain for radial integration: {0:.2f} seconds\n'.format(t2-t1))

ner = int((maxrt-minrt)/isotropic1024coarse['dx'])

rbins = np.linspace(minrt,maxrt,ner+1)

###############################################

###################################
######## Reading Data #############
###################################

folder = '/home/jhelsas/scratch/slab64'

##########

filename = 'ref-Q-'+str(rank)+'.npz'
file = folder + "/" + filename

comm.Barrier(); t1=MPI.Wtime()
content = np.load(file)
    
Q = np.zeros(FFT.real_shape(), dtype=FFT.float)
Q[:,:,:] = content['Q'].astype(FFT.float)
    
comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    print("Finished loading")
    sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))
    
##########

filename = 'ref-R-'+str(rank)+'.npz'
file = folder + "/" + filename

comm.Barrier(); t1=MPI.Wtime()
content = np.load(file)
    
R = np.zeros(FFT.real_shape(), dtype=FFT.float)
R[:,:,:] = content['R'].astype(FFT.float)
    
comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    print("Finished loading")
    sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))

#########

filename = 'ref-strainrate-'+str(rank)+'.npz'
file = folder + "/" + filename

comm.Barrier(); t1=MPI.Wtime()
content = np.load(file)
    
S2 = np.zeros(FFT.real_shape(), dtype=FFT.float)
    
S2[:,:,:] = content['S2'].astype(FFT.float)
    
comm.Barrier(); t2=MPI.Wtime()
if(rank==0):
    print("Finished loading")
    sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))

##################################################
    
#######################################
##### Finding field parameters ########
#######################################

avgE = np.average(S2)
avgEGl=np.zeros(1,dtype=FFT.float)
comm.Allreduce([avgE,MPI.DOUBLE],[avgEGl,MPI.DOUBLE],op=MPI.SUM)
avgE = avgEGl[0]/nproc
if rank == 0:
    print(avgE)
    
avg = avgE

#################################################

minS2 = S2.min(); maxS2 = S2.max()

minS2Gl=np.zeros(nproc,dtype=FFT.float)
maxS2Gl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minS2,MPI.DOUBLE],[minS2Gl,MPI.DOUBLE])
comm.Allgather([maxS2,MPI.DOUBLE],[maxS2Gl,MPI.DOUBLE])

minE = minS2Gl.min(); maxE = maxS2Gl.max()

################################################
######### Finding min and max Q/R ##############
################################################

comm.Barrier()
    
R = R/avg
Q = Q/(avg**(1.5))
    
##################################

minQ = Q.min(); maxQ = Q.max()

minQGl=np.zeros(nproc,dtype=FFT.float)
maxQGl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minQ,MPI.DOUBLE],[minQGl,MPI.DOUBLE])
comm.Allgather([maxQ,MPI.DOUBLE],[maxQGl,MPI.DOUBLE])

minQ = minQGl.min(); maxQ = maxQGl.max()

##################################

minR = R.min(); maxR = R.max()

minRGl=np.zeros(nproc,dtype=FFT.float)
maxRGl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minR,MPI.DOUBLE],[minRGl,MPI.DOUBLE])
comm.Allgather([maxR,MPI.DOUBLE],[maxRGl,MPI.DOUBLE])

minR = minRGl.min(); maxR = maxRGl.max()

comm.Barrier()

##################################

del avgE,S2

comm.Barrier(); t1=MPI.Wtime()

minJ = -10; maxJ =  10; E_bins = 100
tl = np.linspace(minJ,maxJ,num=E_bins,endpoint=True) 

if rank == 0:
    print("Q and R min/max : ",minQ,maxQ,minR,maxR)
    print("Computation Boundaries : ",minJ,maxJ)
    print(tl)

lcorr = []; llogr = []; volFr = []
threshold = 0 #((10.0)**3)/((1024.0)**3)

comm.Barrier(); t1=MPI.Wtime()

######################################
######## Main computation ############
######################################

dt = 0.2
tQ = -3.07
tR = 3

tQm = tQ; tQM = tQ+dt; tRm = tR; tRM = tR+dt
Idx = (Q>tQm)&(Q<tQM)&(R>tRm)&(R<tRM)
if rank==0:
    print(tQm,tQM,tRm,tRM)
        
chi[:,:,:] = 0
chi[Idx] = 1  

comm.Barrier(); t1=MPI.Wtime()     
    
vf = np.sum(chi)
vgl = np.zeros(1,dtype=np.int64)
comm.Allreduce([vf,MPI.INT64_T],[vgl,MPI.INT64_T],op=MPI.SUM)
vf = vgl

if rank==0:
    print(vf)

comm.Barrier(); t1=MPI.Wtime()

if vf>threshold:
    cchi = FFT.fftn(chi,cchi)
    tmp = cchi*(cchi.conj())
    corr = FFT.ifftn(tmp,corr)
    corr[:,:,:] = corr[:,:,:]/(Nx*Ny*Nz)
    
    corrLoc,redges = np.histogram(r2rt,bins = rbins,weights=corr)
    r2Loc,r2edges = np.histogram(r2rt,bins = rbins)
    
    corrSum = np.zeros(corrLoc.shape,dtype=corrLoc.dtype)
    comm.Allreduce([corrLoc,MPI.DOUBLE],[corrSum,MPI.DOUBLE],op=MPI.SUM)
    r2Sum = np.zeros(r2Loc.shape,dtype=r2Loc.dtype)
    comm.Allreduce([r2Loc,MPI.DOUBLE],[r2Sum,MPI.DOUBLE],op=MPI.SUM)
else:
    corrSum = np.zeros(rbins.shape)
    r2Sum = np.ones(rbins.shape)
    
comm.Barrier(); t1=MPI.Wtime()

######################################
######## Ploting the data ############
######################################

if rank==0:
    print("dumping and exiting");

#os.remove('dump.npz')
np.savez('dump.npz',vf=vf,corrSum=corrSum,r2Sum=r2Sum,minrt=minrt,maxrt=maxrt,ner=ner,tQ=tQ,tR=tR,dt=dt)