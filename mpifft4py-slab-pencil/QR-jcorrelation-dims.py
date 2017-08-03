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

if rank==0:
    print(FFT.real_shape())

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
    
################################

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

##################################

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
if rank==0:
    print(minE/avg,maxE/avg)

##################################################
    
comm.Barrier()

Q = Q/avg
R = R/(avg**1.5)

###################################################

minQ = np.min(Q); maxQ = np.max(Q)

minQGl=np.zeros(nproc,dtype=FFT.float)
maxQGl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minQ,MPI.DOUBLE],[minQGl,MPI.DOUBLE])
comm.Allgather([maxQ,MPI.DOUBLE],[maxQGl,MPI.DOUBLE])

if rank==0:
    print(minQ,maxQ)

minQ = minQGl.min(); maxQ = maxQGl.max()

##########################################

minR = np.min(R); maxR = np.max(R)

minRGl=np.zeros(nproc,dtype=FFT.float)
maxRGl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minR,MPI.DOUBLE],[minRGl,MPI.DOUBLE])
comm.Allgather([maxR,MPI.DOUBLE],[maxRGl,MPI.DOUBLE])

if rank==0:
    print(minR,maxR)
    
minR = minRGl.min(); maxR = maxRGl.max()

if rank==0:
    for k in range(nproc):
        print(round(minQGl[k],5),round(maxQGl[k],5),round(minRGl[k],5),round(maxRGl[k],5))

if rank==0:
    print(minQ,maxQ,minR,maxR)
    
#############################################

comm.Barrier(); t1=MPI.Wtime()

minJ = -10.0; maxJ =  10.0; E_bins = 100+1
tl = np.linspace(minJ,maxJ,num=E_bins,endpoint=True) 

if rank == 0:
    print("Q and R min/max : ",minQ,maxQ,minR,maxR)

##################################

if rank==0:
    print("Computation Boundaries : ",minJ,maxJ)
    print(tl)
    
lcorr = []; llogr = []; volFr = []

threshold = ((10.0)**3)/((1024.0)**3)

######################################

comm.Barrier(); t1=MPI.Wtime()

for i in range(E_bins-1):
    comm.Barrier(); istart=MPI.Wtime()
    if(rank==0):
        print("Line - "+str(i))
    
    for j in range(E_bins-1):
        comm.Barrier(); jstart=MPI.Wtime()
                
        tQm = tl[i]; tQM = tl[i+1]; tRm = tl[j]; tRM = tl[j+1]
        Index = (Q>tQm)&(Q<tQM)&(R>tRm)&(R<tRM)
        
        chi[:,:,:] = 0
        chi[Index] = 1
        
        vf = np.average(chi)
        vgl = np.zeros(1,dtype=FFT.float)
        comm.Allreduce([vf,MPI.DOUBLE],[vgl,MPI.DOUBLE],op=MPI.SUM)
        vf = vgl/nproc
                    
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
            r2Loc = np.ones(rbins.shape)
                        
        if rank==0:
            volFr.append(vf)
            llogr.append(r2Sum)
            lcorr.append(corrSum)
        
        comm.Barrier(); jend=MPI.Wtime()
        if rank==0:
            print("time for iteration: "+str(jend-jstart))
        
    comm.Barrier(); iend=MPI.Wtime()    
    if rank==0:
        print("time to process line: "+str(iend-istart))

comm.Barrier(); t2=MPI.Wtime()

if rank==0:
    print("Total computing time: "+str(t2-t1))
    
#################################################################

if rank==0:
    eta = 0.00280
    
    rbins = np.linspace(minrt,maxrt,1+ner)    
    bins = (rbins[0:ner]+rbins[1:ner+1])/2
    tempRp = bins[(bins/eta>42.5)&(bins/eta<425)]/eta
    
    fiits = []    
    for i in range(E_bins-1):
        for j in range(E_bins-1):
            tQm = tl[i]; tQM = tl[i+1];            
            tRm = tl[j]; tRM = tl[j+1];
            
            if(volFr[i*(E_bins-1)+j]>threshold):
                tcorr = lcorr[i*(E_bins-1)+j][llogr[i*(E_bins-1)+j]>0]
                tlogr = llogr[i*(E_bins-1)+j][llogr[i*(E_bins-1)+j]>0]
                tbins = bins[llogr[i*(E_bins-1)+j]>0]
                
                corrF = tcorr/tlogr
                tempCorrF = corrF[(tbins/eta>42.5)&(tbins/eta<425)]
                idx = (tempCorrF>0)                
                
                if(len(tempCorrF[idx])>0):
                    fit = np.polyfit(np.log(tempRp[idx]),np.log(tempCorrF[idx]/corrF[0]),1)
                else:
                    fit = np.array([-4,0])
            else:
                fit = np.array([-4,0])
                
            fiits.append(fit[0])
            print('t = ({one:.7f},{two:.7f})*sigma_2: Linear fit [alpha A] = {tree:.3f}'.format(one=(tQm+tQM)/2,two=(tRm+tRM)/2,tree=fit[0]+3))
            
    fiits = np.array(fiits)

if rank==0:
    svolFr = np.array(volFr); sllogr = np.array(llogr);slcorr = np.array(lcorr)
    np.savez("QR-jcorrelation.npz",tl=tl,volFr=svolFr,llogr=sllogr,lcorr=slcorr)
    
if rank==0:
    print(fiits.shape)
    np.savez("QR-jcorrelation-dims.npz",fiits=fiits,E_bins=E_bins,tl=tl)