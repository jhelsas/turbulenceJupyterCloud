
import os
import sys
import math
import pyJHTDB
import numpy as np
import pyfftw as ft 
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
from pyJHTDB import libJHTDB
from pyJHTDB.dbinfo import isotropic1024coarse

from mpiFFT4py.slab import R2C

Nx = isotropic1024coarse['nx']; Ny = isotropic1024coarse['ny']; Nz = isotropic1024coarse['nz']
Lx = isotropic1024coarse['lx']; Ly = isotropic1024coarse['ly']; Lz = isotropic1024coarse['lz']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
if(rank==0):
    print("n_proc = "+str(nproc))
    print("rank = "+str(rank))

# Computational Domain

nx=Nx//nproc; ny=Ny; nz=Nz
nz_half=nz//2
nek=int(math.sqrt(2.0)/3*Nx)
time = 0.0

chkSz = 32
slabs = nx//chkSz

N = np.array([Nx,Ny,Nz],dtype=int)
L = np.array([Lx,Ly,Lz],dtype=float)

FFT = R2C(N, L, MPI.COMM_WORLD, "double", communication='Alltoallw')

dx = isotropic1024coarse['dx']
ner = int(1024*np.sqrt(3))

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

comm.Barrier(); t1=MPI.Wtime()
for i in range(nx):
    X[i,:,:] = (i+rank*nx)*isotropic1024coarse['dx']

for j in range(ny):
    Y[:,j,:] = j*isotropic1024coarse['dy']
    
for k in range(nz):
    Z[:,:,k] = k*isotropic1024coarse['dz']
    
r2[:,:,:] = X[:,:,:]**2+Y[:,:,:]**2+Z[:,:,:]**2

r2rt = np.sqrt(r2)

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

cacheEnstrophyData = False
loadEnstrophyFromCache = True

folder = "/home/admin/scratch/slab64"
filename = "aws-enstrophy-"+str(rank)+".npz"
file = folder + "/" + filename

if(loadEnstrophyFromCache):
    comm.Barrier(); t1=MPI.Wtime()
    content = np.load(file)
    
    w2 = ft.zeros_aligned(FFT.real_shape(), dtype=FFT.float)
    
    #if(int(content['nproc'])!=nproc):
    #    print("Unmatched number of processes. Must first pre-process to adequate number of process")
    w2[:,:,:] = content['w2'].astype(FFT.float)
    
    comm.Barrier(); t2=MPI.Wtime()
    if(rank==0):
        print("Finished loading")
        sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))

if(cacheEnstrophyData):
    
    comm.Barrier(); t1=MPI.Wtime()
    np.savez(file,w2=w2,nproc=nproc)
    comm.Barrier(); t2=MPI.Wtime()
    if(rank==0):
        sys.stdout.write('Caching the data: {0:.2f} seconds\n'.format(t2-t1))

cacheStrainrateData = False
loadStrainrateFromCache = True

folder = "/home/admin/scratch/slab64"
filename = "aws-strainrate-"+str(rank)+".npz"
file = folder + "/" + filename

if(loadStrainrateFromCache):
    comm.Barrier(); t1=MPI.Wtime()
    content = np.load(file)
    
    S2 = ft.zeros_aligned(FFT.real_shape(), dtype=FFT.float)
    
    #if(int(content['nproc'])!=nproc):
    #    print("Unmatched number of processes. Must first pre-process to adequate number of process")
    S2[:,:,:] = content['S2'].astype(FFT.float)
    
    comm.Barrier(); t2=MPI.Wtime()
    if(rank==0):
        print("Finished loading")
        sys.stdout.write('Load from disk: {0:.2f} seconds\n'.format(t2-t1))

if(cacheEnstrophyData):
    
    comm.Barrier(); t1=MPI.Wtime()
    np.savez(file,S2=S2,nproc=nproc)
    comm.Barrier(); t2=MPI.Wtime()
    if(rank==0):
        sys.stdout.write('Caching the data: {0:.2f} seconds\n'.format(t2-t1))

w2[:,:,:] = 0.5*w2[:,:,:]

avgO = np.average(w2)
avgOGl=np.zeros(1,dtype=FFT.float)

comm.Allreduce([avgO,MPI.DOUBLE],[avgOGl,MPI.DOUBLE],op=MPI.SUM)
avgO = avgOGl[0]/nproc

########

avgE = np.average(S2)
avgEGl=np.zeros(1,dtype=FFT.float)

comm.Allreduce([avgE,MPI.DOUBLE],[avgEGl,MPI.DOUBLE],op=MPI.SUM)
avgE = avgEGl[0]/nproc

########

if rank == 0:
    print(avgO,avgE,(avgE-avgO)/avgO)
    
avg = avgE

##########################

minw2 = w2.min()
maxw2 = w2.max()

minwGl=np.zeros(nproc,dtype=FFT.float)
maxwGl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minw2,MPI.DOUBLE],[minwGl,MPI.DOUBLE])
comm.Allgather([maxw2,MPI.DOUBLE],[maxwGl,MPI.DOUBLE])

minO = minwGl.min()
maxO = maxwGl.max()

comm.Barrier()

##########################

minS2 = S2.min()
maxS2 = S2.max()

minS2Gl=np.zeros(nproc,dtype=FFT.float)
maxS2Gl=np.zeros(nproc,dtype=FFT.float)

comm.Allgather([minS2,MPI.DOUBLE],[minS2Gl,MPI.DOUBLE])
comm.Allgather([maxS2,MPI.DOUBLE],[maxS2Gl,MPI.DOUBLE])

minE = minS2Gl.min()
maxE = maxS2Gl.max()

comm.Barrier()

minJ = min(minO,minE)
maxJ = max(maxO,maxE)

if rank == 0:
    print("Separate : ",minO/avg,maxO/avg,minE/avg,maxE/avg)
    print("Joint : ",minJ/avg,maxJ/avg)

comm.Barrier()

if rank==0:
    print("<w^2> : "+str(avgO))
    print("min w2/<w^2> : "+str(minw2/avg))
    print("min w2/<w^2> : "+str(maxw2/avg))
    print("<w^2> : "+str(avgE))
    print("min w2/<w^2> : "+str(minw2/avg))
    print("min w2/<w^2> : "+str(maxw2/avg))
    
if rank==0:
    print("log: ",np.log(minJ/avg),np.log(maxJ/avg))
    print("log_10: ",np.log(minJ/avg)/np.log(10),np.log(maxJ/avg)/np.log(10))

lcorr = []
llogr = []
volFr = []

comm.Barrier(); t1=MPI.Wtime()

######################################

dt = 0.118441158993
t = avg
        
tOm = t
tOM = t*(1+dt)
        
tEm = t
tEM = t*(1+dt)
        
Index = (w2>tOm)&(w2<tOM)&(S2>tEm)&(S2<tEM)
        
chi[:,:,:] = 0
chi[Index] = 1

vf = np.average(chi)
vgl = np.zeros(1,dtype=FFT.float)
comm.Allreduce([vf,MPI.DOUBLE],[vgl,MPI.DOUBLE],op=MPI.SUM)
vf = vgl 

volFr.append(vf)
if vf<=0.:
    corrSum = np.zeros(rbins.shape)
    r2Loc = np.ones(rbins.shape)
else:
    cchi = FFT.fftn(chi,cchi)
    tmp = cchi*(cchi.conj())
    corr[:,:,:] = corr[:,:,:]/(Nx*Ny*Nz)
            
    corrLoc,redges = np.histogram(r2rt,bins = rbins,weights=corr)
    r2Loc,r2edges = np.histogram(r2rt,bins = rbins)
            
    corrSum = np.zeros(corrLoc.shape,dtype=corrLoc.dtype)
    comm.Allreduce([corrLoc,MPI.DOUBLE],[corrSum,MPI.DOUBLE],op=MPI.SUM)
    r2Sum = np.zeros(r2Loc.shape,dtype=r2Loc.dtype)
    comm.Allreduce([r2Loc,MPI.DOUBLE],[r2Sum,MPI.DOUBLE],op=MPI.SUM)

if rank==0:
    volFr.append(vf)
    llogr.append(r2Loc)
    lcorr.append(corrLoc)
            
comm.Barrier(); t2=MPI.Wtime()

if rank==0:
    print("Single iteration timing: ",t2-t1)

E_bins = 115
dt = 0.118441158993
tl = np.logspace(np.log(minJ),np.log(maxJ),num=E_bins,endpoint=True,base=np.e) 

if rank==0:
    print(np.log(tl/avg)/np.log(10))

lcorr = []
llogr = []
volFr = []

comm.Barrier(); t1=MPI.Wtime()

######################################

for i in range(E_bins-1):
    comm.Barrier(); istart=MPI.Wtime()
    for j in range(E_bins-1):
        comm.Barrier(); jstart=MPI.Wtime()
        
        tOm = tl[i]
        tOM = tl[i+1]
        
        tEm = tl[j]
        tEM = tl[j+1]
        
        Index = (w2>tOm)&(w2<tOM)&(S2>tEm)&(S2<tEM)
        
        chi[:,:,:] = 0
        chi[Index] = 1
        
        vf = np.average(chi)
        vgl = np.zeros(1,dtype=FFT.float)
        comm.Allreduce([vf,MPI.DOUBLE],[vgl,MPI.DOUBLE],op=MPI.SUM)
        vf = vgl/nproc
                    
        if vf<=0.:
            corrSum = np.zeros(rbins.shape)
            r2Loc = np.ones(rbins.shape)
        else:
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

if rank==0:
    for i in range(E_bins-1):
        for j in range(E_bins-1):
            print(volFr[i*(E_bins-1)+j].shape)
            print(lcorr[i*(E_bins-1)+j].shape)
            print(llogr[i*(E_bins-1)+j].shape)

if rank==0:
    eta = 0.00280
    
    rbins = np.linspace(minrt,maxrt,1+ner)    
    bins = (rbins[0:ner]+rbins[1:ner+1])/2
    tempRp = bins[(bins/eta>42.5)&(bins/eta<425)]/eta
    
    fiits = []    
    for i in range(E_bins-1):
        for j in range(E_bins-1):
            tOm = tl[i]
            tOM = tl[i+1]
            
            tEm = tl[j]
            tEM = tl[j+1]
            
            if(volFr[i*(E_bins-1)+j]>0):
                tcorr = lcorr[i*(E_bins-1)+j][llogr[i*(E_bins-1)+j]>0]
                tlogr = llogr[i*(E_bins-1)+j][llogr[i*(E_bins-1)+j]>0]
                tbins = bins[llogr[i*(E_bins-1)+j]>0]
                
                corrF = tcorr/tlogr
                tempCorrF = corrF[(tbins/eta>42.5)&(tbins/eta<425)]
                idx = (tempCorrF>0)                
                
                if(len(tempCorrF[idx])>0):
                    fit = np.polyfit(np.log(tempRp[idx]),np.log(tempCorrF[idx]/corrF[0]),1)
                else:
                    fit = np.array([-3,0])
            else:
                fit = np.array([-3,0])
                
            fiits.append(fit[0])
            print('t = ({one:.7f},{two:.7f})*sigma_2: Linear fit [alpha A] = {tree:.3f}'.format(one=np.log(np.sqrt(tOm*tOM)/avg)/np.log(10),two=np.log(np.sqrt(tEm*tEM)/avg)/np.log(10),tree=fit[0]+3))
            
    fiits = np.array(fiits)

if rank==0:
    print(fiits.shape)
    np.savez("joint-corr-dims.npz",fiits=fiits,E_bins=E_bins,tl=tl,dt=dt)

if rank==0: 
    pfiits = np.reshape(fiits,(E_bins-1,E_bins-1))
    pfiits = pfiits+3
    pfiits[pfiits==0.] = np.nan
    
    fig = plt.figure(figsize=(12,12))
    
    plt.title(r'$D(\chi_\omega,\chi_\epsilon)$',size=20)
    plt.ylabel(r'$\log_{10}{( S^2/\langle S^2\rangle)}$',size=20)
    plt.xlabel(r'$\log_{10}{( (\omega^2/2)/\langle S^2\rangle)}$',size=20)
    
    plt.xlim([-7.,3.])
    plt.ylim([-7.,3.])
    
    plt.grid()
    
    bmin = np.log(np.sqrt(tl[0]*tl[1])/avg)/np.log(10)
    bmax = np.log(np.sqrt(tl[E_bins-2]*tl[E_bins-1])/avg)/np.log(10)
    print(bmin,bmax)
    cax = plt.imshow(pfiits.T,interpolation='None',extent=(bmin,bmax,bmin,bmax),origin='top',aspect='equal',vmin=0.,vmax=3.)
    
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    plt.savefig('joint-dimension-computation.pdf', format='pdf')