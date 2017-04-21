# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import numpy as np
from mpi4py import MPI
import math
from Derivatives import Derivatives
from TensorTransformation import TensorTransformation

class QuantitiesOfInterest:
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self,nx,ny,nz,nproc,my_id):
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.nproc=nproc
        self.my_id=my_id
        return
    
    def GetEnergySpectrum(self,cvx,cvy,cvz,k2):
        nek=int((math.sqrt(2.0)/3.)*(self.nx*self.ny*self.nz*self.nproc)**(1./3.))
        k2[0,0,0]=1e-6
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
    
    # Definition of velocity gradient tensor: A_{ij}=\partial{V_i}/\partial{X_j}
    def GetVelocityGradient(self,cvi,kj):
        myDeriv=Derivatives(self.nx,self.ny,self.nz,self.nproc,self.my_id)
        Aij=myDeriv.GetDPhiDXhi_C2R(cvi,kj)
        return Aij.flatten()
    
    # R=0.5(A_{ij}-A_{ji})
    #        _        _        _           _
    #       |0  -w3  w2|      |a11  a12  a13|
    # R=0.5*|w3  0  -w1|    A=|a21  a22  a23|
    #       |-w2 w1  0 |      |a31  a32  a33|
    #        --      --        --         --
    def GetVorticity(self,a12,a21,a13,a31,a23,a32):
        w1=(a32-a23).flatten()
        w2=(a13-a31).flatten()
        w3=(a21-a12).flatten()
        return w1,w2,w3
    
    # Q=-(1/2)(SijSji+RijRji)
    def GetQ(self,a11,a12,a13,a21,a22,a23,a31,a32,a33):
        if((len(a11.shape)*len(a12.shape)*len(a13.shape)*len(a21.shape)*len(a22.shape)*len(a23.shape)*\
            len(a31.shape)*len(a32.shape)*len(a33.shape))>1.0):
            print('Change the input arrays of GetQ function to 1D!!')
            return
        #sijsji=s11*s11+s22*s22+s33*s33+2.0*(s12*s12+s13*s13+s23*s23)
        #rijrji=-2.0*(r12*r12+r23*r23+r13*r13)        
        sijsji=a11*a11+a22*a22+a33*a33+0.5*((a12+a21)**2+(a13+a31)**2+(a23+a32)**2)
        rijrji=-0.5*((a12-a21)**2+(a13-a31)**2+(a23-a32)**2)
        return -(1./2.)*(sijsji+rijrji)
    
    # r=-(1/3)(SijSjkSki+3RijRjkSki)
    def GetR(self,a11,a12,a13,a21,a22,a23,a31,a32,a33):
        if((len(a11.shape)*len(a12.shape)*len(a13.shape)*len(a21.shape)*len(a22.shape)*len(a23.shape)*\
            len(a31.shape)*len(a32.shape)*len(a33.shape))>1.0):
            print('Change the input arrays of GetR function to 1D!!')
            return
        #SijSjkSki=s11*s11*s11+s22*s22*s22+s33*s33*s33+3.0*(s12*s12*(s11+s22)+s13*s13*(s11+s33)+s23*s23*(s22+s33))+6.0*s12*s23*s13
        #RijRjkSki=-r12*r12*(s11+s22)-r13*r13*(s11+s33)-r23*r23*(s22+s33)-2.*s23*r12*r13-2.*s12*r13*r23+2.*s13*r12*r23        
        SijSjkSki=a11**3+a22**3+a33**3+0.75*((a12+a21)**2*(a11+a22)+(a13+a31)**2*(a11+a33)+(a23+a32)**2*(a22+a33))\
        +0.75*(a12+a21)*(a13+a31)*(a23+a32)
        RijRjkSki=-0.25*((a12-a21)**2*(a11+a22)+(a13-a31)**2*(a11+a33)+(a23-a32)**2*(a22+a33))\
        -0.25*((a23+a32)*(a12-a21)*(a13-a31)+(a12+a21)*(a13-a31)*(a23-a32)-(a13+a31)*(a12-a21)*(a23-a32))
        return -(1./3.)*(SijSjkSki+3.*RijRjkSki)
    
    # Ref: Ooi et.al.(1999),JFM,"A study of the evolution and characteristics..."
    # Qw=-(1/2)RijRji [used for normalizing the Q and R]
    def GetQwAvg(self,a12,a21,a13,a31,a23,a32):
        if((len(a12.shape)*len(a21.shape)*len(a13.shape)*len(a31.shape)*len(a23.shape)*len(a32.shape))>1.0):
            print('Change the input arrays of GetQwAvg function to 1D!!')
            return
        #Qw=r12*r12+r13*r13+r23*r23
        Qw=0.25*((a12-a21)**2+(a13-a31)**2+(a23-a32)**2)
        
        locsum=np.float32(np.sum(Qw))
        Qw_avg=np.zeros(1,dtype='float32')
        
        comm=MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[Qw_avg,MPI.REAL],op=MPI.SUM)
        Qw_avg=comm.bcast(Qw_avg,root=0)

        Qw_avg=Qw_avg/(self.nproc*len(Qw))
        return Qw_avg
    
    def GetStrainrateVortAlign(self,a11,a12,a13,a21,a22,a23,a31,a32,a33):
        myTT=TensorTransformation()
        v1,v2,v3=self.GetVorticity(a12,a21,a13,a31,a23,a32)
        cosalfa,cosbeta,cosgama=myTT.CosVectorTensor(v1,v2,v3,a11,a22,a33,\
                                                     0.5*(a12+a21),0.5*(a13+a31),0.5*(a23+a32))
        return cosalfa,cosbeta,cosgama
    
    #    Ref. Chong et al. (1990); Suman & Girimaji (2010)
    #        ___________________
    #       |         |         |
    #       |   SFS   |   UFC   |
    #    ^  |    (0)  |   (5)   |
    #    |  |         |         | s1=>+(2/27)*((-3Q)^(3/2))-r=0
    #    Q 0|_________|_________| s2=>-(2/27)*((-3Q)^(3/2))-r=0
    #       |        /|\    UFC |
    #       | SFS   / | \   (4) |
    #       | (1) s2  |  s1     |
    #       |     /   |   \   *<-----s1<0
    # s2>0---->* / (2)|(3) \    |
    #       |___/_____|_____\___|
    #          SN/S/S 0 UN/S/S
    #                R-->
    # OUTPUT: topoFraction[0:5], where index => 0->SFS(Q>0),1->SFS(Q<0),2->SNSS,3->UNSS,4->UFC(Q<0),5->UFC(Q>0)
    def GetTopologyFraction(self,q,r):
        tol=1.e-5
        nZones=6
        topoPop=np.zeros(nZones)
        Q=q.flatten()
        R=r.flatten()
        lenQ=len(Q)
        if lenQ!= len(R):
            print('length of Q and R are not same!!')
            return topoPop
        
        zone0=len(Q[(Q>tol)&(R<-tol)])
        zone5=len(Q[(Q>tol)&(R>tol)])
        
        Q_zone34=Q[(Q<-tol)&(R>tol)]; R_zone34=R[(Q<-tol)&(R>tol)]
        Q_zone12=Q[(Q<-tol)&(R<-tol)]; R_zone12=R[(Q<-tol)&(R<-tol)]
        
        zone1=len(Q_zone12[(-(2./27.)*((-3.*Q_zone12)**1.5)-R_zone12)>tol])
        zone2=len(Q_zone12[(-(2./27.)*((-3.*Q_zone12)**1.5)-R_zone12)<-tol])
        
        zone3=len(Q_zone34[((2./27.)*((-3.*Q_zone34)**1.5)-R_zone34)>tol])
        zone4=len(Q_zone34[((2./27.)*((-3.*Q_zone34)**1.5)-R_zone34)<-tol])
        
        topoPop[0]=zone0
        topoPop[1]=zone1
        topoPop[2]=zone2
        topoPop[3]=zone3
        topoPop[4]=zone4
        topoPop[5]=zone5
                   
        topoFraction=np.zeros(nZones,dtype='float32')
        locsum=np.float32(topoPop)
        comm = MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[topoFraction,MPI.REAL],op=MPI.SUM)
        topoFraction=topoFraction/(lenQ*self.nproc)
        return topoFraction
    
    # JUST FOR FUN!!
    # see http://www.astroml.org/book_figures/chapter3/fig_conditional_probability.html
    def banana_distribution(self,N):
        # create a truncated normal distribution
        theta = np.random.normal(0, np.pi / 8, 10000)
        theta[theta >= np.pi / 4] /= 2
        theta[theta <= -np.pi / 4] /= 2
        # define the curve parametrically
        r = np.sqrt(1. / abs(np.cos(theta) ** 2 - np.sin(theta) ** 2))
        r += np.random.normal(0, 0.08, size=10000)
        x = r * np.cos(theta + np.pi / 4)
        y = r * np.sin(theta + np.pi / 4)
        return x, y