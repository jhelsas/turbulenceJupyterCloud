# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

import numpy as np
from mpi4py import MPI
import math

class FlowStatistics:
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
    
    def FindPDF(self,vx,boundL,boundU,npdf,nproc):
        lenv=len(vx.flatten())
        dpdf=(boundU-boundL)/(npdf-1)
        xkp=np.linspace(boundL,boundU,npdf)

        rcvsum=np.zeros(npdf-1,dtype='float32')

        locsum,edges=np.histogram(vx,bins=xkp)
        locsum=np.float32(locsum)
        comm = MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[rcvsum,MPI.REAL],op=MPI.SUM)
        rcvsum=rcvsum/(dpdf*nproc*lenv)

        pdfx=np.zeros((2,npdf-1),dtype='float32')
        pdfx[0,:]=xkp[0:npdf-1]+0.5*dpdf
        pdfx[1,:]=rcvsum[:]

        return pdfx
    
    def FindJointPDF(self,x,y,lowerLim_x,upperLim_x,lowerLim_y,upperLim_y,bins_x,bins_y,nproc):
        
        if len(np.shape(x))>1:
            print('Error!! The variables should not be multidimensional')
        if len(np.shape(y))>1:
            print('Error!! The variables should not be multidimensional')
        
        lenx=len(x)
        
        xedges=np.linspace(lowerLim_x,upperLim_x,bins_x); dx=xedges[1]-xedges[0]
        yedges=np.linspace(lowerLim_y,upperLim_y,bins_y); dy=yedges[1]-yedges[0]
        locsum,xedges,yedges=np.histogram2d(y,x,bins=(xedges, yedges))
        
        locsum=np.float32(locsum)
        jPDF=np.zeros_like(locsum)        
        
        comm = MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[jPDF,MPI.REAL],op=MPI.SUM)
        jPDF=jPDF/(dx*dy*nproc*lenx)
        
        # How to plot jPDF:
        # import matplotlib.pyplot as plt
        # X,Y=np.meshgrid(xedges,yedges)
        # plt.pcolormesh(X,Y,jPDF)
        return jPDF,xedges,yedges
    
    # http://www.math.uah.edu/stat/expect/Skew.html
    def FindSkewness(self,field,nproc):
        fld=field.flatten()
        fldlen=len(fld)
        
        fldAvg=np.zeros(1,dtype='float32')        
        
        locsum=np.float32(np.sum(fld))
        comm=MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[fldAvg,MPI.REAL],op=MPI.SUM)
        fldAvg=comm.bcast(fldAvg,root=0)
        fldAvg=fldAvg/(nproc*fldlen)
        
        skns1=np.zeros(1,dtype='float32')
        skns2=np.zeros(1,dtype='float32')
        
        locsum_skns1=np.float32(np.sum(((fld-fldAvg)**3)))
        locsum_skns2=np.float32(np.sum(((fld-fldAvg)**2)))
                
        comm.Reduce([locsum_skns1,MPI.REAL],[skns1,MPI.REAL],op=MPI.SUM)
        comm.Reduce([locsum_skns2,MPI.REAL],[skns2,MPI.REAL],op=MPI.SUM)
        
        skns1=comm.bcast(skns1,root=0)
        skns2=comm.bcast(skns2,root=0)
        
        skns=(skns1/(nproc*fldlen))/((skns2/(nproc*fldlen))**1.5)
        
        return skns
    
    # http://www.math.uah.edu/stat/expect/Skew.html
    # NOTE: Please donot use it as it has not been validated yet!!
    def FindKurtosis(self,field,nproc):
        fld=field.flatten()
        fldlen=len(fld)
        
        fldAvg=np.zeros(1,dtype='float32')        
        
        locsum=np.float32(np.sum(fld))
        comm=MPI.COMM_WORLD
        comm.Reduce([locsum,MPI.REAL],[fldAvg,MPI.REAL],op=MPI.SUM)
        fldAvg=comm.bcast(fldAvg,root=0)
        fldAvg=fldAvg/(nproc*fldlen)
        
        kurt1=np.zeros(1,dtype='float32')
        kurt2=np.zeros(1,dtype='float32')
        
        locsum_kurt1=np.float32(np.sum(((fld-fldAvg)**4.)))
        locsum_kurt2=np.float32(np.sum(((fld-fldAvg)**2.)))
                
        comm.Reduce([locsum_kurt1,MPI.REAL],[kurt1,MPI.REAL],op=MPI.SUM)
        comm.Reduce([locsum_kurt2,MPI.REAL],[kurt2,MPI.REAL],op=MPI.SUM)
        
        kurt1=comm.bcast(kurt1,root=0)
        kurt2=comm.bcast(kurt2,root=0)
        
        kurt=(kurt1/(nproc*fldlen))/((kurt2/(nproc*fldlen))**2.)
        
        return kurt