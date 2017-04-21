# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class Statistics:
    # space for importing namespace
    import numpy as np
    from mpi4py import MPI
    import math
    
    # use this space to define variables at class level
    
    ####################################
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
        
    # Use the below space for method definitions
    
    def FindPDF(self,vx,bound1,bound2,npdf,lx,ly,lz,nproc):
        dpdf=(bound2-bound1)/(npdf-1)
        xkp=self.np.linspace(bound1,bound2,npdf)

        rcvsum=self.np.zeros(npdf-1,dtype='float32')

        locsum,edges=self.np.histogram(vx,bins=xkp)
        locsum=self.np.float32(locsum)
        comm = self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[rcvsum,self.MPI.REAL],op=self.MPI.SUM)
        rcvsum=rcvsum/(dpdf*nproc*lx*ly*lz)

        pdfx=self.np.zeros((2,npdf-1),dtype='float32')
        pdfx[0,:]=xkp[0:npdf-1]+0.5*dpdf
        pdfx[1,:]=rcvsum[:]

        return pdfx
    
    def FindPDF_joint(self,x,y,lowerLim_x,upperLim_x,lowerLim_y,upperLim_y,bins_x,bins_y,nproc):
        
        if len(self.np.shape(x))>1:
            print('Error!! The variables should not be multidimensional')
        if len(self.np.shape(y))>1:
            print('Error!! The variables should not be multidimensional')
        
        lenx=len(x)
        
        xedges=self.np.linspace(lowerLim_x,upperLim_x,bins_x); dx=xedges[1]-xedges[0]
        yedges=self.np.linspace(lowerLim_y,upperLim_y,bins_y); dy=yedges[1]-yedges[0]
        locsum,xedges,yedges=self.np.histogram2d(y,x,bins=(xedges, yedges))
        
        locsum=self.np.float32(locsum)
        jPDF=self.np.zeros_like(locsum)        
        
        comm = self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[jPDF,self.MPI.REAL],op=self.MPI.SUM)
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
        
        fldAvg=self.np.zeros(1,dtype='float32')        
        
        locsum=self.np.float32(self.np.sum(fld))
        comm=self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[fldAvg,self.MPI.REAL],op=self.MPI.SUM)
        fldAvg=comm.bcast(fldAvg,root=0)
        fldAvg=fldAvg/(nproc*fldlen)
        
        skns1=self.np.zeros(1,dtype='float32')
        skns2=self.np.zeros(1,dtype='float32')
        
        locsum_skns1=self.np.float32(self.np.sum(((fld-fldAvg)**3)))
        locsum_skns2=self.np.float32(self.np.sum(((fld-fldAvg)**2)))
                
        comm.Reduce([locsum_skns1,self.MPI.REAL],[skns1,self.MPI.REAL],op=self.MPI.SUM)
        comm.Reduce([locsum_skns2,self.MPI.REAL],[skns2,self.MPI.REAL],op=self.MPI.SUM)
        
        skns1=comm.bcast(skns1,root=0)
        skns2=comm.bcast(skns2,root=0)
        
        skns=(skns1/(nproc*fldlen))/((skns2/(nproc*fldlen))**1.5)
        
        return skns
    
    # http://www.math.uah.edu/stat/expect/Skew.html
    def FindKurtosis(self,field,nproc):
        fld=field.flatten()
        fldlen=len(fld)
        
        fldAvg=self.np.zeros(1,dtype='float32')        
        
        locsum=self.np.float32(self.np.sum(fld))
        comm=self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[fldAvg,self.MPI.REAL],op=self.MPI.SUM)
        fldAvg=comm.bcast(fldAvg,root=0)
        fldAvg=fldAvg/(nproc*fldlen)
        
        kurt1=self.np.zeros(1,dtype='float32')
        kurt2=self.np.zeros(1,dtype='float32')
        
        locsum_kurt1=self.np.float32(self.np.sum(((fld-fldAvg)**4.)))
        locsum_kurt2=self.np.float32(self.np.sum(((fld-fldAvg)**2.)))
                
        comm.Reduce([locsum_kurt1,self.MPI.REAL],[kurt1,self.MPI.REAL],op=self.MPI.SUM)
        comm.Reduce([locsum_kurt2,self.MPI.REAL],[kurt2,self.MPI.REAL],op=self.MPI.SUM)
        
        kurt1=comm.bcast(kurt1,root=0)
        kurt2=comm.bcast(kurt2,root=0)
        
        kurt=(kurt1/(nproc*fldlen))/((kurt2/(nproc*fldlen))**2.)
        
        return kurt
    
    def FindSkewness_OBSOLETE(self,field,nproc):
        fld=field.flatten()
        fldlen=len(fld)
        
        fldAvg=self.np.zeros(1,dtype='float32')
        fldAvgALL=self.np.zeros(1*nproc,dtype='float32')
        
        locsum=self.np.float32(self.np.sum(fld))
        comm=self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[fldAvg,self.MPI.REAL],op=self.MPI.SUM)
        comm.Allgather([fldAvg,self.MPI.REAL],[fldAvgALL,self.MPI.REAL])

        fldAvgALL=fldAvgALL/(nproc*fldlen)
        
        skns1=(fld-fldAvgALL[0])**3
        skns2=(fld-fldAvgALL[0])**2
        
        rcvsum_skns1=self.np.zeros(1,dtype='float32')
        locsum_skns1=self.np.float32(self.np.sum(skns1))
        rcvsum_skns2=self.np.zeros(1,dtype='float32')
        locsum_skns2=self.np.float32(self.np.sum(skns2))
                
        comm.Reduce([locsum_skns1,self.MPI.REAL],[rcvsum_skns1,self.MPI.REAL],op=self.MPI.SUM)
        comm.Reduce([locsum_skns2,self.MPI.REAL],[rcvsum_skns2,self.MPI.REAL],op=self.MPI.SUM)
        
        skns1_ALL=self.np.zeros(1*nproc,dtype='float32')
        skns2_ALL=self.np.zeros(1*nproc,dtype='float32')
        comm.Allgather([rcvsum_skns1,self.MPI.REAL],[skns1_ALL,self.MPI.REAL])
        comm.Allgather([rcvsum_skns2,self.MPI.REAL],[skns2_ALL,self.MPI.REAL])
        
        skns=(skns1_ALL[0]/(nproc*fldlen))/((skns2_ALL[0]/(nproc*fldlen))**1.5)
        
        return skns
    
    def FindPDF_joint_OBSOLETE(self,r,q,lowerLim_r,upperLim_r,lowerLim_q,upperLim_q,bins_r,bins_q,nproc):
        
        if len(self.np.shape(q))>1:
            print('Error!! The variables should not be multidimensional')
        if len(self.np.shape(r))>1:
            print('Error!! The variables should not be multidimensional')
        
        lenq=len(q)
                
        locsum,edges_r,edges_q=self.np.histogram2d(r,q,bins=[bins_r,bins_q],\
                                                   range=[[lowerLim_r,upperLim_r],[lowerLim_q,upperLim_q]])
        
        dq=(upperLim_q-lowerLim_q)/(bins_q)
        dr=(upperLim_r-lowerLim_r)/(bins_r)
        
        jPDF=self.np.zeros((bins_q,bins_r),dtype='float32')
        
        locsum=self.np.float32(locsum)
        comm = self.MPI.COMM_WORLD
        comm.Reduce([locsum,self.MPI.REAL],[jPDF,self.MPI.REAL],op=self.MPI.SUM)
        jPDF=jPDF/(dq*dr*nproc*lenq)
        
        #Q,R=self.np.meshgrid(edges_q,edges_r)
        
        #fig = plt.figure(figsize=(7, 3))
        #ax = fig.add_subplot(132)
        #ax.set_title('pcolormesh: exact bin edges')
        #ax.pcolormesh(X,Y,np.transpose(data))
        #ax.set_aspect('equal')
        
        return jPDF,edges_r,edges_q
    