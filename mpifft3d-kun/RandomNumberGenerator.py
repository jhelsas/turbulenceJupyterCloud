# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class RandomNumberGenerator:
    # space for importing namespace
    import numpy as np
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
        
    def GetRandNumber_real(self,lowerLimit,upperLimit,nx,ny,nz):
        randNumber=lowerLimit+(upperLimit-lowerLimit)*self.np.random.random((nx,ny,nz))
        return randNumber
    
    def GetRandNumber_complex(self,lowerLimit,upperLimit,nx,ny,nz):
        randNumber=self.GetRandNumber_real(lowerLimit,upperLimit,nx,ny,nz) \
        +1j*self.GetRandNumber_real(lowerLimit,upperLimit,nx,ny,nz)
        return randNumber