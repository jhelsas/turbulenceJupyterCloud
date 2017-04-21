# DESCRIPTION OF THE CLASS:

# PROVIDE EXAMPLE:

class Filters:
    # space for importing namespace
    import numpy as np
    from mpi4py import MPI
    
    # Constructor: Default values to variables can also be assigned under constructor
    def __init__(self):
        #self.data=[]
        return
        
    def FilterTheComplexField(self,field,k2,k_cutoff,fltrType):
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
        fltr_width=self.np.pi/k_cutoff
        fltrd_field=field*self.np.exp(-k2*fltr_width*fltr_width/24.0)
        return fltrd_field
            
    def SharpFilter(self,field,k2,k_cutoff):
        k=self.np.sqrt(k2)
        fltrd_field=field.astype(field.dtype)
        fltrd_field[k>k_cutoff]=0.0
        return fltrd_field
    
    def BoxFilter(self,field,k2,k_cutoff):
        fltr_width=self.np.pi/k_cutoff
        k=self.np.sqrt(k2)
        fltrd_field=field*self.np.sin(k*fltr_width/2.0)/(k*fltr_width/2.0)
        return fltrd_field