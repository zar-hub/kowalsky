import string
import numpy as np

mfac = {
    'c': 0.01,
    'm': 0.001,
    'k': 1000
}

# MAIN CLASS
class phival:
    """
    Phisycal value class
    Todo:
        let user choose how the error is handled
    """
    def __init__(self, val=0, unit=None, err=0):
        if unit == None:
            unit=[0,0,0]
        if type(unit) == dict:
            self.unit = unit
        else:
            self.unit = {
                'm': unit[0],
                'k': unit[1],
                's': unit[2]
            }
        self.val = val
        self.err = err

    def parse(self, str):
        i = 0
        # get val
        while i < len(str):         
            if(str[i].isalpha()):
                break
            i = i+1
        self.val = float(str[0:i])
        str = str[i:]
        # get unit 
        # now assuming only a unit is given
        # procede backwords, its easyer... maybe
        self.unit[str[-1]] = 1
        if len(str) > 1:
             self.val = self.val * mfac[str[0:-1]]    
    
    def __repr__(self) -> str:
        return "{}<m[{}]k[{}]s[{}]>err[{}]".format(self.val, self.unit['m'], self.unit['k'], self.unit['s'], self.err) 
    
    def like(self, x):
        if self.unit == x.unit:
            return True
        return False
   
    def __add__(self, x):
        if(not self.like(x)):
            print("can not sum {} and {}. Wrong unit type.".format(self, x))
            return -1
        return phival(self.val + x.val, self.unit, self.err + x.err)
    
    def __mul__(self, x):
        return phival(  self.val * x.val, 
                        (self.unit['m']+x.unit['m'], self.unit['k']+x.unit['k'], self.unit['s']+x.unit['s']),
                        self.err + x.err)
        
# CLASS HELPER FUNCTIONS

# DATA ANALYSIS
def importData(fname, skip=0):
  return np.loadtxt(fname,skiprows=skip)

def E(arr):
    """
    Calculates average 
    Todo:
        add float check
    """
    return np.sum(arr)/np.size(arr)

def V(arr, avg):
    """
    Calculates variance
    assumes EQUALLI PROBABLE VALUES
    Todo:
        improve formula to deal with cancellation error
    """
    return np.sum((arr-avg)**2)/(np.size((arr-avg)**2)-1)

def analysis(arr, fprint=0, file=0):
    """
    Gives:
        avg:    average num
        var:   sigma^2
        sdev:    sigma     
    Todo:
        file hadling   
    """
    avg = E(arr)
    var = V(arr, avg)
    sig = np.sqrt(var)
    print(  "arr:\n",arr,  type(arr),
            "\naverage:\t", avg,  type(avg),
            "\nvariance:\t", var,  type(var),
            "\nsigma:  \t", sig, type(sig)  )
    if fprint:
        print(  "\naverage:", avg,
                "\nvariance:", var,
                "\nsigma:", sig,
                file=file )
    return {"avg": avg, "var": var, "sig": sig}

def cluster(cluster, stat=1, fprint=0):
    """
    Analyze a big table
    Features:
        group data by first colum
    Todo:
        only works for 2D arrays atm
    """
    if fprint:
        file=open("sherlockOut.txt", "w")
    i = 0
    pivot = cluster[:,0]
    while i < pivot.size:
        j=i
        while (j<pivot.size) and (pivot[j] == pivot[i]): # find next different number
            j = j+1
        if stat: 
            print("\nstart, end, pivot value: ", i, j, pivot[i])
        if fprint:
            print("pivot: ", pivot[i], file=file, end="")
        analysis(cluster[i:j-1,1], fprint=fprint, file=file)       # give it to analysis
        i = j
    if fprint:
        file.close()