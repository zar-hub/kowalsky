import numpy as np
from statistic_module import *
from overload import *


# Useful functions
def addNumbers(file: str, n: int, start: int = 0):
    f = open(file, "a")
    for i in range(n):
        f.write(str(start+i)+'\n')
    f.close()


# Phisycs module
phifac = {
    'c': -2,
    'm': -3,
    'k': 3
}

phiunits = {
    'm': [1, 0, 0, 0],
    'g': [0, 1, 0, 0],
    's': [0, 0, 1, 0],
    'A': [0, 0, 0, 1],
    'N': [1, 1, -2, 0],
    'J': [1, 2, -2, 0]

}


def phi_unpack(str: str):
    '''
    Unpacks a string in the format:
    fac: fator in phifac
    unit: unit in phiunits
    exp: exponent of unit
    '''
    # Get unit and number
    unit = "".join(i for i in str if i.isalpha())
    exp = str.replace(unit, '')
    if exp == '':
        exp = 1
    else:
        exp = float(exp)
        if exp - round(exp) == 0:  # check if it should be int or float
            exp = int(exp)
    fac = ''

    if unit not in phiunits.keys():
        # Try to find a factor
        for i in range(len(unit)):
            if unit[:i] in phifac.keys():
                fac, unit = unit[:i], unit[i:]

    if unit not in phiunits.keys():
        # Failed to find a factor
        print(f'{str} is not a valid type')
        raise TypeError

    return fac+unit, exp


def phi_parse(str: str):
    list = str.strip().split(' ')
    unit = [i for item in list for i in phi_unpack(item)]
    unit = {unit[i]: unit[i+1] for i in range(0, len(unit), 2)}
    return unit


class phiVar:
    """
    Phisycal value class\n
    If initialized with a string the items should be separated with a

    Todo:
        let user choose how the error is handled
    """

    def __init__(self, val: float = 0, err: float = 0, u=None):
        if u is None:
            u = [0, 0, 0]
        if type(u) == dict:
            self.unit = u
        elif isinstance(u, list):
            self.unit = {
                'm': u[0],
                'kg': u[1],
                's': u[2]
            }
        elif isinstance(u, str):
            self.unit = phi_parse(u)

        self.val = val
        self.err = err

    def __repr__(self) -> str:
        s = ''.join(key+f'{val} ' for key, val in self.unit.items())
        return s

    def like(self, x):
        if self.unit == x.unit:
            return True
        return False

    def __add__(self, x):
        if (not self.like(x)):
            print("can not sum {} and {}. Wrong unit type.".format(self, x))
            return -1
        return phiVar(self.val + x.val, self.unit, self.err + x.err)

    def __sub__(self, x):
        if (not self.like(x)):
            print("can not sum {} and {}. Wrong unit type.".format(self, x))
            return -1
        return phiVar(self.val - x.val, self.unit, self.err + x.err)

    def __mul__(self, x):
        return phiVar(self.val * x.val,
                      (self.unit['m']+x.unit['m'], self.unit['k'] +
                       x.unit['k'], self.unit['s']+x.unit['s']),
                      abs(x.val)*self.err + abs(self.val)*x.err)

    def __truediv__(self, x):
        return phiVar(self.val / x.val,
                      (self.unit['m']-x.unit['m'], self.unit['k'] -
                       x.unit['k'], self.unit['s']-x.unit['s']),
                      self.err/abs(x.val) + abs(self.val)*x.err/(x.val**2))


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
    var = Var(arr, avg)
    sig = np.sqrt(var)
    print("arr:\n", arr,  type(arr),
          "\naverage:\t{:.4f}".format(avg),  type(avg),
          "\nvariance:\t{:.4f}".format(var),  type(var),
          "\nsigma:  \t{:.4f}".format(sig), type(sig))
    if fprint:
        print("\naverage:", avg,
              "\nvariance:", var,
              "\nsigma:", sig,
              file=file)
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
        file = open("sherlockOut.txt", "w")
    i = 0
    pivot = cluster[:, 0]
    while i < pivot.size:
        j = i
        # find next different number
        while (j < pivot.size) and (pivot[j] == pivot[i]):
            j = j+1
        if stat:
            print("\nstart, end, pivot value: ", i, j, pivot[i])
        if fprint:
            print("pivot: ", pivot[i], file=file, end="")
        analysis(cluster[i:j-1, 1], fprint=fprint,
                 file=file)       # give it to analysis
        i = j
    if fprint:
        file.close()
