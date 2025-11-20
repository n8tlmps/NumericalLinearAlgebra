import numpy as np

a = 1
b = 1
c = np.sqrt(a**2 + b**2)

### givens rotation function
def givensrotation(a, b):
    hypot = np.sqrt(a**2 + b**2)
    cos = a / hypot
    sin = -b / hypot
    return cos, sin



if __name__ == "__main__":
    print(cos)
    
