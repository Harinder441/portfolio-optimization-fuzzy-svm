import numpy as np
a=np.array([1,0,-1,2,3])
sv = a > 1e-5
print(sv)
ind = np.arange(5)[sv]
print(ind)
