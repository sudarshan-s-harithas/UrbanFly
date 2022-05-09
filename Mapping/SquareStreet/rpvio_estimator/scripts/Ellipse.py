import numpy as np
import matplotlib as plt


K = np.array( [ [ 100.5135 , 25.64  ] , 
				[ 100.254 , 25.5346  ]  ] )


A =  np.linalg.inv( (K@(K.T) ) )

A  = K.T@A 

one = np.ones( ( 1, 2))

A = one@A 

print(A)
