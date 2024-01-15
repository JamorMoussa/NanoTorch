import neuranet as nnt
import neuranet.nn as nn 


l = nn.Linear(10, 3)
l2 = nn.Linear(3, 2)

print(l(l2((nnt.rand(4, 2)))).shape)

"""
Linear(
    Paramerts:
               [[0.70942343 0.77254096 0.12916525 0.09169435 0.21316721]
               [0.09420515 0.29974063 0.1239921  0.5242108  0.1645537 ]
               [0.60643729 0.99028322 0.31028219 0.46085306 0.12154007]]
    grad: 

    )

"""