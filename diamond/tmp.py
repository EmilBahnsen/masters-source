from qutip import *
from qutip.testing import *

comm = commutator(controlled_gate(sigmax(),3,0,1), controlled_gate(sigmax(),3,0,2))
print(comm)

run()
