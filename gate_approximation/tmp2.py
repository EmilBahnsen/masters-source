from tf_qc import *
import txtutils
tf.random.set_seed(0)
m = PrePostQFTUIQFT()
data = random_state_vectors(1, 4)
m.compile(loss=Mean1mFidelity())
m.fit(data, data, epochs=0)
print(*m.variables, sep='\n')
print(txtutils.ndtotext(m.matrix.numpy()))
