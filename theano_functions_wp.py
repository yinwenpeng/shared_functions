import numpy as np
import theano
import theano.tensor as T

def adadelta(params, gparams, learning_rate = 1.0, rho = 0.95, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        delta_acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        update = (g * T.sqrt(delta_acc + epsilon) / T.sqrt(acc_new + epsilon))
        updates.append((p, p - learning_rate * update))

        delta_acc_new = rho * delta_acc + (1 - rho) * update ** 2
        updates.append((delta_acc, delta_acc_new))
    return updates


