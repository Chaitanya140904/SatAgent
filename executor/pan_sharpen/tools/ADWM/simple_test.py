import scipy.io as sio
import numpy as np
data = {
    'ms': np.random.rand(64, 64, 8) * 2047,
    'pan': np.random.rand(256, 256) * 2047
}
sio.savemat('test_ms.mat', {'ms': data['ms']})
sio.savemat('test_pan.mat', {'pan': data['pan']})