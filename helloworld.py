import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

data = np.random.random((2, 100))
plt.plot(data)
plt.show()

print(datetime.now())