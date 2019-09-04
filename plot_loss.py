from matplotlib import pyplot as plt
import numpy as np
import os

loss_dir = 'loss'
losses = os.listdir(loss_dir)
loss_list = []

for i, loss in enumerate(losses):
    if i%500 == 0:
        loss_list.append(np.load(os.path.join(loss_dir, loss)).item())

fig = plt.figure()
plt.plot(loss_list)
plt.savefig('loss.png')