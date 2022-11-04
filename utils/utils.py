import matplotlib.pyplot as plt
import numpy as np

def plot_recon_error(origin, recons):
    '''
    :param origin: array or list
    :param recons: array or list
    :return: plot of the reconstruction error
    '''
    x_axis = 200
    plt.figure(figsize=(10,4))
    origin = np.array(origin)
    recons = np.array(recons)
    y1 = origin.flatten()[:x_axis]
    plt.plot(y1,'b')
    y2 = recons.flatten()[:x_axis]
    plt.plot(y2,'r')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.fill_between(np.arange(x_axis), y1, y2, color='lightcoral')
    plt.legend(labels=['Input', 'Reconstruction', 'Error'])
    plt.show()