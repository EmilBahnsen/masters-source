import matplotlib.pyplot as plt
import os
import numpy as np

dir_fid = './approx_results/model_a_fid'
dir_std = './approx_results/model_a_std'


def plot_data(path, fid_std):
    plt.xlim([0, 200])
    for entry in os.scandir(path):
        p: str = entry.path
        data = np.loadtxt(p, delimiter=',', skiprows=1, usecols=(1,2))
        if fid_std == 1:
            print(np.max(data[:,1]))
        else:
            print(np.min(data[:,1]))
        if p.find('22_27_40') == -1:
            plt.plot(data[:,0], data[:,1], 'k', linewidth=3)
        else:
            plt.plot(data[:,0], data[:,1], '--k', linewidth=3)

ax1 = plt.subplot(2,1,1)
plt.title('QFT Approximation with Simple Diamond Substitution')
plot_data(dir_fid, 1)
plt.ylim([0, 1])
plt.ylabel('Fidelity ($F$)')
print()

ax2 = plt.subplot(2,1,2, sharex=ax1)
plot_data(dir_std, 2)
plt.ylim([0, .2])
plt.ylabel('Fidelity std. ($\sigma_F$)')
plt.xlabel('Iterations')
# make these tick labels invisible
plt.setp(ax1.get_xticklabels(), visible=False)

plt.savefig('model_a_approx.pdf')
plt.show()
