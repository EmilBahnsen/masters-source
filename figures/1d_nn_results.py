import matplotlib.pyplot as plt
import os
import pickle

def get_data(path):
    with open(path, 'rb') as f:
        fig: plt.Figure = pickle.load(f)
        ax: plt.Axes = fig.axes[0]
        return ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), ax.lines[1].get_xdata(), ax.lines[1].get_ydata()

f: plt.Figure = plt.figure(figsize=(3*2, 3*2))
st = f.suptitle('Diamond Assisted Neural Network Results', fontsize="x-large")
for i in range(6):
    dx, dy, fx, fy = get_data(f'./1d_nn_results/fig{i+1}.pickle')
    ax = f.add_subplot(3,2,i+1)
    if i < 4:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylabel('$f(x)$')
    else:
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
    plt.plot(dx, dy, '.b')
    plt.plot(fx, fy, '-r')

ax1: plt.Axes = f.axes[0]
ax1.lines[0].set_label('Data')
ax1.lines[1].set_label('Fit')
ax1.legend()
# shift subplots down:
f.tight_layout()
f.subplots_adjust(top=0.92)
plt.savefig('1d_nn_result.pdf')