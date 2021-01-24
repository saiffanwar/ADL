from matplotlib import pyplot as plt
import pickle as pck
from scipy import interpolate
import numpy as np
# from matplotlib import rc

plt.style.use('seaborn')
# plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text', usetex=True)

def get_points(filename):
    infile = open(filename,'rb')
    xs = []
    ys = []
    points = pck.load(infile)
    try:
        for x,y in points:
            # print(x,y)
            xs.append(y)
            ys.append(x)
    except:
        for x,y, z in points:
            # print(x,y)
            xs.append(z)
            ys.append(x)
    # return xs,ys
    # print(y[-1])
    x_new = np.linspace(10,1000, 1000)
    intfunc = interpolate.interp1d(xs,ys,fill_value='extrapolate', kind='nearest')
    y_interp = intfunc(x_new)
    infile.close()
    return x_new, y_interp


def plotter():
    USABLE_WIDTH_mm = 180
    USABLE_HEIGHT_mm = 80
    YANK_RATIO = 0.0393701
    USABLE_WIDTH_YANK = USABLE_WIDTH_mm*YANK_RATIO
    USABLE_HEIGHT_YANK = USABLE_HEIGHT_mm*YANK_RATIO
    SUBPLOT_FONT_SIZE = 10
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(USABLE_WIDTH_YANK, USABLE_HEIGHT_YANK), tight_layout=True)
    x,y = get_points('losses.pkl')
    axes.plot(x,y, 'green')
    x,y = get_points('accuracies.pkl')
    axes.plot(x,y, 'red')
    # axes[0].set_ylabel('Training set MSE',fontsize=SUBPLOT_FONT_SIZE)
    axes.set_ylabel(r'MSE',fontsize=SUBPLOT_FONT_SIZE)
    # axes[0].set_xticklabels([])
    axes.set_xlabel('Epochs Elapsed',fontsize=SUBPLOT_FONT_SIZE)
    axes.set_xlim(xmin=10)
    # axes.set_yscale('log')
    # axes[1].set_xlim(xmin=0)
    axes.legend(['Training Set', 'Test Set'], loc='upper right', ncol=1, fancybox=True, shadow=True, fontsize = SUBPLOT_FONT_SIZE)
    fig.savefig('Results.pdf')
    plt.show()

plotter()