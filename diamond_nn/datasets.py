import pickle

# import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import glob, importlib, os, pathlib, sys
from mlxtend.plotting import plot_decision_regions

π = math.pi

data_size = 50
data_range = np.linspace(0., 1., data_size)

def circle_in_plain():
    plain_x = np.array([[[x0, x1] for x1 in data_range] for x0 in data_range])
    plain_x = np.reshape(plain_x, (data_size**2, 2))
    idx_inside = (plain_x[..., 0] - 0.5)**2 + (plain_x[..., 1] - 0.5)**2 < 0.5
    # outside the circle -1 and inside it's +1
    plain_labels = -np.ones((data_size**2,))
    plain_labels[idx_inside] = 1

    return shuffle_data(plain_x, plain_labels)


def two_circles_in_plain():
    plain_x = np.array([[[x0, x1] for x1 in data_range] for x0 in data_range])
    plain_x = np.reshape(plain_x, (data_size**2, 2))
    idx_inside1 = (plain_x[..., 0] - .25)**2 + (plain_x[..., 1] - .25)**2 < .25
    idx_inside2 = (plain_x[..., 0] - .75)**2 + (plain_x[..., 1] - .75)**2 < .25
    idx_inside = np.logical_or(idx_inside1, idx_inside2)
    # outside the circle -1 and inside it's +1
    plain_labels = -np.ones((data_size**2,))
    plain_labels[idx_inside] = 1

    return shuffle_data(plain_x, plain_labels)


def checkerboard_in_plain(minus_plus=True):
    data_size = 47
    data_range = np.linspace(0., 1., data_size)
    plain_x = np.array([[[x0, x1] for x1 in data_range] for x0 in data_range])
    plain_x = np.reshape(plain_x, (data_size ** 2, 2))
    labels = np.logical_xor( (plain_x[:,0]//(1/5))%2==0, (plain_x[:,1]//(1/5))%2==0 )
    labels = np.array(labels, int)
    if minus_plus:
        labels[labels == 0] = -1
    return shuffle_data(plain_x, labels)


def article_2003_09887_data(index):
    np.random.seed(0)
    def make_points(n_points):
        return np.random.uniform(0, 1, size=[n_points, 2])
    def make_points_strip(n_points):
        return np.append(
            np.random.uniform(.25, .65, size=[n_points, 1]),
            np.random.uniform(0, 1, size=[n_points, 1]),
            axis=-1
        )
    idx0 = None
    idx1 = None
    def make_elipsis_idx(xy, x0, y0, w, h):
        return (xy[:, 0] - x0) ** 2 / w ** 2 + (xy[:, 1] - y0) ** 2 / h ** 2 < 1
    if index == 0:
        n_points = 2000
        xy = make_points(n_points)
        idx0 = make_elipsis_idx(xy, .33, .5, .25, .4)
        idx1 = make_elipsis_idx(xy, .66, .5, .25, .4)
        idx_and_half = np.logical_and(idx0, idx1)
        idx_and_half[len(idx_and_half)//2:] = 0
        idx1[idx_and_half] = 0
    elif index == 1:
        n_points1 = 2000
        n_points2 = 500
        xy = np.append(make_points(n_points1), make_points_strip(n_points2), axis=0)
        n_points = n_points1 + n_points2
        h = .15
        w = .3
        idx0 =                     make_elipsis_idx(xy, .33, .8, w, h)
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .33, .2, w, h))
        idx1 =                     make_elipsis_idx(xy, .66, .5, w, h)
    elif index == 2:
        n_points1 = 800
        n_points2 = 1100
        xy = np.append(make_points(n_points1), make_points_strip(n_points2), axis=0)
        n_points = n_points1 + n_points2
        h = .13
        w = .3
        idx0 =                     make_elipsis_idx(xy, .325, .625, w, h)
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .325, .150, w, h))
        idx1 =                     make_elipsis_idx(xy, .675, .850, w, h)
        idx1 = np.logical_or(idx1, make_elipsis_idx(xy, .675, .375, w, h))
    elif index == 3:
        n_points = 2700
        xy = make_points(n_points)
        h = .18
        w = .18
        idx0 =                     make_elipsis_idx(xy, .3, .7, w, h)
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .7, .3, w, h))
        idx1 =                     make_elipsis_idx(xy, .3, .3, w, h)
        idx1 = np.logical_or(idx1, make_elipsis_idx(xy, .7, .7, w, h))
    elif index == 4:
        n_points = 2800
        xy = make_points(n_points)
        h = .12
        w = .12
        idx0 = np.zeros(n_points)
        idx1 = np.zeros(n_points)
        flag = False
        for x in [.2, .5, .8]:
            for y in [.2, .5, .8]:
                if flag:
                    idx0 = np.logical_or(idx0, make_elipsis_idx(xy, x, y, w, h))
                else:
                    idx1 = np.logical_or(idx1, make_elipsis_idx(xy, x, y, w, h))
                flag = ~flag
    elif index == 5:
        n_points = 2200
        xy = make_points(n_points)
        h = .1
        w = .1
        idx0 = np.zeros(n_points)
        idx1 = np.zeros(n_points)
        flag = False
        for x in np.linspace(1/8, 7/8, 4):
            for y in np.linspace(1/8, 7/8, 4):
                if flag:
                    idx0 = np.logical_or(idx0, make_elipsis_idx(xy, x, y, w, h))
                else:
                    idx1 = np.logical_or(idx1, make_elipsis_idx(xy, x, y, w, h))
                flag = ~flag
            flag = ~flag
    elif index == 6:
        n_points = 1200
        xy = make_points(n_points)
        r = .2
        idx0 = make_elipsis_idx(xy, .5, .5, r, r)
        idx1 = ~idx0
    elif index == 7:
        n_points = 1200
        xy = make_points(n_points)
        r = .15
        idx0 =                     make_elipsis_idx(xy, .7, .7, r, r)
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .3, .3, r, r))
        idx1 = ~idx0
    elif index == 8:
        n_points = 1200
        xy = make_points(n_points)
        r = .15
        idx0 =                     make_elipsis_idx(xy, .3, .3, r, r)
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .3, .7, r, r))
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .7, .3, r, r))
        idx0 = np.logical_or(idx0, make_elipsis_idx(xy, .7, .7, r, r))
        idx1 = ~idx0

    idx_all = np.logical_or(idx0, idx1)
    labels = np.zeros(n_points)
    labels[idx1] = 1

    return xy[idx_all], labels[idx_all]


def plot_article_result_from_files(python_files, weights_files):
    for i, (file_path, weight_path) in enumerate(zip(python_files, weights_files)):
        print('plotting', file_path)
        ax = plt.subplot(3, 3, i + 1)
        plt.minorticks_on()
        if i == 0 or i == 3 or i == 6:
            plt.ylabel('$x_2$')
        if i == 6 or i == 7 or i == 8:
            plt.xlabel('$x_1$')
        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        if i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8:
            plt.setp(ax.get_yticklabels(), visible=False)
        # import model
        path, file = os.path.split(file_path)
        sys.path.append(path)
        module_name = pathlib.Path(file).stem
        module = importlib.import_module(module_name)
        model = module.U3_U()
        # Load data
        plain_x, plain_labels = article_2003_09887_data(i)
        plain_x = normalize_data(plain_x)
        model(plain_x[:2])
        # load weights
        with open(weight_path, 'br') as w_file:
            weights = pickle.load(w_file)
            model.set_weights(weights)
        plot_decision_regions(X=plain_x, y=np.array(plain_labels, int), clf=model, ax=ax, scatter_kwargs={'alpha': 0.5})
        sys.path.remove(path)  # Rm the path of the file again


def simple_1D_functions(index):
    np.random.seed(0)
    x = np.random.uniform(-1, 1, 100)
    y = None
    if index == 0:
        y = x**2
    elif index == 1:
        y = np.exp(x)
    elif index == 2:
        y = np.sin(π*x)
    elif index == 3:
        y = np.abs(x)
    return x, y


def complex_1D_functions(index):
    np.random.seed(0)
    x = np.random.uniform(-1, 1, 100)
    if index == 0:
        y = np.sin(2*π*x)
    elif index == 1:
        y = np.sin(π*x**2)
    elif index == 2:
        y = np.sin(π * x) * np.cos(π / 2 * x)
    elif index == 3:
        y = np.sin(2*π*x)*np.exp(x)
    elif index == 4:
        y = np.sin(2*π*x)*np.sin(4*π*x)*np.exp(x)
    return x, y


def _plot(plain_x, plain_labels):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(plain_x[:, 0], plain_x[:, 1], c=plain_labels, cmap=plt.get_cmap('bwr'), s=.5)


def normalize_data(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def shuffle_data(*args):
    shuffle_idx = np.array(range(args[0].shape[0]))
    np.random.shuffle(shuffle_idx)
    out_arrays = []
    for arg in args:
        out_arrays.append(arg[shuffle_idx])
    return out_arrays


if __name__ == '__main__':
    def plot_article_data():
        f = plt.figure(figsize=(5, 5))
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.minorticks_on()
            plt.xticks([0, .5, 1], ['0.0', '0.5', '1.0'] if i == 6 or i == 7 or i == 8 else [])
            plt.yticks([0, .5, 1], ['0.0', '0.5', '1.0'] if i == 0 or i == 3 or i == 6 else [])
            plain_x, plain_labels = article_2003_09887_data(i)
            print(f'{len(plain_x)} points in #{i}')
            _plot(plain_x, plain_labels)
        plt.show()

    def plot_article_data_fit(base_path, rest_paths, title) -> plt.Figure:
        plt.rc('figure', titlesize=20)
        plt.rc('axes', labelsize=16)
        fig = plt.figure(figsize=(10, 10))
        fig.tight_layout()
        st = fig.suptitle(title)#, fontsize="x-large")
        py_files = [os.path.join(base_path, rest_path, 'fit_qc_article.py') for rest_path in rest_paths]
        w_files = [os.path.join(base_path, rest_path, 'weights.pickle') for rest_path in rest_paths]
        plot_article_result_from_files(py_files, w_files)
        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.9)
        return fig

    def plot_article_result_ANN():
        fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/', [
            'no_qc_relu_data0/2020-04-21T15:20:50.893499',
            'no_qc_relu_data1/2020-04-21T15:21:04.681599',
            'no_qc_relu_data2/2020-04-21T15:21:18.986233',
            'no_qc_relu_data3/2020-04-21T15:19:29.176211',
            'no_qc_relu_data4/2020-04-21T15:19:42.408143',
            'no_qc_relu_data5/2020-04-21T15:19:56.018477',
            'no_qc_relu_data6/2020-04-21T15:22:12.826665',
            'no_qc_relu_data7/2020-04-21T15:22:25.773578',
            'no_qc_relu_data8/2020-04-21T15:20:35.557015'
        ], 'ANN')
        fig.savefig('x4_result/ANN.pdf')

    def plot_article_result_encoding():
        fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/', [
            'encoding_data0/2020-04-21T16:26:19.284485',
            'encoding_data1/2020-04-21T16:13:08.954627',
            'encoding_data2/2020-04-21T16:34:58.566531',
            'encoding_data3/2020-04-21T16:35:45.971069',
            'encoding_data4/2020-04-21T16:15:31.916835',
            'encoding_data5/2020-04-21T16:02:11.122255',
            'encoding_data6/2020-04-21T16:23:55.183942',
            'encoding_data7/2020-04-21T16:31:47.862786',
            'encoding_data8/2020-04-21T16:25:31.480935'
        ], 'encoding only')
        fig.savefig('x4_result/encoding_only.pdf')

    def plot_article_result_encoding_U3():
        fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/', [
            'encoding_U3_data0/2020-04-21T00:47:47.899659',
            'encoding_U3_data1/2020-04-21T00:48:35.255852',
            'encoding_U3_data2/2020-04-21T00:06:11.361664',
            'encoding_U3_data3/2020-04-21T00:08:24.129193',
            'encoding_U3_data4/2020-04-21T00:50:59.834629',
            'encoding_U3_data5/2020-04-21T00:44:30.531915',
            'encoding_U3_data6/2020-04-21T00:45:17.944469',
            'encoding_U3_data7/2020-04-21T00:53:26.180366',
            'encoding_U3_data8/2020-04-21T00:46:58.328750'
        ], 'encoding -> U3')
        fig.savefig('x4_result/encoding_U3.pdf')

    def plot_article_result_encoding_U3_U():
        fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/', [
            'encoding_U3_U_data0/2020-04-21T15:12:03.366958',
            'encoding_U3_U_data1/2020-04-21T14:54:21.278735',
            'encoding_U3_U_data2/2020-04-21T15:13:50.831770',
            'encoding_U3_U_data3/2020-04-21T14:59:00.099988',
            'encoding_U3_U_data4/2020-04-21T15:15:35.195738',
            'encoding_U3_U_data5/2020-04-22T22:26:01.473103',
            'encoding_U3_U_data6/2020-04-21T15:09:23.374249',
            'encoding_U3_U_data7/2020-04-21T15:10:17.180348',
            'encoding_U3_U_data8/2020-04-22T22:37:56.536299'
        ], 'Diamond Assisted Binary Classification')
        fig.savefig('x4_result/encoding_U3_U.pdf')

    # --- FOR MY ARTICLE ---
    _ = [
        'encoding_UXY_U_measure0_data0/2020-10-05T21:51:08.873049',
        'encoding_UXY_U_measure0_data1/2020-10-05T15:54:23.293032',
    ]
    result_layer1 = [
        'encoding_U3_U_measure0_data0/',
        'encoding_U3_U_measure0_data1/',
        'encoding_U3_U_measure0_data2/',
        'encoding_U3_U_measure0_data3/',
        'encoding_U3_U_measure0_data4/',
        'encoding_U3_U_measure0_data5/',
        'encoding_U3_U_measure0_data6/',
        'encoding_U3_U_measure0_data7/',
        'encoding_U3_U_measure0_data8/'
    ]
    result_layer2 = [
        'encoding_U3_U_U3_U_measure0_data0/',
        'encoding_U3_U_U3_U_measure0_data1/',
        'encoding_U3_U_U3_U_measure0_data2/',
        'encoding_U3_U_U3_U_measure0_data3/',
        'encoding_U3_U_U3_U_measure0_data4/',
        'encoding_U3_U_U3_U_measure0_data5/',
        'encoding_U3_U_U3_U_measure0_data6/',
        'encoding_U3_U_U3_U_measure0_data7/',
        'encoding_U3_U_U3_U_measure0_data8/'
    ]
    result_layer3 = [
        'encoding_U3_U_U3_U_U3_U_measure0_data0/',
        'encoding_U3_U_U3_U_U3_U_measure0_data1/',
        'encoding_U3_U_U3_U_U3_U_measure0_data2/',
        'encoding_U3_U_U3_U_U3_U_measure0_data3/',
        'encoding_U3_U_U3_U_U3_U_measure0_data4/',
        'encoding_U3_U_U3_U_U3_U_measure0_data5/',
        'encoding_U3_U_U3_U_U3_U_measure0_data6/',
        'encoding_U3_U_U3_U_U3_U_measure0_data7/',
        'encoding_U3_U_U3_U_U3_U_measure0_data8/'
    ]

    def get_run_with_max_acc_across_runs(path) -> str:
        max_acc = 0
        max_path = None
        for item in os.listdir(path):
            result_path = os.path.join(path, item)
            if os.path.isdir(result_path):
                hist_file = os.path.join(result_path, 'history.pickle')
                try:
                    with open(hist_file, 'rb') as f:
                        hist = pickle.load(f)
                        this_max = max(hist['accuracy'])
                        if this_max > max_acc:
                            max_acc = this_max
                            max_path = result_path
                except:
                    continue
        print(max_path)
        return max_path

    def plot_results_1_layer():
        fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/',
                                    result_layer1, 'Fit using 1 diamond layer')
        fig.savefig('x4_result/one_layer.pdf')

    def plot_acc_across_layers(base_path, rest_paths_s, title):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=15)
        fig = plt.figure()
        plt.xlim([.9, 4])
        plt.ylim([.65, 1.01])
        plt.xticks([1,2,3])
        plt.yticks([.7,.8,.9,1])
        plt.xlabel('Layers', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title)
        max_acc = np.zeros((3,9))
        for n_layer, rest_paths in enumerate(rest_paths_s):
            hist_files = [os.path.join(
                get_run_with_max_acc_across_runs(os.path.join(base_path, rest_path)),
                'history.pickle') for rest_path in rest_paths]
            for n_dataset, hist_file in enumerate(hist_files):
                with open(hist_file, 'rb') as f:
                    hist = pickle.load(f)
                    max_acc[n_layer, n_dataset] = max(hist['accuracy'])
            # Make the collage of results
            # result_end_paths = [os.sep.join(hist_file.rsplit(r"/")[-3:-1]) for hist_file in hist_files]
            # fig = plot_article_data_fit('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/',
            #                             result_end_paths,
            #                             'Fit using ' + str(n_layer) + ' diamond layer' + ('s' if n_layer > 1 else ''))
            # fig.savefig('x4_result/layer' + str(n_layer) + '.pdf')
        x_layers = [1,2,3]
        plt.plot(x_layers, max_acc[:,0], label='1', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,1], label='2', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,2], label='3', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,3], label='4', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,4], label='5', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,5], label='6', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,6], label='7', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,7], label='8', markersize=5, marker='o')
        plt.plot(x_layers, max_acc[:,8], label='9', markersize=5, marker='o')
        plt.legend()
        fig.savefig('acc_layers_plot.pdf')

    # print(get_run_with_max_acc_across_runs('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article/encoding_UXY_U_measure0_data0'))
    plot_acc_across_layers('/home/emil/pycharm_project/diamond_nn/x4/fit_qc_article',
                           [result_layer1, result_layer2, result_layer3], 'Accuracy accross datasets')

    # plt.figure()
    # plt.scatter(plain_x[:, 0]**2 + plain_x[:, 1]**2, plain_x[:, 1], c=plain_labels, cmap=plt.get_cmap('bwr'))
    # plt.show()
