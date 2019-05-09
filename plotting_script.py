import pickle
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


def read_pickle(url):
    return pickle.load(open(url, 'rb'), encoding='latin1')[0]


# Read the results generated from the script simulation_script
# Example
graph_relu = read_pickle('res_relu_300x200_10sims.pkl')
graph_elu = read_pickle('res_elu_300x200_10sims.pkl')
graph_tanh = read_pickle('res_tanh_300x200_10sims.pkl')



#we define a function that plots confidence intervals
def mean_CI(arr, lower_quantile, upper_quantile):
    lower = []
    upper = []
    mean = []

    for i in range(len(arr[0])):
        lower.append(np.percentile(arr[:,i], lower_quantile))
        upper.append(np.percentile(arr[:,i], upper_quantile))
        mean.append(np.mean(arr[:,i]))
        
    return [np.array(lower), np.array(upper), np.array(mean)]




# Please run the simulations for all of the three activations before doing this step
ranges = [0,100]
graphs_raw = [graph_relu, graph_elu, graph_tanh]
colors = ['r', 'b', 'g']
labels = ['relu', 'elu', 'tanh']

# Getting CI for each graph with lower quantile 5% and upper quantile 95%
graphs = [mean_CI(np.array(graph), 5, 95, ranges) for graph in graphs_raw]


#define a function that plots mean and CI
def plot_mean_and_CI(mean, lower, upper, label, color_mean=None, color_shading=None):
    plt.fill_between(range(len(mean)), upper, lower, color = color_shading, alpha=.5)
    plt.plot(mean, color_mean, linestyle='--', label=label)


# Plotting graphs with CI
plt.style.use('ggplot')
fig = plt.figure(figsize=(10,6))
for i in range(len(graphs)):
    plot_mean_and_CI(graphs[i][2], graphs[i][0], graphs[i][1], labels[i], color_mean=colors[i], color_shading = colors[i])

plt.xlabel('epoch', fontsize=20)
plt.ylabel('Validation Accuracy', fontsize=20)
plt.legend(loc='lower right', fontsize=19)
plt.tick_params(labelsize=13)
plt.rcParams['axes.axisbelow'] = True
leg = plt.legend(loc='lower right', fontsize=19, shadow=True, facecolor='white')