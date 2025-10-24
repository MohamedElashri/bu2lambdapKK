import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import mplhep as hep

# Set LHCb style globally
plt.style.use(hep.style.LHCb2)

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0,0))

def plot_data(ax, data: list, label, histstyle, weights=None, color="black", errorbar=True, mkstyle="o"):
    """
    histstyle: ranges, bins, density
    """
    # Extract only the parameters that numpy.histogram() accepts
    hist_params = {}
    for key in ['range', 'bins', 'density']:
        if key in histstyle:
            hist_params[key] = histstyle[key]
    
    data_hist, bins = np.histogram(data, weights=weights, **hist_params)
    data_hist_errors = np.sqrt(np.abs(data_hist)+1)
    
    try:
        density = histstyle.get("density", False)
        if density is True:
            data_hist_errors = np.sqrt(data_hist / len(data) * np.sum(data_hist))
    except:
        pass
    
    bin_center = (bins[1:] + bins[:-1]) / 2
    bin_width = (bins[1:] - bins[:-1]) / 2
    
    # Only include errorbar-compatible parameters in plot_style
    # These are the parameters that errorbar accepts
    errorbar_params = ['alpha', 'solid_capstyle', 'solid_joinstyle', 
                      'dash_capstyle', 'dash_joinstyle', 'visible',
                      'animated', 'zorder', 'fillstyle']
    
    plot_style = {}
    for key in errorbar_params:
        if key in histstyle:
            plot_style[key] = histstyle[key]
    
    # Get linewidth if available
    linewidth = histstyle.get('linewidth', 2 if errorbar else 4)
    
    if errorbar is True:
        ax.errorbar(x=bin_center, y=data_hist, xerr=bin_width, yerr=data_hist_errors,
                  label=label, ecolor=color, mfc=color, color=color,
                  elinewidth=linewidth, linewidth=linewidth,
                  markersize=6, marker=mkstyle, fmt=' ', **plot_style)
    else:
        ax.errorbar(x=bin_center, y=data_hist, xerr=bin_width, fmt=mkstyle,
                  label=label, ecolor=color, mfc=color, color=color,
                  elinewidth=linewidth, linewidth=linewidth,
                  **plot_style)
    
    return ax, data_hist, data_hist_errors

def plot_hist_ratio(datasets: list, weights: list, labels: list, plot_errors: bool, histstyle):
    """
    datasets: [hist1, hist2]
    weights: [hist_weight1, hist_weight2]
    """
    fig = plt.gcf()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[5, 1.15])
    main_ax = fig.add_subplot(grid[0])
    
    _, hist1, hist_err1 = plot_data(main_ax, datasets[0], label=labels[0], histstyle=histstyle, weights=weights[0], color="black", errorbar=plot_errors)
    _, hist2, hist_err2 = plot_data(main_ax, datasets[1], label=labels[1], histstyle=histstyle, weights=weights[1], color="darkgreen", errorbar=plot_errors)
    
    main_ax.legend(fontsize=20)
    bins = histstyle["bins"]
    ranges = histstyle["range"]
    
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    bin_edges = np.linspace(ranges[0], ranges[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ratio = (hist1/hist2).tolist()
    ratio = np.nan_to_num(ratio, nan=1)
    print(ratio)
    ratio[ratio <= 0] = 1
    ratio[ratio > 100] = 1
    
    ratio_error = np.sqrt(hist_err1/hist2**2+hist1**2*hist_err2**2/hist2**4)
    ratio_error = np.nan_to_num(ratio_error, nan=0)
    ratio_error[ratio_error == 0] = 1
    
    plt.errorbar(bin_centers, ratio, yerr=None, color="gray", fmt='o-')
    subplot_ax.axhline(y=1, color='red', linestyle='--', linewidth=1, label=None)
    subplot_ax.set_yticks(ticks=[-1, 1, 3])
    subplot_ax.set_ylim([-2, 4])
    subplot_ax.set_ylabel(r"$\mathcal{R}$")
    
    return fig, main_ax, subplot_ax