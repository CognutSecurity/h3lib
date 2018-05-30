"""
This utility helper file is used for visualizing a dataset from .csv file.
The main usage of this file is for easy check how the training dataset looks like,
and to give developers a quick overview of the data they are dealing with.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import numpy as np
import sys
import os.path
import getopt
import importlib
from h3mlcore.utils.PlotHelper import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, TSNE
from bokeh.plotting import figure
from bokeh.io import output_file, show, save
from bokeh.models.tools import BoxZoomTool, PanTool, WheelZoomTool, ResetTool, HoverTool, SaveTool
from bokeh.models.ranges import Range1d
from bokeh.models.sources import ColumnDataSource


class DataViz(object):
    """A class for dataset viewer used to analyse features in data
    set straightforwardly

    Args: Input a config JSON string to configure the style of the plots
    Returns: a DataViz object

    """

    def __init__(self, config):
        '''
        init a dataviz obj for plotting dataset
        :param config: configuration for plotting
        :return: a figure handler of bokeh
        '''

        if config.has_key('dot_size'):
            self.dot_size = config['dot_size']
        else:
            self.dot_size = 4
        if config.has_key('font_size'):
            self.font_size = config['font_size']
        else:
            self.font_size = 12
        if config.has_key('line_width'):
            self.line_width = config['line_width']
        else:
            self.line_width = 2
        if config.has_key('alpha'):
            self.alpha = config['alpha']
        else:
            self.alpha = 0.3
        if config.has_key('colormap'):
            # default 3 colors
            self.colormap = config['colormap']
        else:
            # binary colors
            self.colormap = 'viridis'
        if config.has_key('color'):
            self.color = config['color']
        else:
            self.color = 'navy'
        if config.has_key('width'):
            self.w = config['width']
        else:
            self.w = 480
        if config.has_key('height'):
            self.h = config['height']
        else:
            self.h = 320
        if config.has_key('output_file'):
            self.output_file = config['output_file']
        else:
            self.output_file = "myplots.html"

        self.binary_colors = ['#4285F4', '#EA4335']
        self.default_tools = ["pan", "wheel_zoom", "reset"]

        output_file(self.output_file, title=str.upper(
            ' '.join(self.output_file.split('/')[-1].split('.')[:-1])))

    def _get_figure_instance(self,
                             xlabel="x label name",
                             ylabel="y label name",
                             xlim=None, ylim=None,
                             width=None, height=None,
                             **kwargs):
        """return a figure instance

        Args:
          kwargs: parameters for creating a figure in bokeh
          xlabel:  (Default value = "x label name")
          ylabel:  (Default value = "y label name")
          xlim:  (Default value = None)
          ylim:  (Default value = None)
          width:  (Default value = None)
          height:  (Default value = None)
          **kwargs: 

        Returns:
          a bokeh plot figure obj

        """

        f = figure(plot_width=self.w,
                   plot_height=self.h,
                   output_backend='webgl',
                   toolbar_location='above',
                   active_scroll='wheel_zoom',
                   tools=self.default_tools,
                   **kwargs
                   )
        f.xaxis.axis_label = xlabel
        f.yaxis.axis_label = ylabel

        if xlim is not None and len(xlim) == 2:
            f.x_range = Range1d(xlim[0], xlim[1])
        if ylim is not None and len(ylim) == 2:
            f.y_range = Range1d(ylim[0], ylim[1])

        if width is not None:
            f.plot_width = width
        if height is not None:
            f.plot_height = height

        return f

    def feature_scatter1d(self,
                          X,
                          names=None,
                          title="Feature Distribution",
                          xlabel="Feature IDs",
                          ylabel="Feature Values",
                          xlim=None, ylim=None,
                          width=None, height=None):
        """

        Args:
          X: 
          names:  (Default value = None)
          title:  (Default value = "Feature Distribution")
          xlabel:  (Default value = "Feature IDs")
          ylabel:  (Default value = "Feature Values")
          xlim:  (Default value = None)
          ylim:  (Default value = None)
          width:  (Default value = None)
          height:  (Default value = None)

        Returns:

        """
        n, d = X.shape
        indices = []
        for i in np.arange(d):
            indices.extend((i + 1) * np.ones(n))

        f = self._get_figure_instance(title=title,
                                      xlabel=xlabel,
                                      ylabel=ylabel,
                                      xlim=xlim,
                                      ylim=ylim,
                                      width=width, height=height,
                                      x_range=names)
        f.circle(np.asarray(indices), X.T.flatten(), color='gray',
                 size=self.dot_size, alpha=self.alpha)
        # if names is not None:
        #    f.set(x_range=names)
        f.xaxis.major_label_orientation = - np.pi / 2
        return f

    def project2d(self, X, y=None, method='pca', g=0.5,
                  title="Sample Distribution",
                  xlabel="1st Dim.",
                  ylabel="2nd Dim.",
                  legend=None, legend_orientation='vertical', legend_localtion='top_right',
                  xlim=None, ylim=None,
                  width=None, height=None):

        """Project high-dimensiona data to 2D for visulaization
        using methods e.g., pca, kernel-pca, mds

        Args:
          X: N*d dataset
          y: labels/None if not given (Default value = None)
          method: string in ['pca','kpca','mds'] (Default value = 'pca')
          g:  (Default value = 0.5)
          title:  (Default value = "Sample Distribution")
          xlabel:  (Default value = "1st Dim.")
          ylabel:  (Default value = "2nd Dim.")
          legend:  (Default value = None)
          legend_orientation:  (Default value = 'vertical')
          legend_localtion:  (Default value = 'top_right')
          xlim:  (Default value = None)
          ylim:  (Default value = None)
          width:  (Default value = None)
          height:  (Default value = None)

        Returns:
          projected dataset X_project

        """

        n, d = X.shape

        if y is not None and y.size != X.shape[0]:
            exit("Data dims are not matched!")
        else:
            if d > 2:
                n_comp = 2
                if method == 'pca':
                    projector = PCA(n_components=n_comp)
                    X_proj = projector.fit_transform(X)
                elif method == 'kpca':
                    projector = KernelPCA(
                        n_components=n_comp, kernel='rbf', gamma=g)
                    X_proj = projector.fit_transform(X)
                elif method == 'mds':
                    projector = MDS(n_components=n_comp)
                    X_proj = projector.fit_transform(X)
                elif method == 'tsne':
                    projector = TSNE(n_components=n_comp, perplexity=100)
                    X_proj = projector.fit_transform(X)
                else:
                    print 'No projector found!'
                    X_proj = X
            else:
                X_proj = X

        f = self._get_figure_instance(title=title,
                                      xlabel=xlabel,
                                      ylabel=ylabel,
                                      xlim=xlim,
                                      ylim=ylim,
                                      width=width, height=height)
        if y is None:
            f.circle(X_proj[:, 0], X_proj[:, 1],
                     color=self.binary_colors[0], size=self.dot_size)
        else:
            if np.unique(y).size > 2:
                colors = getattr(importlib.import_module(
                    'bokeh.palettes'), self.colormap)
                colors = colors.__call__(np.unique(y).size)
            else:
                colors = self.binary_colors
            if legend is None:
                legend = ['class ' + str(c)
                          for c in np.arange(np.unique(y).size)]
            for c, l in enumerate(legend):
                f.circle(X_proj[y == np.unique(y)[c], 0], X_proj[y == np.unique(y)[c], 1],
                         line_color=None, size=self.dot_size,
                         fill_color=colors[c], legend=l)
            f.legend.location = legend_localtion
            f.legend.orientation = legend_orientation

        return f, X_proj

    def plot_corr(self, X, names=None,
                  title='Feature Correlations',
                  width=None, height=None):
        """Correlation matrix plot

        Args:
          X: 
          names:  (Default value = None)
          title:  (Default value = 'Feature Correlations')
          width:  (Default value = None)
          height:  (Default value = None)

        Returns:

        """

        n, d = X.shape
        xcorr = np.corrcoef(X.T)
        XX, YY = np.meshgrid(np.arange(1, d + 1), np.arange(1, d + 1))
        colors = []
        alphas = []
        for corr in xcorr.ravel():
            if corr > 0:
                colors.append(self.binary_colors[0])
                alphas.append(corr)
            elif corr < 0:
                colors.append(self.binary_colors[1])
                alphas.append(-corr)
            else:
                colors.append('lightgrey')
                alphas.append(self.alpha)

        dsource = ColumnDataSource(data=dict(
            xname=XX.ravel(),
            yname=YY.ravel(),
            colors=colors,
            alphas=alphas,
            corrs=xcorr.ravel()
        ))

        hover_tooltips = dict({
            'xname': '@xname',
            'yname': '@yname',
            'corr': '@corrs'
        })

        f = self._get_figure_instance(title=title, x_range=names, y_range=names, xlabel='', ylabel='',
                                      width=width, height=height)
        f.tools = [PanTool(), ResetTool()]
        f.add_tools(HoverTool(tooltips=hover_tooltips))
        f.grid.grid_line_color = None
        f.axis.axis_line_color = None
        f.axis.major_tick_line_color = None
        f.axis.major_label_text_font_size = "6pt"
        f.axis.major_label_standoff = 0
        f.xaxis.major_label_orientation = - np.pi / 2

        f.rect('xname', 'yname', 0.9, 0.9, source=dsource,
               color='colors', alpha='alphas', line_color=None,
               hover_line_color='black', hover_color='colors')
        return f

    def fill_between(self, xticks, mean, std, title='Error bar plot',
                     xlabel="xticks", ylabel="y values",
                     legend=None, xlim=None, ylim=None,
                     legend_loc='bottom_right', legend_orientation='vertical',
                     width=None, height=None):
        """plot a shaded error bar plot according to mean and std

        Args:
          xticks: 
          mean: 
          std: 
          title:  (Default value = 'Error bar plot')
          xlabel:  (Default value = "xticks")
          ylabel:  (Default value = "y values")
          legend:  (Default value = None)
          xlim:  (Default value = None)
          ylim:  (Default value = None)
          legend_loc:  (Default value = 'bottom_right')
          legend_orientation:  (Default value = 'vertical')
          width:  (Default value = None)
          height:  (Default value = None)

        Returns:

        """
        fig = self._get_figure_instance(title=title,
                                        xlabel=xlabel, ylabel=ylabel,
                                        xlim=xlim, ylim=ylim,
                                        width=width, height=height)

        band_x = np.append(xticks, xticks[::-1])
        if type(legend) is list:
            if len(legend) == 2:
                colors = self.binary_colors
            else:
                colors = getattr(importlib.import_module(
                    'bokeh.palettes'), self.colormap)
                colors = colors.__call__(len(legend))
            for m, s, c, l in zip(mean, std, colors, legend):
                band_y = np.append(m - s, (m + s)[::-1])
                fig.patch(band_x, band_y, color=c, fill_alpha=self.alpha)
                fig.line(xticks, m, line_width=self.line_width,
                         line_color=c, legend=l)
                fig.circle(xticks, m, size=self.dot_size, color=c)
        else:
            band_y = np.append(mean - std, (mean + std)[::-1])
            fig.patch(band_x, band_y, color=self.color, fill_alpha=self.alpha)
            fig.line(xticks, mean, line_width=self.line_width,
                     line_color=self.color, legend=legend)
            fig.circle(xticks, mean, size=self.dot_size, color=self.color)

        fig.legend.location = legend_loc
        fig.legend.orientation = legend_orientation

        return fig

    def simple_curves(self, xticks, yvalues, title='Simple curves plot',
                      xlabel="xticks", ylabel="y values",
                      legend=None, xlim=None, ylim=None,
                      width=None, height=None,
                      legend_loc='bottom_right', legend_orientation='vertical'):
        """plot simple curves
        :return: figure

        Args:
          xticks: 
          yvalues: 
          title:  (Default value = 'Simple curves plot')
          xlabel:  (Default value = "xticks")
          ylabel:  (Default value = "y values")
          legend:  (Default value = None)
          xlim:  (Default value = None)
          ylim:  (Default value = None)
          width:  (Default value = None)
          height:  (Default value = None)
          legend_loc:  (Default value = 'bottom_right')
          legend_orientation:  (Default value = 'vertical')

        Returns:

        """

        fig = self._get_figure_instance(title=title,
                                        xlabel=xlabel, ylabel=ylabel,
                                        xlim=xlim, ylim=ylim,
                                        width=width, height=height)
        fig.add_tools(HoverTool(tooltips={
            'y': '@y'
        }))

        if type(legend) is list:
            if len(legend) == 2:
                colors = self.binary_colors
            else:
                colors = getattr(importlib.import_module(
                    'bokeh.palettes'), self.colormap)
                colors = colors.__call__(len(legend))
            for y, c, l in zip(yvalues, colors, legend):
                dsource = ColumnDataSource(data={
                    'x': xticks,
                    'y': y
                    # l: y
                })
                fig.line(xticks, y, line_width=self.line_width,
                         line_color=c, legend=l)
                fig.circle(x='x', y='y', size=self.dot_size,
                           color=c, source=dsource)
        else:
            dsource = ColumnDataSource(data={
                'x': xticks,
                'y': yvalues,
                # legend: yvalues
            })
            fig.line(xticks, yvalues, line_width=self.line_width,
                     line_color=self.color, legend=legend)
            fig.circle(x='x', y='y', size=1, color=self.color, source=dsource)

        fig.legend.location = legend_loc
        fig.legend.orientation = legend_orientation

        return fig

    def send_to_server(self, server="ssh.informatik.tu-muenchen.de", port=22, user="xiaohu"):
        '''
        send plots to remote hosting server
        :param server: server IP
        :param port: server port
        :param user: server user
        :param pw: pass
        :param ssh_file: pubkey
        :return: status
        '''

        import paramiko
        from termcolor import colored

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        home_dir = os.path.expanduser('~')
        ssh.connect(hostname=server, port=port, username=user,
                    key_filename='/'.join([home_dir, '.ssh/id_rsa.pub']))
        ftp = ssh.open_sftp()
        stats = ftp.put(self.output_file, "/u/halle/xiaohu/home_page/html-data/h3demo/" +
                        self.output_file.split('/')[-1])
        print colored('{:s} is transferred to {:s} at {:s}'.format(
            self.output_file.split('/')[-1], server, str(stats.st_atime)), 'green')
        return stats
