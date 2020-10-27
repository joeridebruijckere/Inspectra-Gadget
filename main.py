# -*- coding: utf-8 -*-
"""
Inspectra-Gadget

Author: Joeri de Bruijckere

Last updated on Oct 26 2020

"""

# TODO: Properly handle files that have not completed the first two sweeps

from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os
import copy
import json
import io
from datetime import timedelta, datetime
from stat import ST_CTIME
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from matplotlib import cm
from matplotlib.widgets import Cursor
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.patches as patches
try: # lmfit is used for fitting the evolution of the properties of multiple peaks 
    from lmfit.models import LorentzianModel, GaussianModel, ConstantModel
    lmfit_imported = True
except ModuleNotFoundError:
    lmfit_imported = False
try: # used for single peak fitting
    from scipy.signal import find_peaks
    find_peaks_imported = True
except ModuleNotFoundError:
    find_peaks_imported = False
from collections import OrderedDict
from textwrap import wrap

try:
    import qcodes as qc
    qcodes_imported = True
except ModuleNotFoundError:
    qcodes_imported = False

import design
import filters
import fits

# Set default plot settings
DEFAULT_COLORMAP = 'magma'
DEFAULT_REVERSE_COLORMAP = True
DEFAULT_PLOT_SETTINGS = {}
DEFAULT_PLOT_SETTINGS['title'] = '<filename>'
DEFAULT_PLOT_SETTINGS['xlabel'] = ''
DEFAULT_PLOT_SETTINGS['ylabel'] = ''
DEFAULT_PLOT_SETTINGS['clabel'] = ''
DEFAULT_PLOT_SETTINGS['titlesize'] = '16'
DEFAULT_PLOT_SETTINGS['labelsize'] = '16' 
DEFAULT_PLOT_SETTINGS['ticksize'] = '16'
DEFAULT_PLOT_SETTINGS['linewidth'] = '1'
DEFAULT_PLOT_SETTINGS['spinewidth'] = '0.8'
DEFAULT_PLOT_SETTINGS['columns'] = '0,1,2'
DEFAULT_PLOT_SETTINGS['colorbar'] = 'True'
DEFAULT_PLOT_SETTINGS['minorticks'] = 'False'
DEFAULT_PLOT_SETTINGS['delimiter'] = ''
DEFAULT_PLOT_SETTINGS['linecolor'] = 'black'
DEFAULT_PLOT_SETTINGS['maskcolor'] = 'white'
DEFAULT_PLOT_SETTINGS['lut'] = '512'
DEFAULT_PLOT_SETTINGS['rasterized'] = 'True'
DEFAULT_PLOT_SETTINGS['dpi'] = '300'
DEFAULT_PLOT_SETTINGS['transparent'] = 'False'
DEFAULT_PLOT_SETTINGS['rc-filter'] = ''
DEFAULT_PLOT_SETTINGS['metadata'] = 'True'

# List of custom presets
PRESETS = [{'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5',
            'titlesize': '9',
            'canvas_bounds': (0.425,0.4,0.575,0.6), # (left, bottom, right, top)
            'show_meta_settings': False},
           {'title': '<metadataname>', 'labelsize': '16', 'ticksize': '16', 'spinewidth': '0.8',
            'titlesize': '16',
            'canvas_bounds': (0.425,0.4,0.575,0.6), # (left, bottom, right, top)
            'show_meta_settings': True},
           {'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5'},
           {'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5'}]

# Default settings meta.json files (Triton 6 data)
DEFAULT_CHANNEL = 'lockin_curr/X'
CONVERT_MICROSIEMENS_TO_ESQUAREDH = True
DEFAULT_VALUE_RCFILTER_CORRECT = False # If True: apply rc-filter correction by default upon loading data
DEFAULT_RC_FILTER = 8240 # Default resistance rc-filter(s)
DEFAULT_SHOW_METADATANAME = True
COMBINATION_GATE_SWEEPS_POSSIBLE = True # TODO fix for general case
CHANNELS_TO_SHOW = ['source', 'g1', 'g2', 'g3', 'g4', 'bg', 'sg', 'Bx', 'Bz', 'T']

#['source', 'g1', 'g1f', 'g2', 'g2f', 'g3', 'g3f', 
#                    'g4', 'g4f', 'g7', 'g7f', 'bg', 'Bx', 'Bz', 'T']

# QCoDeS default settings
DEFAULT_INDEX_DEPENDENT_PARAMETER = 0 # upon opening a dataset the dependent parameter with this index will be plotted

# Editor settings
PRINT_FUNCTION_CALLS = False # print function commands in terminal when called
SHOW_ERRORS = True # raise errors for debugging

# Matplotlib settings; font type is chosen such that text (labels, ticks, etc.) is recognized by Illustrator
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.cursive'] = ['Arial']
rcParams['mathtext.fontset'] = 'custom'

# Colormaps
cmaps = OrderedDict()
cmaps['Uniform'] = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential'] = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential (2)'] = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = [
    'Pastel1', 'Pastel2', 'Paired', 'Accent',
    'Dark2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

# Add custom hybrid colormaps to matplotlib register
cmaps['Hybrid'] = ['magma+bone_r','inferno+bone_r']
for cmap in cmaps['Hybrid']:
    n_colors = 512
    top = cm.get_cmap(cmap.split('+')[1], n_colors)
    bottom = cm.get_cmap(cmap.split('+')[0], n_colors)
    newcolors = np.vstack((top(np.linspace(0, 1, n_colors)),
                           bottom(np.linspace(0, 1, n_colors))))
    newcolors_r = newcolors[::-1]
    newcmp = ListedColormap(newcolors, name=cmap)
    newcmp_r = ListedColormap(newcolors_r, name=cmap+'_r')
    cm.register_cmap(cmap=newcmp)
    cm.register_cmap(cmap=newcmp_r)    

# Only include colormaps that are in the matplotlib register
for cmap_type in cmaps.copy():
    cmaps[cmap_type][:] = [cmap for cmap in cmaps[cmap_type] if cmap in plt.colormaps()]
    if cmaps[cmap_type] == []:
        del cmaps[cmap_type]
        

class Editor(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.window_title = 'Inspectra Gadget'
        self.window_title_auto_refresh = ''
        self.setupUi(self)
        self.set_theme()
        self.init_plot_settings()
        self.init_view_settings()
        self.init_filters()
        self.init_connections()
        self.init_canvas()
        self.linked_folder = None
        self.linked_files = []
        
    def set_theme(self):
        pass
        #pal = self.window().palette()
        #pal.setColor(QtGui.QPalette.Window, QtGui.QColor(0x40434a))
        #pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(0xd6d6d6))
        #self.window().setPalette(pal)
        
    def init_plot_settings(self):
        #font_sizes = ['xx-small','x-small','small','medium','large','x-large','xx-large']
        font_sizes = ['8', '9', '10', '12', '18', '24']
        self.settings_menu_list = OrderedDict()
        self.settings_menu_list['title'] = ['<filename>','<metadataname>']
        self.settings_menu_list['xlabel'] = ['Gate voltage (V)', '$V_{\mathrm{g}}$ (V)',
                                             'Bias voltage (mV)', '$V$ (mV)',
                                             'Magnetic Field (T)', '$B$ (T)', 'Angle (degrees)']
        self.settings_menu_list['ylabel'] = ['Bias voltage (mV)', '$V$ (mV)', 'Gate voltage (V)', '$V_{\mathrm{g}}$ (V)', 
                                             'd$I$/d$V$ (μS)', 'd$I$/d$V$ $(e^{2}/h)$', 'Angle (degrees)', 'Temperature (mK)']
        self.settings_menu_list['clabel'] = ['$I$ (nA)', '$I$ (a.u.)', 'Current (nA)', 
                                             'd$I$/d$V$ (μS)', 'd$I$/d$V$ ($G_0$)', 'd$I$/d$V$ (a.u.)', 
                                             'd$I$/d$V$ $(e^{2}/h)$', 'log$^{10}$(d$I$/d$V$ $(e^{2}/h)$)', 
                                             'd$^2I$/d$V^2$ (a.u.)', '|d$^2I$/d$V^2$| (a.u.)']
        self.settings_menu_list['titlesize'] = font_sizes
        self.settings_menu_list['labelsize'] = font_sizes
        self.settings_menu_list['ticksize'] = font_sizes
        self.settings_menu_list['colorbar'] = ['True', 'False']
        self.settings_menu_list['columns'] = ['0,1,2','0,1,3','0,2,3','1,2,4']
        self.settings_menu_list['minorticks'] = ['True','False']
        self.settings_menu_list['delimiter'] = ['',',']
        self.settings_menu_list['linecolor'] = ['black', 'red', 'white', 'blue', 'green']
        self.settings_menu_list['maskcolor'] = ['black','white']
        self.settings_menu_list['lut'] = ['128','256','512','1024']
        self.settings_menu_list['rasterized'] = ['False','True']
        self.settings_menu_list['dpi'] = ['figure']
        self.settings_menu_list['transparent'] = ['True', 'False']
        self.settings_menu_list['metadata'] = ['True', 'False']
        table = self.settings_table
        table.setColumnCount(2)
        table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        for col in range(2):
            table.horizontalHeader().setSectionResizeMode(col, 
                                  QtWidgets.QHeaderView.ResizeToContents)
        table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self.open_plot_settings_menu)
        self.copied_settings = None
    
    def init_view_settings(self):
        self.cmaps = cmaps
        for cmap_type in self.cmaps:    
            self.colormap_type_box.addItem(cmap_type)
        self.colormap_box.addItems(list(self.cmaps.values())[0])
        self.min_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.max_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.mid_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.copied_view_settings = None
    
    def init_filters(self):
        self.default_filter_settings = {
                'Derivative': {'Name': 'Derivative', 'Method': 'Midpoint',
                               'Setting 1': '0', 'Setting 2': '1', 'Checked': 2},
                'Smoothen': {'Name': 'Smoothen', 'Method': 'Gaussian',
                             'Setting 1': '0', 'Setting 2': '2', 'Checked': 2},
                'Sav-Gol': {'Name': 'Sav-Gol', 'Method': 'Y deriv 1',
                            'Setting 1': '7', 'Setting 2': '2', 'Checked': 2},                               
                'Crop X': {'Name': 'Crop X', 'Method': 'Absolute',
                           'Setting 1': '-1', 'Setting 2': '1', 'Checked': 0},                               
                'Crop Y': {'Name': 'Crop Y', 'Method': 'Absolute',
                           'Setting 1': '-10', 'Setting 2': '10', 'Checked': 0},
                'Roll X': {'Name': 'Roll X', 'Method': 'Index',
                           'Setting 1': '0', 'Setting 2': '0', 'Checked': 0},                               
                'Roll Y': {'Name': 'Roll Y', 'Method': 'Index',
                           'Setting 1': '0', 'Setting 2': '0', 'Checked': 0},
                'Cut X': {'Name': 'Cut X', 'Method': 'Index',
                          'Setting 1': '0', 'Setting 2': '0', 'Checked': 0},                               
                'Cut Y': {'Name': 'Cut Y', 'Method': 'Index',
                          'Setting 1': '0', 'Setting 2': '0', 'Checked': 0},                                
                'Swap XY': {'Name': 'Swap XY', 'Method': '',
                            'Setting 1': '', 'Setting 2': '', 'Checked': 2},
                'Flip': {'Name': 'Flip', 'Method': 'Left Right',
                         'Setting 1': '', 'Setting 2': '', 'Checked': 2},
                'Normalize': {'Name': 'Normalize', 'Method': 'Maximum',
                              'Setting 1': '0', 'Setting 2': '0', 'Checked': 2},
                'Offset': {'Name': 'Offset', 'Method': 'X',
                           'Setting 1': '0', 'Setting 2': '', 'Checked': 2},               
                'Absolute': {'Name': 'Absolute', 'Method': '',
                             'Setting 1': '', 'Setting 2': '', 'Checked': 2},                 
                'Multiply': {'Name': 'Multiply', 'Method': 'Z',
                             'Setting 1': '1', 'Setting 2': '', 'Checked': 2},
                'Slope': {'Name': 'Slope', 'Method': '',
                          'Setting 1': '0', 'Setting 2': '-1', 'Checked': 0},
                'Logarithm': {'Name': 'Logarithm', 'Method': 'Mask',
                              'Setting 1': '', 'Setting 2': '', 'Checked': 2},
                'Band cut': {'Name': 'Band cut', 'Method': 'Y',
                             'Setting 1': '20', 'Setting 2': '25', 'Checked': 0},
                'Interp': {'Name': 'Interp', 'Method': 'linear',
                           'Setting 1': '800', 'Setting 2': '600', 'Checked': 0},
                'Subtract': {'Name': 'Subtract', 'Method': 'Vertical',
                             'Setting 1': '0', 'Setting 2': '', 'Checked': 0},
                'Divide': {'Name': 'Divide', 'Method': 'Z',
                             'Setting 1': '1', 'Setting 2': '', 'Checked': 2},
                'Invert': {'Name': 'Invert', 'Method': 'Z',
                             'Setting 1': '', 'Setting 2': '', 'Checked': 2}}                  
        self.filters_combobox.addItem('<Add Filter>')
        self.filters_combobox.addItems(filters.get_list())
        table = self.filters_table
        table.setColumnCount(4)
        table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in range(1,4):
            table.horizontalHeader().setSectionResizeMode(col, 
                                  QtWidgets.QHeaderView.ResizeToContents)
        self.copied_filters = None
        
    def init_connections(self):
        self.open_files_button.clicked.connect(self.open_files)
        self.delete_files_button.clicked.connect(lambda: self.remove_files('current'))
        self.clear_files_button.clicked.connect(lambda: self.remove_files('all'))
        self.file_list.itemChanged.connect(self.file_checked)
        self.file_list.itemClicked.connect(self.file_clicked)
        self.file_list.itemDoubleClicked.connect(self.file_double_clicked)
        self.settings_table.itemChanged.connect(self.plot_setting_edited)
        self.filters_table.itemChanged.connect(self.filters_table_edited)
        self.copy_settings_button.clicked.connect(self.copy_plot_settings)
        self.paste_settings_button.clicked.connect(lambda: self.paste_plot_settings('copied'))
        self.reset_settings_button.clicked.connect(lambda: self.paste_plot_settings('default'))
        self.filters_combobox.currentIndexChanged.connect(self.filters_box_changed)
        self.delete_filters_button.clicked.connect(lambda: self.remove_filters('current'))
        self.clear_filters_button.clicked.connect(lambda: self.remove_filters('all'))
        self.copy_filters_button.clicked.connect(self.copy_filters)
        self.paste_filters_button.clicked.connect(lambda: self.paste_filters('copied'))
        self.up_filters_button.clicked.connect(lambda: self.move_filter(-1))
        self.down_filters_button.clicked.connect(lambda: self.move_filter(1))
        self.previous_button.clicked.connect(self.to_previous_file)
        self.next_button.clicked.connect(self.to_next_file)
        self.copy_view_button.clicked.connect(self.copy_view_settings)
        self.paste_view_button.clicked.connect(lambda: self.paste_view_settings('copied'))
        self.colormap_type_box.currentIndexChanged.connect(self.colormap_type_edited)
        self.colormap_box.currentIndexChanged.connect(self.colormap_edited)
        self.reverse_colors_box.clicked.connect(self.colormap_edited)
        self.min_line_edit.editingFinished.connect(lambda: self.view_setting_edited('Minimum'))
        self.max_line_edit.editingFinished.connect(lambda: self.view_setting_edited('Maximum'))
        self.mid_line_edit.editingFinished.connect(lambda: self.view_setting_edited('Midpoint'))
        self.lock_checkbox.clicked.connect(lambda: self.view_setting_edited('Locked'))
        self.mid_checkbox.clicked.connect(lambda: self.view_setting_edited('MidLock'))
        self.reset_limits_button.clicked.connect(self.reset_color_limits)
        self.save_image_button.clicked.connect(self.save_image)
        self.copy_image_button.clicked.connect(self.copy_canvas_to_clipboard)
        self.load_filters_button.clicked.connect(self.load_filters)
        self.action_filters.triggered.connect(self.save_filters)
        self.action_current_file.triggered.connect(lambda: self.save_session('current'))
        self.action_all_files.triggered.connect(lambda: self.save_session('all'))
        self.action_checked_files.triggered.connect(lambda: self.save_session('checked'))
        self.action_merge_raw.triggered.connect(lambda: self.merge_files(raw_data=True))
        self.action_merge_processed.triggered.connect(lambda: self.merge_files(raw_data=False))
        self.action_live_tracking.triggered.connect(lambda: self.start_auto_refresh(time_interval=1))
        self.action_refresh_30s.triggered.connect(lambda: self.start_auto_refresh(time_interval=30))
        self.action_refresh_5m.triggered.connect(lambda: self.start_auto_refresh(time_interval=300))
        self.action_refresh_30m.triggered.connect(lambda: self.start_auto_refresh(time_interval=1800))
        self.action_refresh_stop.triggered.connect(self.stop_auto_refresh)
        self.action_open_files_from_folder.triggered.connect(self.open_files_from_folder)
        self.action_save_files_as_PNG.triggered.connect(lambda: self.save_files_as('.png'))
        self.action_save_files_as_PDF.triggered.connect(lambda: self.save_files_as('.pdf'))
        self.action_preset_0.triggered.connect(lambda: self.apply_preset(0))
        self.action_preset_1.triggered.connect(lambda: self.apply_preset(1))
        self.action_preset_2.triggered.connect(lambda: self.apply_preset(2))
        self.action_preset_3.triggered.connect(lambda: self.apply_preset(3))
        self.action_refresh_stop.setEnabled(False)
        self.action_link_to_folder.triggered.connect(lambda: self.update_link_to_folder(new_folder=True))
        self.action_unlink_folder.triggered.connect(self.unlink_folder)
        self.refresh_file_button.clicked.connect(self.refresh_plot)
        self.up_file_button.clicked.connect(lambda: self.move_file('up'))
        self.down_file_button.clicked.connect(lambda: self.move_file('down'))
        self.file_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.open_item_menu)
    
    def init_canvas(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.mouse_click_canvas)
        self.canvas.mpl_connect('scroll_event', self.mouse_scroll_canvas)
        self.navi_toolbar = NavigationToolbarMod(self.canvas, self)
        self.graph_layout.addWidget(self.navi_toolbar)
        self.graph_layout.addWidget(self.canvas)
        self.subplot_grid = [(1,1),(1,2),(2,2),(2,2),(2,3),(2,3),(2,4),(2,4),
                             (3,4),(3,4),(3,4),(3,4),(4,4),(4,4),(4,4),(4,4),
                             (4,5),(4,5),(4,5),(4,5),(4,5),(5,5),(5,5),(5,5),
                             (5,5)]
        if DEFAULT_PLOT_SETTINGS['metadata'] == 'True':
            self.figure.subplots_adjust(0.17,0.2,0.8,0.85)

    def open_files(self, filenames=None):
        if PRINT_FUNCTION_CALLS:
            print('open_files')
        try:
            if not filenames:
                if qcodes_imported:
                    filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
                            self, 'Open File', '', 'Data Files (*.dat *.npy *.db)')
                else:                    
                    filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
                            self, 'Open File', '', 'Data Files (*.dat *.npy)')
            if filenames:
                self.file_list.itemChanged.disconnect(self.file_checked)
                for filename in filenames:
                    print('Open '+filename)
                    extension = os.path.splitext(filename)[1]
                    if extension == '.dat': 
                        self.add_file(filename)
                    elif extension == '.npy':
                        item_data_all = np.load(filename, allow_pickle=True)
                        for item_data in item_data_all:
                            self.add_file(item_data['File Name'], data=item_data)
                    elif extension == '.db': # QCoDeS database
                        qc.initialise_or_create_database_at(filename)
                        exp = qc.load_last_experiment()
                        datasets = exp.data_sets()
                        for dataset in datasets:
                            filename = 'Run #{} {} ({}) {}'.format(dataset.captured_run_id, dataset.exp_name, 
                                                                   dataset.sample_name, dataset.run_timestamp())
                            self.add_file(filename, data=dataset)
                last_item = self.file_list.item(self.file_list.count()-1)
                self.file_list.setCurrentItem(last_item)
                for item_index in range(self.file_list.count()-1):
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
                self.show_current_all()
                self.file_list.itemChanged.connect(self.file_checked)
                last_item.setCheckState(QtCore.Qt.Checked)
        except:
            print('Could not open file(s)...')
            if SHOW_ERRORS:
                raise
            
    def add_file(self, file, data=None, do_load_data=True):
        if PRINT_FUNCTION_CALLS:
            print('add_file')
        item = QtWidgets.QListWidgetItem()
        try:
            item.setData(QtCore.Qt.UserRole, 
                         Data(filepath=file, canvas=self.canvas, 
                              data=data, do_load_data=do_load_data))
            item.setText(os.path.basename(file))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.file_list.addItem(item)
            if item.data(QtCore.Qt.UserRole).meta_data:
                item.setText(item.data(QtCore.Qt.UserRole).meta_data_name)
        except:
            print('Could not add', file,'...')
            if SHOW_ERRORS:
                raise
    
    def remove_files(self, which='current'):
        if PRINT_FUNCTION_CALLS:
            print('remove_files')
        update_plots = False
        if self.file_list.count() > 0:
            if which == 'current':
                items = [self.file_list.currentItem()]
            elif which == 'all':
                items = [self.file_list.item(n) for n in range(self.file_list.count())]
            for item in items:
                data = item.data(QtCore.Qt.UserRole) 
                if data.filepath in self.linked_files and data.duplicate == False: # TODO if duplicates exist file is not removed
                    self.linked_files.remove(item.data(QtCore.Qt.UserRole).filepath)
                if item.checkState() == 2:
                    update_plots = True
                index = self.file_list.row(item)
                self.file_list.takeItem(index)
                del item

        self.show_current_all()
        if update_plots: 
            self.update_plots()
    
    def file_checked(self, item):
        if PRINT_FUNCTION_CALLS:
            print('file_checked')
        try:
            if item.checkState() == 2:
                self.file_list.setCurrentItem(item)
            self.update_plots()
        except:
            print('Could not update plot(s)...')
            if SHOW_ERRORS:
                raise
    
    def file_clicked(self):
        if PRINT_FUNCTION_CALLS:
            print('file_clicked')
        try:
            self.show_current_all()
        except:
            print('Could not show current settings...')
            if SHOW_ERRORS:
                raise
            
    def file_double_clicked(self, item):
        if PRINT_FUNCTION_CALLS:
            print('file_double_clicked')
        try:
            self.file_list.itemChanged.disconnect(self.file_checked)
            for item_index in range(self.file_list.count()):
                self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
            item.setCheckState(QtCore.Qt.Checked)
            self.file_list.itemChanged.connect(self.file_checked)
            self.update_plots()
        except:
            print('Could not update plot(s)...')
            if SHOW_ERRORS:
                raise
    
    def update_plots(self):
        if PRINT_FUNCTION_CALLS:
            print('update_plots')
        self.figure.clear()
        file_list = self.file_list
        checked_items = [file_list.item(index) for index in range(file_list.count()) 
                         if file_list.item(index).checkState() == 2]
        if checked_items:
            self.subplot_rows, self.subplot_cols = self.subplot_grid[len(checked_items)-1]
            for index, item in enumerate(checked_items):
                try:
                    data = item.data(QtCore.Qt.UserRole)
                    if data.processed_data == None: # if data is loaded for first time
                        data.load_data()
                        if data.meta_data:
                            data.interpret_meta_file()
                            data.processed_to_raw()
                            data.rcfilter_correct = DEFAULT_VALUE_RCFILTER_CORRECT
                            data.apply_all_filters(update_color_limits=False, refresh_unit_conversion=True)
                        else:
                            data.apply_all_filters()
                    data.figure = self.figure
                    data.axes = data.figure.add_subplot(
                            self.subplot_rows, self.subplot_cols, index+1)
                    
                    if len(data.columns) == 2:
                        data.add_plot_2d()
                    else:
                        data.add_plot()
                        data.reset_view_settings()
                        data.apply_view_settings()
                        try:
                            data.linecut_window
                        except AttributeError:
                            pass
                        else:
                            data.update_linecut()
                        try:
                            data.multi_linecuts_window
                        except AttributeError:
                            pass
                        else:
                            if data.multi_linecuts_window.isVisible():
                                data.update_multiple_linecuts()
                except:
                    print('Could not open', data.filepath)
                    raise
        self.show_current_all()
        self.canvas.draw()
          
    def refresh_plot(self):
        if PRINT_FUNCTION_CALLS:
            print('refresh_plot')
        if self.linked_folder:
            self.update_link_to_folder(new_folder=False)
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            data.refresh_data(update_color_limits=False, refresh_unit_conversion=False)
            self.update_plots()
            
    def to_next_file(self):
        if PRINT_FUNCTION_CALLS:
            print('to_next_file')
        checked_items = [(self.file_list.item(index), index) for index in range(self.file_list.count()) 
                         if self.file_list.item(index).checkState() == 2]
        if len(checked_items) == 1 and self.file_list.count() > 1 and checked_items[0][1]+1 < self.file_list.count():
            item = checked_items[0][0]
            next_item = self.file_list.item(checked_items[0][1]+1)
            self.file_list.itemChanged.disconnect(self.file_checked)
            item.setCheckState(QtCore.Qt.Unchecked)
            next_item.setCheckState(QtCore.Qt.Checked)
            self.file_list.setCurrentItem(next_item)
            self.file_list.itemChanged.connect(self.file_checked)
            self.update_plots()
        
    def to_previous_file(self):
        if PRINT_FUNCTION_CALLS:
            print('to_previous_file')
        checked_items = [(self.file_list.item(index), index) for index in range(self.file_list.count()) 
                         if self.file_list.item(index).checkState() == 2]
        if len(checked_items) == 1 and self.file_list.count() > 1 and checked_items[0][1] > 0:
            item = checked_items[0][0]
            previous_item = self.file_list.item(checked_items[0][1]-1)
            self.file_list.itemChanged.disconnect(self.file_checked)
            item.setCheckState(QtCore.Qt.Unchecked)
            previous_item.setCheckState(QtCore.Qt.Checked)
            self.file_list.setCurrentItem(previous_item)
            self.file_list.itemChanged.connect(self.file_checked)
            self.update_plots()
        
    def start_auto_refresh(self, time_interval):
        if PRINT_FUNCTION_CALLS:
            print('start_auto_refresh')
        self.auto_refresh_timer = QtCore.QTimer()
        self.auto_refresh_timer.setInterval(time_interval*1000)
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_call)
        self.action_refresh_stop.setEnabled(True)
        self.auto_refresh_timer.start()
        self.window_title_auto_refresh = ' - Auto-Refreshing Enabled'
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
        self.auto_refresh_call()
        
    def auto_refresh_call(self):
        if self.linked_folder:
            self.update_link_to_folder(new_folder=False)
        self.window_title_auto_refresh = ' - Auto-Refreshing Enabled (Refreshing...)'
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
        checked_items = [self.file_list.item(index) for index in range(self.file_list.count()) 
                         if self.file_list.item(index).checkState() == 2]        
        # Refresh all checked items
        if checked_items:
            for index, item in enumerate(checked_items):
                item.data(QtCore.Qt.UserRole).refresh_data(update_color_limits=False, 
                                                           refresh_unit_conversion=False)
        self.update_plots()
        self.window_title_auto_refresh = ' - Auto-Refreshing Enabled'
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)        
        
        # Update progress of last item (if checked)
        last_item = self.file_list.item(self.file_list.count()-1)
        data = last_item.data(QtCore.Qt.UserRole)
        if last_item.checkState() == 2 and data.last_modified_time:
            if datetime.now().timestamp() - data.last_modified_time > 120 or data.progress_fraction == 1:
                print('Stop auto refresh...')
                data.remaining_time_string = ''
                self.stop_auto_refresh()
            else:
                self.remaining_time_label.setText(data.remaining_time_string)
        else:
            self.remaining_time_label.setText('')
            
    def stop_auto_refresh(self):
        if PRINT_FUNCTION_CALLS:
            print('stop_auto_refresh')
        self.auto_refresh_timer.stop()
        self.action_refresh_stop.setEnabled(False)
        self.window_title_auto_refresh = ''
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
        self.remaining_time_label.setText('')
        
    def move_file(self, direction):
        if PRINT_FUNCTION_CALLS:
            print('move_file')
        current_item = self.file_list.currentItem()
        if current_item:
            current_row = self.file_list.currentRow()
            if direction == 'up' and current_row > 0:
                new_row = current_row-1
            elif direction == 'down' and current_row < self.file_list.count()-1:
                new_row = current_row+1
            else:
                new_row = current_row
            if new_row != current_row:
                if current_item.checkState() == 2 and self.file_list.item(new_row).checkState() == 2:
                    update_canvas = True
                else:
                    update_canvas = False
                self.file_list.takeItem(current_row)
                self.file_list.insertItem(new_row, current_item)
                self.file_list.setCurrentRow(new_row)
                if update_canvas:
                    self.update_plots()
                    self.canvas.draw()
        
    def show_current_all(self):
        if PRINT_FUNCTION_CALLS:
            print('show_current_all')
        self.show_current_plot_settings()
        self.show_current_view_settings()
        self.show_current_filters()
    
    def show_current_plot_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('showCurrentSettings')
        item = self.file_list.currentItem()
        if item:
            table = self.settings_table
            table.itemChanged.disconnect(self.plot_setting_edited)
            table.setRowCount(0)
            settings = item.data(QtCore.Qt.UserRole).settings
            for key, value in list(settings.items()):
                row = table.rowCount()
                table.insertRow(row)
                property_item = QtWidgets.QTableWidgetItem(key)
                property_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 0, property_item)
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
            table.itemChanged.connect(self.plot_setting_edited)
            
    def show_current_view_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('show_current_view_settings')
        current_item = self.file_list.currentItem()
        if current_item:
            settings = current_item.data(QtCore.Qt.UserRole).view_settings
            self.min_line_edit.setText('%.4g' % settings['Minimum'])
            self.max_line_edit.setText('%.4g' % settings['Maximum'])
            self.mid_line_edit.setText('%.4g' % settings['Midpoint'])
            if settings['Locked']:
                self.lock_checkbox.setCheckState(QtCore.Qt.Checked)
            else:
                self.lock_checkbox.setCheckState(QtCore.Qt.Unchecked)
            if settings['MidLock']:
                self.mid_checkbox.setCheckState(QtCore.Qt.Checked)
            else:
                self.mid_checkbox.setCheckState(QtCore.Qt.Unchecked)
            self.colormap_type_box.currentIndexChanged.disconnect(self.colormap_type_edited)
            self.colormap_type_box.setCurrentText(settings['Color Map Type'])
            self.colormap_type_box.currentIndexChanged.connect(self.colormap_type_edited)
            self.fill_colormap_box()
            self.colormap_box.currentIndexChanged.disconnect(self.colormap_edited)
            self.colormap_box.setCurrentText(settings['Color Map'])
            self.colormap_box.currentIndexChanged.connect(self.colormap_edited)
            if settings['Reverse']:
                self.reverse_colors_box.setCheckState(QtCore.Qt.Checked)
            else:
                self.reverse_colors_box.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.min_line_edit.setText('')
            self.max_line_edit.setText('')
            self.lock_checkbox.setCheckState(QtCore.Qt.Unchecked)
            self.mid_checkbox.setCheckState(QtCore.Qt.Unchecked)
    
    def show_current_filters(self):
        if PRINT_FUNCTION_CALLS:
            print('show_current_filters')
        table = self.filters_table
        table.setRowCount(0)
        current_item = self.file_list.currentItem()
        if current_item:
            filters = current_item.data(QtCore.Qt.UserRole).filters
            for f in filters:
                self.append_filter_to_table()
    
    def plot_setting_edited(self):
        if PRINT_FUNCTION_CALLS:
            print('plot_setting_edited')
        current_item = self.file_list.currentItem()
        data = current_item.data(QtCore.Qt.UserRole)
        data.old_settings = data.settings.copy()
        if current_item:
            settings = data.settings
            table = self.settings_table
            row = table.currentRow()
            setting_name = table.item(row, 0).text()
            value = table.item(row, 1).text()
            settings[setting_name] = value
            table.clearFocus()
            try:
                if setting_name == 'columns':
                    data.columns = [int(s) for s in value.split(',')]
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
                    self.update_plots()
                    self.show_current_all()            
                elif setting_name == 'delimiter':
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
                    self.update_plots()
                    self.show_current_all()
                elif setting_name == 'rc-filter':
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=False)
                    self.update_plots()
                    self.show_current_all()
                elif setting_name == 'linecolor':
                    for line in data.axes.get_lines():
                        line.set_color(value)
                    self.canvas.draw()
                elif setting_name == 'maskcolor':
                    data.apply_colormap()
                    self.canvas.draw()
                elif setting_name == 'lut':
                    data.apply_colormap()
                    self.canvas.draw()
                elif setting_name == 'rasterized' or setting_name == 'colorbar':
                    self.update_plots()
                elif setting_name == 'minorticks':
                    self.update_plots()
                elif setting_name == 'metadata':
                    self.update_plots()
                data.apply_plot_settings()
                self.canvas.draw()
            except: # if invalid value is put in: reset to previous settings
                print('Invalid value of plot setting!')
                self.paste_plot_settings(which='old')
    
    def view_setting_edited(self, edited_setting):
        current_item = self.file_list.currentItem()
        data = current_item.data(QtCore.Qt.UserRole)
        data.old_view_settings = data.view_settings.copy()
        if current_item:
            view_settings = data.view_settings
            try:
                if edited_setting == 'Minimum':
                    new_value = float(self.min_line_edit.text())
                    view_settings[edited_setting] = new_value
                    self.min_line_edit.setText('%.4g' % new_value)
                    self.min_line_edit.clearFocus()
                    data.reset_midpoint()
                    self.mid_line_edit.setText('%.4g' % view_settings['Midpoint'])
                elif edited_setting == 'Maximum':
                    new_value = float(self.max_line_edit.text())
                    view_settings[edited_setting] = new_value
                    self.max_line_edit.setText('%.4g' % new_value)
                    self.max_line_edit.clearFocus()
                    data.reset_midpoint()
                    self.mid_line_edit.setText('%.4g' % view_settings['Midpoint'])
                elif edited_setting == 'Midpoint':
                    if self.mid_line_edit.text():
                        new_value = float(self.mid_line_edit.text())
                        view_settings[edited_setting] = new_value
                    else:
                        data.reset_midpoint()
                        new_value = view_settings[edited_setting]
                    self.mid_line_edit.setText('%.4g' % new_value)
                    self.mid_line_edit.clearFocus()
                elif edited_setting == 'Locked':
                    view_settings[edited_setting] = self.lock_checkbox.checkState()
                elif edited_setting == 'MidLock':
                    view_settings[edited_setting] = self.mid_checkbox.checkState()
                data.apply_view_settings()
                self.canvas.draw()
            except:
                print('Invalid value of view setting!')
                self.paste_view_settings(which='old')
                
    
    def fill_colormap_box(self):
        if PRINT_FUNCTION_CALLS:
            print('fill_colormap_box')
        self.colormap_box.currentIndexChanged.disconnect(self.colormap_edited)
        self.colormap_box.clear()
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentText()])
        self.colormap_box.currentIndexChanged.connect(self.colormap_edited)
    
    def colormap_type_edited(self):
        if PRINT_FUNCTION_CALLS:
            print('colormap_type_edited')
        self.fill_colormap_box()
        self.colormap_edited()
        
    def colormap_edited(self):
        if PRINT_FUNCTION_CALLS:
            print('colormap_edited')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            settings = data.view_settings
            settings['Color Map Type'] = self.colormap_type_box.currentText()
            settings['Color Map'] = self.colormap_box.currentText()
            settings['Reverse'] = self.reverse_colors_box.isChecked()
            if current_item.checkState():
                data.apply_colormap()
                self.canvas.draw()
    
    def filters_table_edited(self, item):
        if PRINT_FUNCTION_CALLS:
            print('filters_table_edited')
        current_item = self.file_list.currentItem()
        data = current_item.data(QtCore.Qt.UserRole)
        data.oldFilters = copy.deepcopy(data.filters)
        if current_item:   
            table = self.filters_table
            try:
                filterItem = table.item(item.row(), 0)
                filter_settings = data.filters[item.row()]
                filter_settings['Checked'] = filterItem.checkState()
                filter_settings['Name'] = filterItem.text()
                filter_settings['Method'] = table.cellWidget(item.row(), 1).currentText()
                filter_settings['Setting 1'] = table.item(item.row(), 2).text()
                filter_settings['Setting 2'] = table.item(item.row(), 3).text()
                table.clearFocus()
                data.apply_all_filters(update_color_limits=False)
                data.reset_view_settings()
                if current_item.checkState():
                    if ((filterItem.checkState()) | (item.column() == 0)):
                        self.update_plots()
                        self.show_current_filters()
                        self.show_current_view_settings()
            except:
                print('Invalid value of filter!')
                self.paste_filters(which='old')
    
    def copy_plot_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('copy_plot_settings')
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_settings = current_item.data(QtCore.Qt.UserRole).settings.copy()
    
    def copy_filters(self):
        if PRINT_FUNCTION_CALLS:
            print('copy_filters')
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_filters = copy.deepcopy(current_item.data(QtCore.Qt.UserRole).filters)
            
    def copy_view_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('copy_view_settings')
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_view_settings = current_item.data(QtCore.Qt.UserRole).view_settings.copy()
    
    def paste_plot_settings(self, which='copied'):
        if PRINT_FUNCTION_CALLS:
            print('paste_plot_settings')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            if which == 'copied':
                if self.copied_settings:
                    data.settings = self.copied_settings.copy()
            elif which == 'default':
                data.settings = data.default_settings.copy()
            elif which == 'old':
                data.settings = data.old_settings.copy()
            self.show_current_plot_settings()
            if current_item.checkState():
                data.apply_plot_settings()
                self.canvas.draw()
    
    def paste_filters(self, which='copied'):
        if PRINT_FUNCTION_CALLS:
            print('paste_filters')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            if which == 'copied':
                if self.copied_filters:
                    data.filters = copy.deepcopy(self.copied_filters)
            elif which == 'old':
                data.filters = copy.deepcopy(data.oldFilters)
            self.show_current_filters()
            data.apply_all_filters()
            self.show_current_view_settings()
            if current_item.checkState():
                self.update_plots()
                self.canvas.draw()

    def paste_view_settings(self, which='copied'):
        if PRINT_FUNCTION_CALLS:
            print('paste_view_settings')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            if which == 'copied':
                if self.copied_view_settings:
                    data.view_settings = self.copied_view_settings.copy()
            elif which == 'old':
                data.view_settings = data.old_view_settings.copy()
            self.show_current_view_settings()
            if current_item.checkState():
                data.apply_view_settings()
                data.apply_colormap()
                self.canvas.draw()

    def open_item_menu(self):
        if PRINT_FUNCTION_CALLS:
            print('open_item_menu')
        current_item = self.file_list.currentItem()
        if current_item:
            checked_items = [self.file_list.item(index) for index in range(self.file_list.count()) 
                             if self.file_list.item(index).checkState() == 2]
            menu = QtWidgets.QMenu(self)
            actions = ['Duplicate...','Check all...']
            if len(checked_items) > 1:
                actions.append('Combine plots...')
            for entry in actions:
                action = QtWidgets.QAction(entry, self)
                menu.addAction(action)
            menu.triggered[QtWidgets.QAction].connect(self.do_item_action)
            menu.popup(QtGui.QCursor.pos())
            
    def do_item_action(self, signal):
        if PRINT_FUNCTION_CALLS:
            print('open_item_menu')
        current_item = self.file_list.currentItem()
        if current_item:
            if signal.text() == 'Duplicate...': # TODO remove duplicate from linked folder list
                self.file_list.itemChanged.disconnect(self.file_checked)
                current_data = current_item.data(QtCore.Qt.UserRole)
                item = QtWidgets.QListWidgetItem()
                item.setData(QtCore.Qt.UserRole, Data(current_data.filepath, self.canvas))
                item.data(QtCore.Qt.UserRole).settings = current_data.settings.copy()
                item.data(QtCore.Qt.UserRole).view_settings = current_data.view_settings.copy()
                item.data(QtCore.Qt.UserRole).filters = copy.deepcopy(current_data.filters)
                item.data(QtCore.Qt.UserRole).duplicate = True
                item.setText(current_item.text())
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.file_list.addItem(item)
                if item.data(QtCore.Qt.UserRole).meta_data:
                    item.setText(item.data(QtCore.Qt.UserRole).meta_data_name)                
                self.file_list.itemChanged.connect(self.file_checked)
                self.update_plots()
                self.file_list.setCurrentItem(item)
            elif signal.text() == 'Check all...':
                self.file_list.itemChanged.disconnect(self.file_checked)
                for item_index in range(self.file_list.count()):                        
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Checked)
                    self.file_list.itemChanged.connect(self.file_checked)
                self.update_plots()
            elif signal.text() == 'Combine plots...':
                try:
                    self.combine_plots()
                except:
                    print('Cannot combine these plots...')
                
    def combine_plots(self):
        items = self.file_list
        checked_items = [items.item(index) for index in range(items.count()) 
                            if items.item(index).checkState() == 2]
        if checked_items:
            title = '[combined] '
            data_list = []
            for item in checked_items:
                split_text = item.text().split(' ')
                title += '{} {} '.format(split_text[0],split_text[1])
                data_list.append(item.data(QtCore.Qt.UserRole))
            self.multi_plot_window = LineCutWindow(parent=data_list, multiple=True)
            self.multi_plot_window.title = title
            self.multi_plot_window.xlabel = data_list[0].settings['xlabel']
            self.multi_plot_window.ylabel = data_list[0].settings['ylabel']
            self.multi_plot_window.zlabel = None
            self.multi_plot_window.draw_plots()
            self.multi_plot_window.show()
    
    def open_plot_settings_menu(self):
        if PRINT_FUNCTION_CALLS:
            print('open_plot_settings_menu')
        table = self.settings_table
        row = table.currentRow()
        column = table.currentColumn()
        if column == 1:
            setting_name = table.item(row, 0).text()
            menu = QtWidgets.QMenu(self)
            if setting_name in self.settings_menu_list.keys():
                for entry in self.settings_menu_list[setting_name]:
                    action = QtWidgets.QAction(entry, self)
                    menu.addAction(action)
                menu.triggered[QtWidgets.QAction].connect(self.replace_plot_setting)
                menu.popup(QtGui.QCursor.pos())
   
    def replace_plot_setting(self, signal):
        if PRINT_FUNCTION_CALLS:
            print('replaceSetting')
        table = self.settings_table
        item = table.currentItem()
        item.setText(signal.text())

    def reset_color_limits(self):
        if PRINT_FUNCTION_CALLS:
            print('reset_color_limits')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            data.reset_view_settings(overrule=True)
            self.show_current_view_settings()
            if current_item.checkState():
                data.apply_view_settings()
                self.canvas.draw()
    
    def filters_box_changed(self):
        if PRINT_FUNCTION_CALLS:
            print('filters_box_changed')
        current_item = self.file_list.currentItem()
        box = self.filters_combobox
        if current_item:
            filter_name = box.currentText()
            new_filter = copy.deepcopy(self.default_filter_settings[filter_name])
            data = current_item.data(QtCore.Qt.UserRole)
            data.apply_filter(new_filter, update_color_limits=False)
            data.reset_view_settings()
            self.append_filter_to_table()
            self.show_current_view_settings()
            if current_item.checkState():
                self.update_plots()
        box.currentIndexChanged.disconnect(self.filters_box_changed)
        box.setCurrentIndex(0)
        box.clearFocus()
        box.currentIndexChanged.connect(self.filters_box_changed)
    
    def append_filter_to_table(self):
        if PRINT_FUNCTION_CALLS:
            print('append_filter_to_table')
        current_item = self.file_list.currentItem()
        if current_item:
            table = self.filters_table
            row = table.rowCount()
            settings = current_item.data(QtCore.Qt.UserRole).filters[row]           
            table.itemChanged.disconnect(self.filters_table_edited)
            table.insertRow(row)           
            type_item = QtWidgets.QTableWidgetItem(settings['Name'])
            type_item.setFlags(QtCore.Qt.ItemIsSelectable | 
                    QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            if settings['Checked']:
                type_item.setCheckState(QtCore.Qt.Checked)
            else:
                type_item.setCheckState(QtCore.Qt.Unchecked)
            method_list = filters.get_list(settings['Name'])
            property_box = NoScrollQComboBox()
            property_box.setFocusPolicy(QtCore.Qt.StrongFocus)
            property_box.addItems(method_list)
            property_index = [i for i, method in enumerate(method_list) if method_list[i] == settings['Method']]
            if property_index != []:
                property_box.setCurrentIndex(property_index[0])
            setting_item_1 = QtWidgets.QTableWidgetItem(str(settings['Setting 1']))
            setting_item_2 = QtWidgets.QTableWidgetItem(str(settings['Setting 2']))
            property_box.currentIndexChanged.connect(lambda: self.filters_table_edited(setting_item_1))
            table.setItem(row, 0, type_item)
            table.setCellWidget(row, 1, property_box)
            table.setItem(row, 2, setting_item_1)
            table.setItem(row, 3, setting_item_2)
            table.item(row, 2).setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            table.item(row, 3).setTextAlignment(int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter))
            table.setCurrentCell(row, 0)
            table.itemChanged.connect(self.filters_table_edited)
    
    def remove_filters(self, which='current'):
        if PRINT_FUNCTION_CALLS:
            print('remove_filters')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            table = self.filters_table
            if which == 'current':
                filter_row = table.currentRow()
                if filter_row != -1:
                    table.removeRow(filter_row)
                    del data.filters[filter_row]
                    data.apply_all_filters(update_color_limits=False)
                    data.reset_view_settings()
            elif which == 'all':
                table.setRowCount(0)
                data.filters = []
                data.apply_all_filters(update_color_limits=False)
                data.reset_view_settings()
            if current_item.checkState():
                data.apply_view_settings()
                self.update_plots()
                self.show_current_view_settings()
     
    def move_filter(self, to=1):
        if PRINT_FUNCTION_CALLS:
            print('move_filter')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            filters = data.filters
            table = self.filters_table
            row = table.currentRow()
            if (((row > 0) & (to == -1)) | ((row < table.rowCount()-1) & (to == 1))):
                filters[row], filters[row+to] = filters[row+to], filters[row]
                self.show_current_filters()
                table.setCurrentCell(row+to, 0)
                if ((table.item(row,0).checkState()) & (table.item(row+to,0).checkState())):
                    data.apply_all_filters()
                    self.update_plots()
                    self.show_current_view_settings()

    def save_image(self):
        if PRINT_FUNCTION_CALLS:
            print('save_as')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            data_name, _ = os.path.splitext(data.filepath)
            formats = 'Adobe Acrobat (*.pdf);;Portable Network Graphic (*.png)'
            filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Figure As...', data_name.replace(':',''), formats)
            if filename:
                print('Save Figure as '+filename+' ...')
                try:
                    dpi = int(data.settings['dpi'])
                except:
                    dpi = 'figure'
                self.figure.savefig(filename, dpi=dpi,
                                    transparent=data.settings['transparent']=='True',
                                    bbox_inches='tight')
                print('Saved!')
                
    def save_filters(self):
        if PRINT_FUNCTION_CALLS:
            print('save_filters')
        current_item = self.file_list.currentItem()
        if current_item:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Filters As...', '', '.npy')
            np.save(filename, current_item.data(QtCore.Qt.UserRole).filters)
            
    def load_filters(self):
        if PRINT_FUNCTION_CALLS:
            print('load_filters')
        current_item = self.file_list.currentItem()
        if current_item:
            filename, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'Open Filters File...', '', '*.npy')
            loaded_filters = np.load(filename[0], allow_pickle=True)
            data = current_item.data(QtCore.Qt.UserRole)
            data.filters = copy.deepcopy(loaded_filters)
            self.show_current_filters()
            data.apply_all_filters()
            self.update_plots()
            self.show_current_view_settings()
    
    def save_session(self, which='current'):
        if PRINT_FUNCTION_CALLS:
            print('save_session')
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'current':
                data = current_item.data(QtCore.Qt.UserRole)
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', os.path.splitext(data.filepath)[0].replace(':',''), '*.npy')
                items = [current_item]
            elif which == 'all':
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', '', '*.npy')
                items = [self.file_list.item(n) for n in range(self.file_list.count())]
            elif which == 'checked':
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', '', '*.npy')
                items = [self.file_list.item(n) for n in range(self.file_list.count()) 
                         if self.file_list.item(n).checkState() == 2]                             
            dictionary_list = []
            for item in items:
                data = item.data(QtCore.Qt.UserRole)
                item_dictionary = {'File Name': data.filename, 'File Path': data.filepath,
                                   'Settings': data.settings, 'Filters': data.filters, 
                                   'View Settings': data.view_settings, 'Raw Data': data.raw_data}
                if data.meta_data:
                    item_dictionary['Meta'] = data.meta_data
                dictionary_list.append(item_dictionary)
            np.save(filename, dictionary_list)
            print('Saved!')
    
    def open_files_from_folder(self):			
        if PRINT_FUNCTION_CALLS:
            print('open_files_from_folder')   
        self.file_list.itemChanged.disconnect(self.file_checked)
        rootdir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        filepaths = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                if file_extension == '.dat':
                    filepath = os.path.join(subdir, file)
                    filepaths.append((os.stat(filepath)[ST_CTIME],filepath))
        filepaths.sort(key=lambda tup: tup[0])
        for creation_time, filepath in filepaths:
            print('Open', filepath)
            self.add_file(filepath)
        last_item = self.file_list.item(self.file_list.count()-1)
        self.file_list.setCurrentItem(last_item)
        for item_index in range(self.file_list.count()-1):
            self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
        self.show_current_all()
        self.file_list.itemChanged.connect(self.file_checked)
        last_item.setCheckState(QtCore.Qt.Checked)
        
    def update_link_to_folder(self, new_folder=True):
        if new_folder:
            self.linked_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.linked_folder:
            self.window_title = 'Inspectra Gadget - Linked to folder '+self.linked_folder
            self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
            new_files = []
            for subdir, dirs, files in os.walk(self.linked_folder):
                for file in files:
                    filename, file_extension = os.path.splitext(file)
                    if file_extension == '.dat':
                        filepath = os.path.join(subdir, file)
                        if filepath not in self.linked_files:
                            new_files.append((os.stat(filepath)[ST_CTIME],filepath))
            if new_files:
                new_files.sort(key=lambda tup: tup[0])
                for new_file in new_files:
                    try:
                        print('Open', new_file[1])
                        self.add_file(new_file[1], do_load_data=False)
                        self.linked_files.append(new_file[1])
                    except:
                        print('Could not open', new_file[1])
                        #raise
                        continue
                if new_folder:
                    last_item = self.file_list.item(self.file_list.count()-1)
                    self.file_list.setCurrentItem(last_item)
                    last_item.setCheckState(QtCore.Qt.Checked)
                    
    def unlink_folder(self):
        if self.linked_folder:
            self.linked_folder = None
            self.window_title = 'Inspectra Gadget'
            self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
    
    def save_files_as(self, extension):
        if PRINT_FUNCTION_CALLS:
            print('save_files_as')
        save_directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        items = [self.file_list.item(n) for n in range(self.file_list.count())]
        for item in items:
            filename = os.path.join(save_directory, item.text().replace(':','') + extension)
            if not os.path.isfile(filename):
                try:            
                    self.file_list.itemChanged.disconnect(self.file_checked)
                    for item_index in range(self.file_list.count()):
                        self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
                    item.setCheckState(QtCore.Qt.Checked)
                    self.file_list.itemChanged.connect(self.file_checked)
                    self.update_plots()
                    data = item.data(QtCore.Qt.UserRole)
                    print('Saving figure as '+filename)
                    try:
                        dpi = int(data.settings['dpi'])
                    except:
                        dpi = 'figure'
                    self.figure.savefig(filename, dpi=dpi, 
                                        transparent=data.settings['transparent']=='True', bbox_inches='tight')
                except:
                    print('Could not save file '+filename+'...')
        
    def mouse_click_canvas(self, event):
        if PRINT_FUNCTION_CALLS:
            print('mouse_click_canvas')
        if self.navi_toolbar.mode == '':
            if event.inaxes:
                x, y = event.xdata, event.ydata
                items = self.file_list
                self.plot_in_focus = [items.item(index) for index in range(items.count()) 
                        if items.item(index).data(QtCore.Qt.UserRole).axes == event.inaxes]
                if self.plot_in_focus:
                    plot_data = self.plot_in_focus[0].data(QtCore.Qt.UserRole)
                    plot_data.selected_x, plot_data.selected_y = x, y
                    data = plot_data.processed_data
                    
                    if (event.button == 1 or event.button == 2) and len(plot_data.columns) == 3 and not plot_data.list_points:
                        index_x = np.argmin(np.abs(data[0][:,0]-x))
                        index_y = np.argmin(np.abs(data[1][0,:]-y))
                        plot_data.selected_indices = [int(index_x), int(index_y)]
                        if event.button == 1:
                            if PRINT_FUNCTION_CALLS:
                                print('leftmouseclickplot')
                            plot_data.orientation = 'horizontal'
                        else:
                            if PRINT_FUNCTION_CALLS:
                                print('middlemouseclickplot')
                            plot_data.orientation = 'vertical'
                        try:
                            plot_data.linecut_window
                        except AttributeError:
                            plot_data.linecut_window = LineCutWindow(plot_data)
                        plot_data.linecut_window.running = True
                        plot_data.update_linecut()
                        self.canvas.draw()
                        plot_data.linecut_window.activateWindow()
                        
                    elif event.button == 3:
                        if PRINT_FUNCTION_CALLS:
                            print('rightmouseclickplot')
                        menu = QtWidgets.QMenu(self)
                        if len(plot_data.columns) == 3:
                            index_x = np.argmin(np.abs(data[0][:,0]-x))
                            index_y = np.argmin(np.abs(data[1][0,:]-y))
                            z_value = data[2][index_x,index_y]
                            coordinates = 'x = %.4g, y = %.4g, z = %.4g (%d, %d)' % (x, y, z_value, index_x, index_y)
                            action = QtWidgets.QAction(coordinates, self)
                            action.setEnabled(False)
                            menu.addAction(action)
                            menu.addSeparator()
                        if plot_data.channels: # Triton 6 data format (meta.json file included)
                            channel_menu = menu.addMenu('Change channel to...')
                            for channel in plot_data.channels[len(plot_data.columns)-1:]:
                                action = QtWidgets.QAction(channel, self)
                                channel_menu.addAction(action)
                        if plot_data.dependent_parameters: # QCoDeS database format
                            channel_menu = menu.addMenu('Select channel...')
                            for par in plot_data.dependent_parameters:
                                action = QtWidgets.QAction(par, self)
                                channel_menu.addAction(action)
                        try:
                            plot_data.settings['rc-filter']
                        except KeyError:
                            pass
                        else:
                            if plot_data.settings['rc-filter'] != '' and plot_data.channels: # Triton 6 data format (meta.json file included)
                                if plot_data.rcfilter_correct == True:
                                    action = QtWidgets.QAction('Disable RC-filter correction...', self)                            
                                else:
                                    action = QtWidgets.QAction('Enable RC-filter correction...', self)
                                menu.addAction(action)
                        if len(plot_data.columns) == 3:
                            if plot_data.measurement_bounds:
                                if plot_data.draw_full_range:
                                    action = QtWidgets.QAction('Show measured range...', self)
                                else:
                                    action = QtWidgets.QAction('Show full range...', self)                                 
                                menu.addAction(action)
                                menu.addSeparator()
                            action = QtWidgets.QAction('Hide linecuts...', self)
                            menu.addAction(action)
                            action = QtWidgets.QAction('Refresh plot...', self)
                            menu.addAction(action)
                            if plot_data.cropping:
                                action = QtWidgets.QAction('Crop x to...', self)
                                menu.addAction(action)
                                action = QtWidgets.QAction('Crop y to...', self)
                                menu.addAction(action)
                                action = QtWidgets.QAction('Crop x and y to...', self)
                                plot_data.crop_to = [x, y]
                            else:
                                action = QtWidgets.QAction('Crop from...', self)
                                plot_data.crop_from = [x, y]
                                plot_data.cropping = False
                            menu.addAction(action)
                            if plot_data.drawing_diagonal_linecut:
                                action = QtWidgets.QAction('Diagonal linecut to...', self)
                                plot_data.linecut_index_to = [int(index_x), int(index_y)]
                                plot_data.linecut_to = [x, y]
                            else:
                                action = QtWidgets.QAction('Diagonal linecut from...', self)
                                plot_data.linecut_index_from = [int(index_x), int(index_y)]
                                plot_data.linecut_from = [x, y]
                            menu.addAction(action)
                            entries = ['Draw circular linecut...',
                                       'Plot vertical linecuts...',
                                       'Plot horizontal linecuts...',
                                       'FFT vertical...',
                                       'FFT horizontal...']
                        elif len(plot_data.columns) == 2:
                            entries = ['Refresh plot...']
                        for entry in entries:
                            action = QtWidgets.QAction(entry, self)
                            menu.addAction(action)
                        menu.triggered[QtWidgets.QAction].connect(self.popup_canvas)
                        menu.popup(QtGui.QCursor.pos())
                
                else: # if colorbar in focus
                    checked_items = [items.item(index) for index in range(items.count()) 
                                    if items.item(index).checkState() == 2]
                    self.cbar_in_focus = [checked_item for checked_item in checked_items
                                        if checked_item.data(QtCore.Qt.UserRole).cbar.ax == event.inaxes]
                    if self.cbar_in_focus:
                        data = self.cbar_in_focus[0].data(QtCore.Qt.UserRole)
                        min_map = data.view_settings['Minimum']
                        max_map = data.view_settings['Maximum']
                        new_value = min_map + y*(max_map-min_map)
                        if event.button == 1:
                            data.view_settings['Minimum'] = new_value
                            data.reset_midpoint()
                        elif event.button == 2:
                            data.view_settings['Midpoint'] = new_value
                        elif event.button == 3:
                            data.view_settings['Maximum'] = new_value
                            data.reset_midpoint()
                        data.apply_view_settings()
                        self.canvas.draw()
                        self.show_current_view_settings()
    
    def popup_canvas(self, signal):
        if PRINT_FUNCTION_CALLS:
            print(signal.text())
        plot_data = self.plot_in_focus[0].data(QtCore.Qt.UserRole)
        if signal.text() == 'Hide linecuts...':
            self.hide_linecuts(plot_data)
        elif signal.text() == 'Refresh plot...':
            plot_data.refresh_data(update_color_limits=False, refresh_unit_conversion=False)
            self.update_plots()
        elif (signal.text() == 'Plot horizontal linecuts...' or
            signal.text() == 'Plot vertical linecuts...'):
            if signal.text() == 'Plot horizontal linecuts...':
                plot_data.multi_orientation = 'horizontal'
            elif signal.text() == 'Plot vertical linecuts...':
                plot_data.multi_orientation = 'vertical'
            plot_data.multi_linecuts_window = LineCutWindow(plot_data, multiple=True)            
            plot_data.update_multiple_linecuts()
        elif signal.text() == 'Enable RC-filter correction...':
            plot_data.rcfilter_correct = True
            plot_data.refresh_data(update_color_limits=True, refresh_unit_conversion=False)
            self.update_plots()
        elif signal.text() == 'Disable RC-filter correction...':
            plot_data.rcfilter_correct = False
            plot_data.refresh_data(update_color_limits=True, refresh_unit_conversion=False)
            self.update_plots()
        elif signal.text() == 'Crop from...':
            plot_data.cropping = True
        elif (signal.text() == 'Crop x and y to...' or
              signal.text() == 'Crop x to...' or
              signal.text() == 'Crop y to...'):
            new_filters = []
            x1, y1 = plot_data.crop_from
            x2, y2 = plot_data.crop_to
            if (signal.text() == 'Crop x and y to...' or
                signal.text() == 'Crop x to...'):
                if x1 > x2:
                    x1, x2 = x2, x1
                new_filters.append({'Name': 'Crop X', 'Method': 'Absolute',
                                   'Setting 1': '%.4g' % x1, 
                                   'Setting 2': '%.4g' % x2, 'Checked': 2})
            if (signal.text() == 'Crop x and y to...' or
                signal.text() == 'Crop y to...'):                    
                if y1 > y2:
                    y1, y2 = y2, y1
                new_filters.append({'Name': 'Crop Y', 'Method': 'Absolute',
                                   'Setting 1': '%.4g' % y1, 
                                   'Setting 2': '%.4g' % y2, 'Checked': 2})
            for new_filter in new_filters:
                plot_data.apply_filter(new_filter)
                self.append_filter_to_table()
            self.show_current_view_settings()
            self.update_plots()
            self.canvas.draw()
            plot_data.cropping = False
        elif signal.text() == 'Diagonal linecut from...':
            if plot_data.list_points:
                self.hide_linecuts(plot_data)
            x1, y1 = plot_data.linecut_from
            plot_data.drawing_diagonal_linecut = True
            plot_data.list_points.append(DraggablePoint(plot_data, x1, y1))
            self.canvas.draw()
        elif signal.text() == 'Diagonal linecut to...':
            x1, y1 = plot_data.linecut_to
            plot_data.list_points.append(DraggablePoint(plot_data, x1, y1))
            plot_data.orientation = 'diagonal'
            try:
                plot_data.linecut_window
            except AttributeError:
                plot_data.linecut_window = LineCutWindow(plot_data)
            plot_data.linecut_window.running = True
            plot_data.update_linecut()
            self.canvas.draw()
            plot_data.drawing_diagonal_linecut = False
            plot_data.linecut_window.activateWindow()
        elif signal.text() == 'Draw circular linecut...':
            if plot_data.list_points:
                self.hide_linecuts(plot_data)
            left, right = plot_data.axes.get_xlim() 
            bottom, top = plot_data.axes.get_ylim()                
            x0, y0 = plot_data.selected_x, plot_data.selected_y
            plot_data.xr, plot_data.yr = 0.1*(right-left), 0.1*(top-bottom)
            plot_data.list_points.append(DraggablePoint(plot_data, x0, y0, draw_line=False, draw_circle=True))
            plot_data.list_points.append(DraggablePoint(plot_data, x0+plot_data.xr, y0, draw_line=False, draw_circle=True))
            plot_data.list_points.append(DraggablePoint(plot_data, x0, y0+plot_data.yr, draw_line=False, draw_circle=True))
            plot_data.orientation = 'circular'
            try:
                plot_data.linecut_window
            except AttributeError:
                plot_data.linecut_window = LineCutWindow(plot_data)
            plot_data.linecut_window.running = True
            plot_data.update_linecut()
            self.canvas.draw()
            plot_data.linecut_window.activateWindow()  
        elif signal.text() == 'Show full range...':
            plot_data.draw_full_range = True
            plot_data.axes.set_xlim(left=min(plot_data.measurement_bounds), 
                                    right=max(plot_data.measurement_bounds))
            self.canvas.draw()
        elif signal.text() == 'Show measured range...':
            plot_data.draw_full_range = False
            plot_data.axes.set_xlim(left=np.amin(plot_data.processed_data[0]), 
                                    right=np.amax(plot_data.processed_data[0]))
            self.canvas.draw()
        elif signal.text() == 'FFT vertical...':
            plot_data.fft_orientation = 'vertical'
            plot_data.open_fft_window()
        elif signal.text() == 'FFT horizontal...':
            plot_data.fft_orientation = 'horizontal'
            plot_data.open_fft_window()
        elif signal.text() in plot_data.channels:
            channel_index = plot_data.channels.index(signal.text())
            plot_data.settings['columns'] = plot_data.settings['columns'][:-1]+str(channel_index)
            plot_data.columns[-1] = channel_index
            if len(plot_data.columns) == 2:
                plot_data.settings['ylabel'] = signal.text()
            else:                
                plot_data.settings['clabel'] = signal.text()
            plot_data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
            self.update_plots()
            self.show_current_all()
        elif qcodes_imported and signal.text() in plot_data.dependent_parameters:
            plot_data.index_dependent_parameter = plot_data.dependent_parameters.index(signal.text())
            plot_data.refresh_data(update_color_limits=True)
            self.update_plots()
            self.show_current_all()
      
    def hide_linecuts(self, plot_data):
        plot_data.linecut_window.running = False         
        for line in reversed(plot_data.axes.get_lines()):
            line.remove()
            del line
        for patch in reversed(plot_data.axes.patches):
            patch.remove()
            del patch
        plot_data.list_points = []
        self.canvas.draw()
        
    def copy_canvas_to_clipboard(self):
        file_list = self.file_list
        checked_items = [file_list.item(index) for index in range(file_list.count()) 
                        if file_list.item(index).checkState() == 2]
        for item in checked_items:
            plot_data = item.data(QtCore.Qt.UserRole)
            plot_data.cursor.horizOn = False
            plot_data.cursor.vertOn = False            
        self.canvas.draw()            
        buf = io.BytesIO()
        try:
            dpi = int(plot_data.settings['dpi'])
        except:
            dpi = 'figure'
        self.figure.savefig(buf, dpi=dpi, 
                            transparent=plot_data.settings['transparent']=='True', bbox_inches='tight')
        QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
        buf.close()
        for item in checked_items:
            plot_data = item.data(QtCore.Qt.UserRole)
            plot_data.cursor.horizOn = True
            plot_data.cursor.vertOn = True                       
        self.canvas.draw()
            
    def mouse_scroll_canvas(self, event):
        if PRINT_FUNCTION_CALLS:
            print('mouse_scroll_canvas')
        if event.inaxes:
            y = event.ydata
            items = self.file_list
            self.plot_in_focus = [items.item(index) for index in range(items.count()) 
                                  if items.item(index).data(QtCore.Qt.UserRole).axes == event.inaxes]
            if self.plot_in_focus:
                data = self.plot_in_focus[0].data(QtCore.Qt.UserRole)
                if len(data.columns) == 3:
                    try:
                        data.linecut_window
                    except AttributeError:
                        pass
                    else:
                        data_shape = data.processed_data[0].shape
                        if data.orientation == 'horizontal':
                            new_index = data.selected_indices[1]+int(event.step)
                            if new_index >= 0 and new_index < data_shape[1]:
                                data.selected_indices[1] = new_index
                        elif data.orientation == 'vertical':
                            new_index = data.selected_indices[0]+int(event.step)
                            if new_index >= 0 and new_index < data_shape[0]:
                                data.selected_indices[0] = new_index
                        data.update_linecut()
                        self.canvas.draw()
            else:
                checked_items = [items.item(index) for index in range(items.count()) 
                        if items.item(index).checkState() == 2]
                self.cbar_in_focus = [checked_item for checked_item in checked_items
                                      if checked_item.data(QtCore.Qt.UserRole).cbar.ax == event.inaxes]
                if self.cbar_in_focus:
                    data = self.cbar_in_focus[0].data(QtCore.Qt.UserRole)
                    min_map = data.view_settings['Minimum']
                    max_map = data.view_settings['Maximum']
                    range_map = max_map-min_map
                    if y > min_map+0.5*range_map:    
                        new_max = max_map + event.step*range_map*0.02
                        data.view_settings['Maximum'] = new_max
                    else:
                        new_min = min_map + event.step*range_map*0.02
                        data.view_settings['Minimum'] = new_min
                    data.reset_midpoint()
                    data.apply_view_settings()
                    self.canvas.draw()
                    self.show_current_view_settings()
        else:
            width, height = self.canvas.get_width_height()
            speed = 0.04
            if event.x < 0.25*width:
                self.figure.subplots_adjust(left=(1+speed*event.step)*self.figure.subplotpars.left)
            elif event.x > 0.75*width:
                self.figure.subplots_adjust(right=(1+speed*0.5*event.step)*self.figure.subplotpars.right)
            else:
                if event.y < 0.25*height:
                    self.figure.subplots_adjust(bottom=(1+speed*event.step)*self.figure.subplotpars.bottom)
                elif event.y > 0.75*height:
                    self.figure.subplots_adjust(top=(1+speed*0.5*event.step)*self.figure.subplotpars.top)
            self.canvas.draw()
            
    def keyPressEvent(self, event):
        if PRINT_FUNCTION_CALLS:
            print('keyPressEvent', event.key(), event.modifiers())  
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            self.copy_canvas_to_clipboard()
        elif event.key() == QtCore.Qt.Key_T and event.modifiers() == QtCore.Qt.ControlModifier:
            if not self.action_refresh_stop.isEnabled():
                self.start_auto_refresh(1)
                print('Start live tracking...')
            else:
                self.stop_auto_refresh()
                print('Stop live tracking...')
            
               
    def merge_files(self, raw_data=True):
        if PRINT_FUNCTION_CALLS:
            print('merge_files')
        try:
            file_list = self.file_list
            checked_items = [file_list.item(index) for index in range(file_list.count()) 
                            if file_list.item(index).checkState() == 2]
            if len(checked_items) > 1:
                filepaths = [item.data(QtCore.Qt.UserRole).filepath for item in checked_items]
                time_stamps = [item.data(QtCore.Qt.UserRole).filename.split('_')[0] for item in checked_items]
                device_name = checked_items[0].data(QtCore.Qt.UserRole).filename.split('_')[1]
                measurement_name = checked_items[0].data(QtCore.Qt.UserRole).filename.split('_')[2]
                output_name = '_'.join(time_stamps)+'_'+device_name+'_'+measurement_name+'_merged.dat'
                output_dir = os.path.dirname(checked_items[0].data(QtCore.Qt.UserRole).filepath)
                output_filepath = output_dir+'/'+output_name
                if raw_data:
                    with open(output_filepath,'w') as out_file:
                        for filepath in filepaths:
                            with open(filepath) as in_file:
                                for line in in_file:
                                    out_file.write(line)
                else:
                    data = [item.data(QtCore.Qt.UserRole).processed_data for item in checked_items]
                    min_x = [np.amin(data[n][0]) for n,_ in enumerate(checked_items)]
                    max_x = [np.amax(data[n][0]) for n,_ in enumerate(checked_items)]
                    min_y = [np.amin(data[n][1]) for n,_ in enumerate(checked_items)]
                    max_y = [np.amax(data[n][1]) for n,_ in enumerate(checked_items)]
                    dmin_x = [np.abs(data[n][0][1,0]-data[n][0][0,0]) for n,_ in enumerate(checked_items)]
                    dmin_y = [np.abs(data[n][1][0,1]-data[n][1][0,0]) for n,_ in enumerate(checked_items)]
                    grid_x = np.arange(np.min(min_x),np.max(max_x),np.min(dmin_x))
                    grid_y = np.arange(np.min(min_y),np.max(max_y),np.min(dmin_y))
                    y_g, x_g = np.meshgrid(grid_y, grid_x)
                    points_x = np.concatenate([data[n][0].flatten() for n,_ in enumerate(checked_items)])
                    points_y = np.concatenate([data[n][1].flatten() for n,_ in enumerate(checked_items)])
                    values_z = np.concatenate([data[n][2].flatten() for n,_ in enumerate(checked_items)])
                    g_data = griddata(np.stack([points_x,points_y],-1), values_z, (x_g,y_g), method='nearest')
                    data0 = checked_items[0].data(QtCore.Qt.UserRole)
                    view_settings = {
                            'Minimum': np.min(g_data), 'Maximum': np.max(g_data), 
                            'Midpoint': 0.5*(np.min(g_data)+np.max(g_data)),
                            'Color Map': 'magma', 'Color Map Type': 'Uniform',
                            'Norm': MidpointNormalize(vmin=np.min(g_data), 
                                                      vmax=np.max(g_data), 
                                                      midpoint= 0.5*(np.min(g_data)+np.max(g_data))),
                            'Locked': 0, 'MidLock': 0, 'Reverse': 2}
                    dictionary_list = [{'File Name': output_name, 'Settings': data0.default_settings,
                                      'Filters': [], 'View Settings': view_settings,
                                      'Raw Data': [x_g,y_g,g_data]}]
                    np.save(output_filepath, dictionary_list)
                print('Merged files into file '+output_filepath)
        except:
            print('Cannot merge these files...')
            
            
    def apply_preset(self, preset_number):
        checked_items = [self.file_list.item(index) for index in range(self.file_list.count()) 
                         if self.file_list.item(index).checkState() == 2]
        if checked_items:
            for item in checked_items:
                data = item.data(QtCore.Qt.UserRole)
                for preset_item in PRESETS[preset_number].items():
                    if preset_item[0] in data.settings.keys():
                        data.settings[preset_item[0]] = preset_item[1]
                    elif preset_item[0] == 'canvas_bounds':
                        b = preset_item[1] # (left, bottom, right, top)
                        data.figure.subplots_adjust(b[0], b[1], b[2], b[3])
                    elif preset_item[0] == 'show_meta_settings':
                        data.show_settings = preset_item[1]
            self.update_plots()
                #data.apply_plot_settings()
                #self.show_current_plot_settings()
                #self.canvas.draw()
    
class Data:
    def __init__(self, filepath, canvas, data=None, do_load_data=True):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.canvas = canvas
        self.default_settings = DEFAULT_PLOT_SETTINGS
        self.default_filters = []
        self.filters = copy.deepcopy(self.default_filters)
        self.settings = self.default_settings.copy()
        self.old_settings = self.default_settings.copy()
        self.columns = [int(s) for s in self.default_settings['columns'].split(',')]
        self.duplicate = False
        self.measurement_bounds = None
        self.draw_full_range = False
        self.channels = []
        self.processed_data = None
        self.meta_data = None
        self.meta_data_name = None
        self.rcfilter_correct = False
        self.four_terminal = False
        self.dataset = None
        self.npy_file = False
        self.qcodes_file = False
        self.selected_indices = [0, 0]
        self.figure = None
        self.axes = None
        self.image = None
        self.cbar = None
        self.orientation = None
        self.multi_orientation = None
        self.cropping = False
        self.drawing_diagonal_linecut = False
        self.drawing_circular_linecut = False
        self.list_points = []
        self.index_dependent_parameter = DEFAULT_INDEX_DEPENDENT_PARAMETER
        self.dependent_parameters = None
        self.remaining_time = None
        self.remaining_time_string = ''
        self.last_modified_time = None
        try: # on Windows
            self.creation_time = os.path.getctime(filepath)
        except:
            try: # on Mac
                self.creation_time = os.stat(filepath).st_birthtime
            except:
                self.creation_time = None

     
        if isinstance(data, dict): # for .npy files, saved using the "Save Session" menu
            self.npy_file = True
            if 'File Path' in data:
                self.filepath = data['File Path']
            self.columns = [int(s) for s in data['Settings']['columns'].split(',')]
            self.raw_data = data['Raw Data']
            for setting, value in data['Settings'].items():
                if setting in self.settings:
                    self.settings[setting] = value
            self.filters = data['Filters']
            self.view_settings = data['View Settings']
            if 'Meta' in data:
                self.interpret_meta_file(data['Meta'], change_labels=False)
                self.rcfilter_correct = DEFAULT_VALUE_RCFILTER_CORRECT
                self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=False)
            else:
                self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=False)
                
        elif qcodes_imported and isinstance(data, qc.dataset.data_set.DataSet): # for QCoDeS databases
            self.qcodes_file = True         
            self.dataset = data
            self.view_settings = {
                    'Minimum': 0, 'Maximum': 0, 'Midpoint': 0,
                    'Color Map': DEFAULT_COLORMAP, 'Color Map Type': 'Uniform',
                    'Norm': MidpointNormalize(vmin=0, vmax=1, midpoint=0.5),
                    'Locked': 0, 'MidLock': 0, 'Reverse': DEFAULT_REVERSE_COLORMAP} 
                       
        else:
            if do_load_data:
                self.load_data()
            meta_file_exists = os.path.isfile(os.path.dirname(self.filepath)+'/meta.json')
            if self.filename == 'data.dat' and meta_file_exists: # Copenhagen meta data file
                self.interpret_meta_file()
                if do_load_data:
                    self.processed_to_raw()
                    self.rcfilter_correct = DEFAULT_VALUE_RCFILTER_CORRECT
                    self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=True)
                else:
                    self.meta_unit_conversion()
            else:
                if do_load_data:
                    self.processed_to_raw()
                    self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=False)
            if do_load_data:
                min_map = np.min(self.processed_data[-1])
                max_map = np.max(self.processed_data[-1])
                mid_map = 0.5*(min_map+max_map)
                self.view_settings = {
                        'Minimum': min_map, 'Maximum': max_map, 'Midpoint': mid_map,
                        'Color Map': DEFAULT_COLORMAP, 'Color Map Type': 'Uniform',
                        'Norm': MidpointNormalize(vmin=min_map, vmax=max_map, midpoint=mid_map),
                        'Locked': 0, 'MidLock': 0, 'Reverse': DEFAULT_REVERSE_COLORMAP}
            else:
                self.view_settings = {
                    'Minimum': 0, 'Maximum': 0, 'Midpoint': 0,
                    'Color Map': DEFAULT_COLORMAP, 'Color Map Type': 'Uniform',
                    'Norm': MidpointNormalize(vmin=0, vmax=1, midpoint=0.5),
                    'Locked': 0, 'MidLock': 0, 'Reverse': DEFAULT_REVERSE_COLORMAP} 

        
    def load_data(self):
        if PRINT_FUNCTION_CALLS:
            print('load_data')
        if not self.qcodes_file:
            column_data = np.genfromtxt(self.filepath, delimiter=self.settings['delimiter'])
            self.measured_data_points = column_data.shape[0]
        else:
            self.dependent_parameters = [p.name for p in self.dataset.dependent_parameters]
            dependent_par = self.dependent_parameters[self.index_dependent_parameter]
            data_dict = self.dataset.get_parameter_data(dependent_par)[dependent_par]
            pars = list(data_dict.keys())
            if len(pars) == 2: # file is 2D
                x_par, y_par = pars[1], pars[0]
                column_data = np.column_stack((data_dict[x_par], data_dict[y_par]))
            else: # file is 3D
                x_par, y_par, z_par = pars[1], pars[2], pars[0]
                column_data = np.column_stack((data_dict[x_par], data_dict[y_par], data_dict[z_par]))
                self.settings['clabel'] =  '{} ({})'.format(self.dataset.paramspecs[z_par].label, self.dataset.paramspecs[z_par].unit)
            self.settings['xlabel'] =  '{} ({})'.format(self.dataset.paramspecs[x_par].label, self.dataset.paramspecs[x_par].unit)
            self.settings['ylabel'] =  '{} ({})'.format(self.dataset.paramspecs[y_par].label, self.dataset.paramspecs[y_par].unit)
        unique_values, indices = np.unique(column_data[:,self.columns[0]], return_index=True)
        
        if len(indices) > 1: # if first column has more than one uniuqe value
            # shape of data
            l0 = np.sort(indices)[1]
            l1 = len(indices)
            
            # ignore last column if unfinished
            if len(column_data[np.sort(indices)[-1]::,0]) < l0:
                l1 = l1-1
                
            if COMBINATION_GATE_SWEEPS_POSSIBLE: # TODO fix this for general case   
            # check if subsequent column is also repeated (e.g. for combination gate sweeps)
                if column_data[1,self.columns[1]] == column_data[0,self.columns[1]]:
                    self.columns = [self.columns[0]] + [i+1 for i in self.columns[1:]]
            
            # check if file is 2D or 3D
            if (column_data[1,self.columns[0]] != column_data[0,self.columns[0]]) or len(self.columns) == 2: # if 2D
                self.raw_data = [column_data[:,x] for x in range(column_data.shape[1])]            
                if len(self.columns) == 3:
                    self.columns = self.columns[:-1]
            else: # if 3D
                if indices[1] > indices[0]: # if first column is sorted from low to high --> reshape normally
                    self.raw_data = [np.reshape(column_data[:l0*l1,x], (l1,l0)) for x in range(column_data.shape[1])]
                else: # if first column is sorted from high to low --> flip and then reshape normally
                    self.raw_data = [np.reshape(column_data[l0*l1-1::-1,x], (l1,l0)) for x in range(column_data.shape[1])]
                    
                if self.raw_data[1][0,0] > self.raw_data[1][0,1]: # flip is second column is sorted from high to low
                    self.raw_data = [np.fliplr(self.raw_data[x]) for x in range(column_data.shape[1])]
                    
            self.settings['columns'] = ','.join([str(i) for i in self.columns])
        else: # if first column has only one unique value --> ignore data; set to zero
            self.raw_data = [np.array([[0,0],[0,0]]) for x in range(column_data.shape[1])]

    def interpret_meta_file(self, meta_data=None, change_labels=True):
        if PRINT_FUNCTION_CALLS:
            print('interpret_meta_file')
        if meta_data:
            self.meta_data = meta_data
        else:
            self.meta_file = os.path.dirname(self.filepath)+'/meta.json'
            with open(self.meta_file) as f:
                self.meta_data = json.load(f)            
        self.channels = [channel['name'] for channel in self.meta_data['columns']]
        if change_labels:
            self.settings['xlabel'] = self.channels[self.columns[0]]
            self.settings['ylabel'] = self.channels[self.columns[1]]
            if len(self.columns) == 3:
                self.settings['clabel'] = self.channels[self.columns[2]]          
        self.meta_data_name = (os.path.basename(os.path.dirname(self.filepath)) + ' ' +
                               self.meta_data['timestamp'].split(' ')[1] + ' ' + self.meta_data['name'])
        if 'source' in self.channels and 'dc_curr' in self.channels:
            self.settings['rc-filter'] = str(DEFAULT_RC_FILTER)
        try:
            new_index = self.channels.index(DEFAULT_CHANNEL)
            self.columns[-1] = new_index
            self.settings['columns'] = self.settings['columns'][:-1]+str(new_index)
        except:
            print('Default channel',DEFAULT_CHANNEL,'not found...')
        
        if DEFAULT_SHOW_METADATANAME:
            self.settings['title'] = '<metadataname>'
        
        self.settings_string = ''
        channels_to_show = CHANNELS_TO_SHOW
        meta_channels = self.meta_data['setup']['channels']
        for instrument in self.meta_data['register']['instruments']:
            if instrument['name'] == 'dac':
                dac = instrument
                break
        for channel in channels_to_show:
            try:
                if channel in meta_channels:
                    for register_channel in self.meta_data['register']['channels']:
                        if register_channel['name'] == channel: 
                            if register_channel['instrument'] == 'dac':  
                                value = dac['current_values'][register_channel['channel_id']]
                                self.settings_string += channel+': '+'{:.3g}'.format(value)+'\n'
                                break
                            elif register_channel['instrument'] == 'smua':
                                for instrument in self.meta_data['register']['instruments']:
                                    if instrument['name'] == 'smua' and 'current_values' in instrument.keys():
                                        value = instrument['current_values']['v']
                                        self.settings_string += channel+': '+'{:.3g}'.format(value)+'\n'
                                        break
                            elif register_channel['instrument'] == 'smub':
                                for instrument in self.meta_data['register']['instruments']:
                                    if instrument['name'] == 'smub' and 'current_values' in instrument.keys():
                                        value = instrument['current_values']['v']
                                        self.settings_string += channel+': '+'{:.3g}'.format(value)+'\n'
                                        break
                    else:
                        if channel == 'Bx' or channel == 'Bz':
                            for instrument in self.meta_data['register']['instruments']:
                                if instrument['name'] == 'magnet':
                                    value = instrument['field'][channel[1:]]
                                    self.settings_string += channel+': '+'{:.3g}'.format(value)+'\n'
                                    break
                elif channel == 'T':
                    for instrument in self.meta_data['register']['instruments']:
                        if instrument['name'] == 'triton':
                            value = instrument['temperatures']['MC']*1000
                            self.settings_string += channel+': '+'{:.3g}'.format(value)+'\n'
                            break
            except:
                print('Could not add channel',channel,'...')
         
    def correct_for_rcfilters(self):
        if PRINT_FUNCTION_CALLS:
            print('correct_for_rcfilters')
            
        if 'lockin_bias/X' in self.channels and 'lockin_curr/X' in self.channels: # assume 4-terminal, current-biased
            self.four_terminal = True
        else:
            self.four_terminal = False
            
        if 'source' in self.channels and 'dc_curr' in self.channels:
            source_index = self.channels.index('source')
            if source_index in self.columns:
                if PRINT_FUNCTION_CALLS:
                    print('Correcting source bias voltage for total rc-filter resistance of',self.settings['rc-filter'],'Ohm...')
                source_divider = self.meta_data['setup']['meta']['source_divider']
                curr_index = self.channels.index('dc_curr')                    
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                source_data = self.raw_data[source_index] / source_divider
                curr_data = self.raw_data[curr_index] / curr_amp
                source_corrected = (source_data - float(self.settings['rc-filter'])*curr_data)*source_divider
                self.processed_data[self.columns.index(source_index)] = np.copy(source_corrected)
        
        if 'lockin_curr/X' in self.channels:
            lockin_index = self.channels.index('lockin_curr/X')
            if lockin_index in self.columns:
                if PRINT_FUNCTION_CALLS:
                    print('Correcting lockin_curr/X for total rc-filter resistance of',self.settings['rc-filter'],'Ohm...')
                for instrument in self.meta_data['register']['instruments']:
                    if instrument['name'] == 'lockin_curr':
                        break
                sine_amplitude = float(instrument['config']['SLVL'])
                lockin_divider = self.meta_data['setup']['meta']['lock_sig_divider']
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                conversion = curr_amp*sine_amplitude/lockin_divider # divide by this to convert lockin-signal to Siemens (1/Ohm)
                lockin_data = self.raw_data[lockin_index]/conversion # in Siemens
                series_resistance = float(self.settings['rc-filter']) # in Ohm
                lockin_corrected = lockin_data/(1.-series_resistance*lockin_data)*conversion
                self.processed_data[self.columns.index(lockin_index)] = np.copy(lockin_corrected)                
    
    def meta_unit_conversion(self):
        if PRINT_FUNCTION_CALLS:
            print('meta_unit_conversion')
        
        # Get bounds to be able to show full range during measurement
        try: 
            bound_from = self.meta_data['job']['from']
            bound_to = self.meta_data['job']['to']
        except KeyError:
            bound_from = self.meta_data['job']['job']['from']
            bound_to = self.meta_data['job']['job']['to']
        
        # Get total number of (to be) measured data points
        self.total_data_points = 1 
        if 'points' in self.meta_data['job'].keys():
            self.total_data_points *= self.meta_data['job']['points']
        if 'points' in self.meta_data['job']['job'].keys():
            self.total_data_points *= self.meta_data['job']['job']['points']
            
        # If x-parameter is combination of channels -> take bounds that belongs to the displayed channel    
        if isinstance(bound_from, list):
            if self.channels[self.columns[0]] in self.meta_data['job']['chans']:
                bound_from = bound_from[self.meta_data['job']['chans'].index(self.channels[self.columns[0]])]
                bound_to = bound_to[self.meta_data['job']['chans'].index(self.channels[self.columns[0]])]
        
        # Unit conversion of different channels (using data from meta.json file)
        if 'source' in self.channels:
            source_index = self.channels.index('source')
            if source_index in self.columns:
                source_divider = self.meta_data['setup']['meta']['source_divider']
                SOURCE_UNIT = 1e-3 # Convert V to mV  
                if PRINT_FUNCTION_CALLS:
                    print('Converting source units to millivolts...')
                if source_divider*SOURCE_UNIT != 1.0:
                    divide = '%.3g' % (source_divider*SOURCE_UNIT)
                    axis = ['X','Y','Z'][self.columns.index(source_index)] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(source_index) == 0:
                    self.settings['xlabel'] = 'Bias voltage (mV)'
                elif self.columns.index(source_index) == 1:
                    self.settings['ylabel'] = 'Bias voltage (mV)'
                if source_index == 0:
                    self.measurement_bounds = [bound_from/(source_divider*SOURCE_UNIT),
                                               bound_to/(source_divider*SOURCE_UNIT)]
        
        # Gate voltages are combinations of a coarse channel and a fine channel
        fine_gates = [channel for channel in self.channels if channel[0] == 'g' and channel[-1] == 'f']
        for gate in fine_gates:
            gate_index = self.channels.index(gate)
            if gate_index in self.columns:                              
                if PRINT_FUNCTION_CALLS:
                    print('Converting '+gate+' units to volts...')
                DAC_FINE_DIVIDER = 200
                DAC_FINE_OFFSET = 0.05
                divide = '%.3g' % DAC_FINE_DIVIDER
                gate_axis = ['X','Y','Z'][self.columns.index(gate_index)] 
                self.filters.append({'Name': 'Divide', 'Method': gate_axis, 
                                     'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if DAC_FINE_OFFSET != 0.0:
                    fine_offset = '%.3g' % DAC_FINE_OFFSET
                    self.filters.append({'Name': 'Offset', 'Method': gate_axis, 
                                         'Setting 1': fine_offset, 'Setting 2': '', 'Checked': 2})                    
                if PRINT_FUNCTION_CALLS:
                    print('Adding coarse gate voltage...')
                coarse_gate = gate[:-1]
                for instrument in self.meta_data['register']['instruments']:
                    if instrument['name'] == 'dac':
                        dac = instrument
                        break
                for dac_channel in self.meta_data['register']['channels']:
                    if dac_channel['name'] == coarse_gate:
                        dac_channel_coarse_gate = dac_channel['channel_id']
                        break
                coarse_offset_dac = dac['current_values'][dac_channel_coarse_gate]
                coarse_offset = '%.3g' % coarse_offset_dac
                self.filters.append({'Name': 'Offset', 'Method': gate_axis, 
                                     'Setting 1': coarse_offset, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(gate_index) == 0:
                    self.settings['xlabel'] = 'Gate voltage '+gate[1:]+' (V)'
                elif self.columns.index(gate_index) == 1:
                    self.settings['ylabel'] = 'Gate voltage '+gate[1:]+' (V)'
                if gate_index == 0:   
                    self.measurement_bounds = [bound_from/DAC_FINE_DIVIDER+DAC_FINE_OFFSET+coarse_offset_dac,
                                               bound_to/DAC_FINE_DIVIDER+DAC_FINE_OFFSET+coarse_offset_dac]
        
        coarse_gates = [channel for channel in self.channels if channel[0] == 'g' and len(channel) == 2]
        for gate in coarse_gates:
            gate_index = self.channels.index(gate)
            if gate_index in self.columns:
                if self.columns.index(gate_index) == 0:
                    self.settings['xlabel'] = 'Gate voltage '+gate[1]+' (V)'
                elif self.columns.index(gate_index) == 1:
                    self.settings['ylabel'] = 'Gate voltage '+gate[1]+' (V)'  
                if gate_index == 0:   
                    self.measurement_bounds = [bound_from, bound_to]
            
        if 'dc_curr' in self.channels:
            curr_index = self.channels.index('dc_curr')
            if curr_index in self.columns:
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                if PRINT_FUNCTION_CALLS:
                    print('Converting dc_curr units to nano-amperes...')
                CURR_UNIT = 1e-9 # Ampere to nano-ampere
                if curr_amp*CURR_UNIT != 1.0:
                    divide = '%.3g' % (curr_amp*CURR_UNIT)
                    axis = ['X','Y','Z'][self.columns.index(curr_index)] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(curr_index) == 0:
                    self.settings['xlabel'] = '$I$ (nA)'
                elif self.columns.index(curr_index) == 1:
                    self.settings['ylabel'] = '$I$ (nA)'
                elif self.columns.index(curr_index) == 2:
                    self.settings['clabel'] = '$I$ (nA)'
        
        if 'lockin_curr/X' in self.channels:
            lockin_index = self.channels.index('lockin_curr/X')
            if lockin_index in self.columns:
                for instrument in self.meta_data['register']['instruments']:
                    if instrument['name'] == 'lockin_curr':
                        break
                sine_amplitude = float(instrument['config']['SLVL'])
                lockin_divider = self.meta_data['setup']['meta']['lock_sig_divider']
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                LOCKIN_UNIT = 1e-6 # Siemens to microsiemens
                conversion_factor = curr_amp*sine_amplitude/lockin_divider*LOCKIN_UNIT
                if conversion_factor != 1.0:
                    if PRINT_FUNCTION_CALLS:
                        print('Correcting lockin_curr/X units to microsiemens...')
                    divide = '%.3g' % conversion_factor
                    axis = ['X','Y','Z'][self.columns.index(lockin_index)] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                    if PRINT_FUNCTION_CALLS:
                        print('Correcting lockin_curr/X units from microsiemens to e^2/h...')
                    multiply_e2h = '%.6g' % 0.0258128
                    axis = ['X','Y','Z'][self.columns.index(lockin_index)] 
                    self.filters.append({'Name': 'Multiply', 'Method': axis, 
                                         'Setting 1': multiply_e2h, 'Setting 2': '', 'Checked': 2})
                
                if self.columns.index(lockin_index) == 1:
                    if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                        self.settings['ylabel'] = 'd$I$/d$V$ ($e^2/h$)'
                    else:
                        self.settings['ylabel'] = 'd$I$/d$V$ (μS)'
                elif self.columns.index(lockin_index) == 2:
                    if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                        self.settings['clabel'] = 'd$I$/d$V$ ($e^2/h$)'
                    else:
                        self.settings['clabel'] = 'd$I$/d$V$ (μS)'

        if 'lockin_bias/X' in self.channels and 'lockin_curr/X' in self.channels: # assume 4-terminal, current-biased
            lockin_bias_index = self.channels.index('lockin_bias/X')
            if lockin_bias_index in self.columns:
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                bias_amp = self.meta_data['setup']['meta']['bias_amp']
                if PRINT_FUNCTION_CALLS:
                    print('Converting lockin_bias/X to 4-terminal dV/dI in units of Ohm...')
                conversion_factor = curr_amp/bias_amp
                if conversion_factor != 1.0:
                    divide = '%.3g' % conversion_factor
                    axis = ['X','Y','Z'][self.columns.index(lockin_bias_index)] 
                    self.filters.append({'Name': 'Multiply', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})                
                if self.columns.index(lockin_bias_index) == 1:
                    self.settings['ylabel'] = 'd$V$/d$I$ ($\Omega$)'
                elif self.columns.index(lockin_bias_index) == 2:
                    self.settings['clabel'] = 'd$V$/d$I$ ($\Omega$)'
                self.four_terminal = True
        else:
            self.four_terminal = False
        
        if 'bg' in self.channels:
            bg_index = self.channels.index('bg')
            if bg_index in self.columns:
                if self.columns.index(bg_index) == 0:
                    self.settings['xlabel'] = 'Backgate voltage (V)'
                elif self.columns.index(bg_index) == 1:
                    self.settings['ylabel'] = 'Backgate voltage (V)'
                if bg_index == 0:   
                    self.measurement_bounds = [bound_from, bound_to]

        if 'sg' in self.channels:
            sg_index = self.channels.index('sg')
            if sg_index in self.columns:
                if self.columns.index(sg_index) == 0:
                    self.settings['xlabel'] = 'Side gates voltage (V)'
                elif self.columns.index(sg_index) == 1:
                    self.settings['ylabel'] = 'Side gates voltage (V)'
                if sg_index == 0:   
                    self.measurement_bounds = [bound_from, bound_to]
        
        if 'Bx' in self.channels:
            if self.channels.index('Bx') == 0:  
                self.measurement_bounds = [bound_from, bound_to]
        
        if 'Bz' in self.channels:
            if self.channels.index('Bz') == 0:  
                self.measurement_bounds = [bound_from, bound_to]
                
        if 'Br' in self.channels:
            if self.channels.index('Br') == 0:  
                self.measurement_bounds = [bound_from, bound_to]
               
    def processed_to_raw(self):
        if PRINT_FUNCTION_CALLS:
            print('processed_to_raw')
        self.processed_data = [np.copy(self.raw_data[i]) for i in self.columns]
    
    def refresh_data(self, update_color_limits=False, refresh_unit_conversion=False):
        if PRINT_FUNCTION_CALLS:
            print('refresh_data')
        try:
            if not self.npy_file: # if data is not directly from a npy file
                self.load_data()           
            self.apply_all_filters(update_color_limits, refresh_unit_conversion)
            
            if self.measured_data_points and self.total_data_points and self.creation_time:
                self.last_modified_time = os.path.getmtime(self.filepath)
                spent_time = self.last_modified_time - self.creation_time
                self.progress_fraction = self.measured_data_points/self.total_data_points             
                total_time = spent_time / self.progress_fraction
                remaining_seconds = total_time - spent_time
                self.remaining_time = timedelta(seconds=remaining_seconds)
                self.remaining_time_string = '    Progress last measurement:  {:d}% Completed  -  Remaining time:'.format(int(self.progress_fraction*100)) + \
                                             ' '+str(self.remaining_time).split('.')[0] + \
                                             '  -  Finishes at: '+str(datetime.now()+self.remaining_time).split('.')[0]
        except:
            print('Could not refresh...')
            if SHOW_ERRORS:
                raise
    
    def add_plot(self):
        if PRINT_FUNCTION_CALLS:
            print('add_plot')
        data = self.processed_data
        cmap_str = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap_str = cmap_str+'_r'
        cmap = cm.get_cmap(cmap_str, lut=int(self.settings['lut']))
        cmap.set_bad(self.settings['maskcolor'])
        self.image = self.axes.pcolormesh(data[0], data[1], data[2], shading='auto', 
                                          norm=self.view_settings['Norm'], cmap=cmap,
                                          rasterized=self.settings['rasterized'])
        if self.settings['colorbar'] == 'True':
            self.cbar = self.figure.colorbar(self.image, orientation='vertical')
        if self.draw_full_range:        
            self.axes.set_xlim(left=min(self.measurement_bounds), 
                               right=max(self.measurement_bounds))
        self.cursor = Cursor(self.axes, useblit=True, 
                             color=self.settings['linecolor'], linewidth=0.5)
        if self.settings['metadata'] == 'True':
            self.axes.text(1.2, 0, self.settings_string, 
                           fontsize=self.settings['ticksize'], 
                           transform=self.axes.transAxes)
        self.apply_plot_settings()

    def add_plot_2d(self):
        if PRINT_FUNCTION_CALLS:
            print('add_plot_2d')
        data = self.processed_data
        cmap = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap = cmap+'_r'
        self.image = self.axes.plot(data[0], data[1], color=cm.get_cmap(cmap)(0.5))
        self.cursor = Cursor(self.axes, useblit=True, 
                             color=self.settings['linecolor'], linewidth=0.5)
        if self.settings['metadata'] == 'True':
            self.axes.text(1.02, 0, self.settings_string, 
                           fontsize=self.settings['ticksize'], 
                           transform=self.axes.transAxes)
        self.apply_plot_settings()        
        
    def reset_view_settings(self, overrule=False):
        if PRINT_FUNCTION_CALLS:
            print('reset_view_settings')
        if self.view_settings['Locked'] == 0 or overrule == True:
            self.view_settings['Minimum'] = np.min(self.processed_data[-1])
            self.view_settings['Maximum'] = np.max(self.processed_data[-1])
            self.view_settings['Midpoint'] = 0.5*(np.min(self.processed_data[-1])+
                             np.max(self.processed_data[-1]))
            self.view_settings['Norm'] = MidpointNormalize(
                    vmin=self.view_settings['Minimum'], 
                    vmax=self.view_settings['Maximum'], 
                    midpoint=self.view_settings['Midpoint'])
            self.view_settings['MidLock'] = 0
            
    def reset_midpoint(self):
        if PRINT_FUNCTION_CALLS:
            print('reset_midpoint')
        if self.view_settings['MidLock'] == 0:
            self.view_settings['Midpoint'] = 0.5*(self.view_settings['Minimum']+
                                                 self.view_settings['Maximum'])

    def adjust_norm_plot(self, midpoint='reset'):
        if PRINT_FUNCTION_CALLS:
            print('adjust_norm_plot '+midpoint)
        min_map = self.view_settings['Minimum']
        mid_map = self.view_settings['Midpoint']
        max_map = self.view_settings['Maximum']
        if midpoint == 'up':
            mid_map = mid_map - (max_map - min_map)*0.05
            if mid_map < min_map:
                mid_map = min_map
        elif midpoint == 'down':
            mid_map = mid_map + (max_map - min_map)*0.05
            if mid_map > max_map:
                mid_map = max_map
        elif midpoint == 'reset':
            mid_map = 0.5*(min_map + max_map)
        self.view_settings['Midpoint'] = mid_map
        self.apply_view_settings()
        
    def apply_plot_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('apply_plot_settings')
        settings = self.settings
        self.axes.set_xlabel(settings['xlabel'], size=settings['labelsize'])
        self.axes.set_ylabel(settings['ylabel'], size=settings['labelsize'])
        if isinstance(self.image, list):
            self.image[0].set_linewidth(float(settings['linewidth']))
        for axis in ['top','bottom','left','right']:
            self.axes.spines[axis].set_linewidth(float(self.settings['spinewidth']))
        self.axes.tick_params(labelsize=settings['ticksize'], width=float(self.settings['spinewidth']))
        if settings['minorticks'] == 'True':    
            self.axes.minorticks_on()
        if settings['title'] == '<filename>':
            self.axes.set_title(self.filename, size=settings['titlesize'])
        elif settings['title'] == '<metadataname>':
            self.axes.set_title("\n".join(wrap(self.meta_data_name,80)), size=settings['titlesize'])
        else:
            self.axes.set_title(settings['title'], size=settings['titlesize'])
        if settings['colorbar'] == 'True' and len(self.columns) == 3:
            self.cbar.ax.set_title(settings['clabel'], size=settings['labelsize'])
            self.cbar.ax.tick_params(labelsize=settings['ticksize']) 
            self.cbar.outline.set_linewidth(float(self.settings['spinewidth']))

    def apply_view_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('apply_view_settings')
        if len(self.columns) == 3:
            self.view_settings['Norm'] = MidpointNormalize(vmin=self.view_settings['Minimum'], 
                                                           vmax=self.view_settings['Maximum'], 
                                                           midpoint=self.view_settings['Midpoint'])
            self.image.set_norm(self.view_settings['Norm'])
            if self.settings['colorbar'] == 'True':
                self.cbar.update_normal(self.image)
                self.cbar.ax.set_title(self.settings['clabel'], 
                                       size=self.settings['labelsize'])
                self.cbar.ax.tick_params(labelsize=self.settings['ticksize'])          
    
    def apply_colormap(self):
        if PRINT_FUNCTION_CALLS:
            print('apply_colormap '+self.view_settings['Color Map'])
        cmap_str = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap_str = cmap_str+'_r'
        cmap = cm.get_cmap(cmap_str, lut=int(self.settings['lut']))
        cmap.set_bad(self.settings['maskcolor'])
        if len(self.columns) == 3:
            self.image.set_cmap(cmap)
        else:
            self.image[0].set_color(cmap(0.5))
            
    def apply_filter(self, filter_settings, update_color_limits=True):
        if PRINT_FUNCTION_CALLS:
            print('apply_filter')
        self.filters.append(filter_settings)
        if filter_settings['Checked']:
            if PRINT_FUNCTION_CALLS:
                print('Applying '+filter_settings['Name']+'...')
            self.processed_data = filters.apply(self.processed_data, filter_settings)
            if update_color_limits:
                self.reset_view_settings()
                self.apply_view_settings()

    def apply_all_filters(self, update_color_limits=True, refresh_unit_conversion=False):
        if PRINT_FUNCTION_CALLS:
            print('apply_all_filters')
        self.processed_to_raw()
        
        if self.meta_data:
            if self.rcfilter_correct:
                self.correct_for_rcfilters()
            if refresh_unit_conversion:
                self.filters = []
                self.meta_unit_conversion()
            if self.four_terminal:
                lockin_bias_index = self.channels.index('lockin_bias/X')
                if lockin_bias_index in self.columns:
                    lockin_curr_index = self.channels.index('lockin_curr/X')
                    self.processed_data[self.columns.index(lockin_bias_index)] = (self.raw_data[lockin_bias_index] / 
                                        self.raw_data[lockin_curr_index])
        
        for _, filter_settings in enumerate(self.filters):
            if filter_settings['Checked']:
                if PRINT_FUNCTION_CALLS:
                    print('Applying '+filter_settings['Name']+'...')
                self.processed_data = filters.apply(self.processed_data, filter_settings)
        if update_color_limits:
            self.reset_view_settings()
            if self.image:
                self.apply_view_settings()
            
    def update_linecut(self):
        if PRINT_FUNCTION_CALLS:
            print('update_linecut')
        if self.linecut_window.running:
            try:
                self.linecut.remove()
                del self.linecut
            except:
                pass
            if self.orientation == 'horizontal':
                x = self.processed_data[0][:,self.selected_indices[1]]
                y = self.processed_data[2][:,self.selected_indices[1]]
                value = self.processed_data[1][0,self.selected_indices[1]]
                self.linecut_window.xlabel = self.settings['xlabel']
                self.linecut_window.zlabel = self.settings['ylabel']
                self.linecut_window.title = self.settings['ylabel']+' = '+str(value)
                self.linecut = self.axes.axhline(
                        y=value, linestyle='dashed', linewidth=1, color=self.settings['linecolor'])
            elif self.orientation == 'vertical':
                x = self.processed_data[1][self.selected_indices[0],:]
                y = self.processed_data[2][self.selected_indices[0],:]
                value = self.processed_data[0][self.selected_indices[0],0]
                self.linecut_window.xlabel = self.settings['ylabel']
                self.linecut_window.zlabel = self.settings['xlabel']
                self.linecut_window.title = self.settings['xlabel']+' = '+str(value)
                self.linecut = self.axes.axvline(
                        x=value, linestyle='dashed', linewidth=1, color=self.settings['linecolor'])
            elif self.orientation == 'diagonal':
                x0, y0 = self.list_points[0].x, self.list_points[0].y
                x1, y1 = self.list_points[1].x, self.list_points[1].y
                
                l_x, l_y = self.processed_data[0].shape
                x_min = np.amin(self.processed_data[0][:,0])
                x_max = np.amax(self.processed_data[0][:,0])
                y_min = np.amin(self.processed_data[1][0,:])
                y_max = np.amax(self.processed_data[1][0,:])
                    
                i_x0 = (l_x-1)*(x0-x_min)/(x_max-x_min)
                i_y0 = (l_y-1)*(y0-y_min)/(y_max-y_min)
                i_x1 = (l_x-1)*(x1-x_min)/(x_max-x_min)
                i_y1 = (l_y-1)*(y1-y_min)/(y_max-y_min)
                
                n = int(np.sqrt((i_x1-i_x0)**2+(i_y1-i_y0)**2))
                x_diag, y_diag = np.linspace(i_x0, i_x1, n), np.linspace(i_y0, i_y1, n)
                y = map_coordinates(self.processed_data[-1], np.vstack((x_diag, y_diag)))
                x = map_coordinates(self.processed_data[0], np.vstack((x_diag, y_diag)))                
                self.linecut_window.xlabel = self.settings['xlabel']
                self.linecut_window.zlabel = ''
                self.linecut_window.title = ''
            elif self.orientation == 'circular':
                x0, y0 = self.list_points[0].x, self.list_points[0].y
                x1, y1 = self.list_points[1].x, self.list_points[2].y
                                
                l_x, l_y = self.processed_data[0].shape
                x_min = np.amin(self.processed_data[0][:,0])
                x_max = np.amax(self.processed_data[0][:,0])
                y_min = np.amin(self.processed_data[1][0,:])
                y_max = np.amax(self.processed_data[1][0,:])
                
                i_x0 = (l_x-1)*(x0-x_min)/(x_max-x_min)
                i_y0 = (l_y-1)*(y0-y_min)/(y_max-y_min)
                i_x1 = (l_x-1)*(x1-x_min)/(x_max-x_min)
                i_y1 = (l_y-1)*(y1-y_min)/(y_max-y_min)
                
                n = int(8*np.sqrt((i_x0-i_x1)**2+(i_y0-i_y1)**2))
                theta = np.linspace(0, 2*np.pi, n)
                
                i_x_circ = i_x0+(i_x1-i_x0)*np.cos(theta) 
                i_y_circ = i_y0+(i_y1-i_y0)*np.sin(theta)
                
                y = map_coordinates(self.processed_data[-1], np.vstack((i_x_circ, i_y_circ)))
                x = theta
               
                self.linecut_window.xlabel = 'Angle (rad)'
                self.linecut_window.zlabel = ''
                self.linecut_window.title = ''

            self.linecut_window.ylabel = self.settings['clabel']
            self.linecut_window.draw_plot(x, y)
            self.linecut_window.show()
    
    def update_multiple_linecuts(self):
        if PRINT_FUNCTION_CALLS:
            print('update_multiple_linecuts')
        if self.multi_orientation == 'horizontal':
            self.multi_linecuts_window.xlabel = self.settings['xlabel']
            self.multi_linecuts_window.zlabel = self.settings['ylabel']
        elif self.multi_orientation == 'vertical':
            self.multi_linecuts_window.xlabel = self.settings['ylabel']
            self.multi_linecuts_window.zlabel = self.settings['xlabel']
        self.multi_linecuts_window.orientation = self.multi_orientation
        self.multi_linecuts_window.ylabel = self.settings['clabel']
        self.multi_linecuts_window.title = ''
        #self.multi_linecuts_window.data = self.processed_data
        self.multi_linecuts_window.draw_plots()
        self.multi_linecuts_window.show()
        
    def open_fft_window(self):
        if PRINT_FUNCTION_CALLS:
            print('open_fft_window')
        if self.fft_orientation == 'vertical':
            self.fft = np.fft.rfft(self.processed_data[-1], axis=1)
        elif self.fft_orientation == 'horizontal':
            self.fft = np.fft.rfft(self.processed_data[-1], axis=0)
        self.fft_window = FFTWindow(self.fft)
        self.fft_window.show()


class LineCutWindow(QtWidgets.QWidget):
    def __init__(self, parent, multiple=False):
        super(self.__class__, self).__init__()
        if isinstance(parent, list):
            self.parent = parent[0]
            self.all_parents = parent
            self.multiple_files = True
        else:
            self.parent = parent
            self.multiple_files = False
        self.setWindowTitle('Inspectra Gadget - Linecut Window')
        self.multiple = multiple
        self.running = True
        self.resize(600, 600)
        self.linewidth = self.parent.settings['linewidth']
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.button_layout = QtWidgets.QHBoxLayout()
        self.figure = Figure(tight_layout={'pad':2})
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.save_button = QtWidgets.QPushButton('Save Data')
        self.save_button.clicked.connect(self.save_data)
        self.save_image_button = QtWidgets.QPushButton('Save Image')
        self.save_image_button.clicked.connect(self.save_image)
        self.copy_image_button = QtWidgets.QPushButton('Copy Image')
        self.copy_image_button.clicked.connect(self.copy_image)
        self.orientation_button = QtWidgets.QPushButton('Hor./Vert.')
        self.orientation_button.clicked.connect(self.change_orientation)
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear_lines)
        self.detect_peaks_button = QtWidgets.QPushButton('Detect Peaks')
        self.detect_peaks_button.clicked.connect(self.detect_peaks)
        if multiple:
            self.init_multiple()
        else:
            self.canvas.mpl_connect('scroll_event', self.mouse_scroll_canvas)
        self.vertical_layout.addWidget(self.canvas)
        self.vertical_layout.addWidget(self.navi_toolbar)
        if self.multiple_files == False:
            self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.save_image_button)
        self.button_layout.addWidget(self.copy_image_button)
        self.button_layout.addWidget(self.orientation_button)
        self.button_layout.addStretch()
        if self.multiple_files == False:
            self.init_fit()
        self.vertical_layout.addLayout(self.button_layout)
        self.setLayout(self.vertical_layout)
                
    def init_multiple(self):
        self.control_layout = QtWidgets.QHBoxLayout()
        self.number_label = QtWidgets.QLabel('Number of lines')
        self.number_line_edit = QtWidgets.QLineEdit('5')
        self.number_line_edit.setFixedSize(40,20)
        self.number_line_edit.editingFinished.connect(self.draw_plots)
        self.all_lines_button = QtWidgets.QPushButton('All')
        self.all_lines_button.setFixedSize(35,22)
        self.all_lines_button.clicked.connect(self.clicked_all_lines)
        self.check_specify_lines = QtWidgets.QCheckBox('Specify lines:')
        self.check_specify_lines.clicked.connect(self.draw_plots)
        self.specify_lines_edit = QtWidgets.QLineEdit('')
        self.specify_lines_edit.editingFinished.connect(self.draw_plots)
        self.offset_label = QtWidgets.QLabel('Offset')
        self.offset_line_edit = QtWidgets.QLineEdit('0')
        self.offset_line_edit.setFixedSize(70,20)
        self.offset_line_edit.editingFinished.connect(self.draw_plots)
        self.check_legend = QtWidgets.QCheckBox('Legend')
        self.check_legend.clicked.connect(self.draw_plots)
        if self.multiple_files == False:
            self.control_layout.addWidget(self.number_label)
            self.control_layout.addWidget(self.number_line_edit)
            self.control_layout.addWidget(self.all_lines_button)
            self.control_layout.addWidget(self.check_specify_lines)
            self.control_layout.addWidget(self.specify_lines_edit)
            self.canvas.mpl_connect('button_press_event', self.mouse_click_canvas)
        self.control_layout.addWidget(self.offset_label)
        self.control_layout.addWidget(self.offset_line_edit)
        self.control_layout.addWidget(self.check_legend)
        self.cmaps = cmaps
        self.colormap_box = QtWidgets.QComboBox()
        self.colormap_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colormap_type_box = QtWidgets.QComboBox()
        self.colormap_type_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)        
        self.cmaps = cmaps
        for cmap_type in self.cmaps:    
            self.colormap_type_box.addItem(cmap_type)
        self.colormap_box.addItems(list(self.cmaps.values())[0])
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.colormap_type_box)
        self.control_layout.addWidget(self.colormap_box)
        self.colormap_type_box.currentIndexChanged.connect(self.colormap_type_edited)
        self.colormap_box.currentIndexChanged.connect(self.draw_plots)
        self.vertical_layout.addLayout(self.control_layout) 
    
    def init_fit(self):
        self.fit_layout = QtWidgets.QHBoxLayout()
        self.manual_box = QtWidgets.QCheckBox('Manual inital guess')
        self.guess_edit = QtWidgets.QLineEdit()
        self.fit_box = QtWidgets.QComboBox()
        self.fit_box.addItems(fits.get_names())
        self.fit_box.setCurrentIndex(0)
        self.fit_box.currentIndexChanged.connect(self.fit_type_changed)
        self.pars_label = QtWidgets.QLabel(fits.get_names(parameters=self.fit_box.currentText()))
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.fit_button.clicked.connect(self.start_fitting)
        self.fit_layout.addStretch()
        self.fit_layout.addWidget(self.manual_box)
        self.fit_layout.addWidget(self.guess_edit)
        self.fit_layout.addWidget(self.pars_label)
        self.fit_layout.addWidget(self.fit_box)        
        self.vertical_layout.addLayout(self.fit_layout)
        self.reverse_checkbox = QtWidgets.QCheckBox('Reverse Fit Order')
        self.reverse_checkbox.setCheckState(QtCore.Qt.Checked)
        if self.multiple and lmfit_imported:
            self.button_layout.addWidget(self.reverse_checkbox)
            self.button_layout.addWidget(self.detect_peaks_button)
        self.button_layout.addWidget(self.fit_button)
        self.button_layout.addWidget(self.clear_button)
        self.peak_estimates = []
    
    def mouse_click_canvas(self, event):
        if self.navi_toolbar.mode == '':
            if event.inaxes and lmfit_imported:
                x = event.xdata                
                if event.button == 1:
                    self.peak_estimates.append(self.axes.axvline(x, linestyle='dashed'))
                    self.canvas.draw()
                elif event.button == 3:
                    if self.peak_estimates:
                        # Remove peak closest to right-click
                        peak_index = np.argmin(np.array([np.abs(p.get_xdata()[0]-x) for p in self.peak_estimates]))
                        self.peak_estimates[peak_index].remove()
                        del self.peak_estimates[peak_index]
                        self.canvas.draw()
             
    def change_orientation(self):
        if self.parent.orientation == 'horizontal':
            self.parent.orientation = 'vertical'
        elif self.parent.orientation == 'vertical':
            self.parent.orientation = 'horizontal'
        self.parent.update_linecut()
        self.parent.canvas.draw()
             
    def clear_lines(self):
        for line in reversed(self.axes.get_lines()):
            if line.get_linestyle() == '--':
                line.remove()
                del line
        self.canvas.draw()
        self.peak_estimates = []
    
    def detect_peaks(self):
        self.clear_lines()
        if self.reverse_checkbox.checkState() == 2: 
            index_trace = np.argmax(self.z[:,0])
        else:
            index_trace = np.argmin(self.z[:,0])
        peaks, _ = find_peaks(self.y[index_trace,:], width=5)
        for peak in peaks:
            self.peak_estimates.append(self.axes.axvline(self.x[index_trace,peak], linestyle='dashed'))
        self.canvas.draw()
    
    def fit_type_changed(self):
        self.pars_label.setText(fits.get_names(parameters=self.fit_box.currentText()))
    
    def start_fitting(self):
        function_name = self.fit_box.currentText()
        if len(self.peak_estimates) > 1:
            center_estimates = [p.get_xdata()[0] for p in self.peak_estimates]
            n_peaks = len(center_estimates)
            self.clear_lines()
            line_indices = list(range(self.z.shape[0]))
            if np.argmin(self.z[:,0]) == 0 and self.reverse_checkbox.checkState() == 2:  
                line_indices.reverse()
            elif np.argmax(self.z[:,0]) == 0 and self.reverse_checkbox.checkState() == 0: 
                line_indices.reverse()
            
            # Initialize objects
            values = None
            peak_values = {}
            peak_values['background_constant'] = []
            peak_values['mean_height'] = []
            peak_values['mean_fwhm'] = []
            peak_values['coordinates'] = self.z[line_indices,0]
            for i in range(len(center_estimates)):
                peak_values['p{}_center'.format(i)] = [] 
                peak_values['p{}_height'.format(i)] = [] 
                peak_values['p{}_fwhm'.format(i)] = []

            # Fit routine for every trace
            for counter, line_index in enumerate(line_indices):
                x, y, z = self.x[line_index,:], self.y[line_index,:], self.z[line_index,0]
                print('Fitting peaks for trace; index = {}, value = {}...'.format(line_index, z))
                if counter == 0:
                    background_estimate = np.amin(y)
                    height_estimate = np.amax(y) - np.amin(y)
                    fwhm_estimate = 0.25*min([abs(center_estimates[0]-c_est) for c_est in center_estimates[1:]])
                    if function_name == 'Lorentzian':
                        sigma_estimate = fwhm_estimate/2
                        amplitude_estimate = height_estimate*sigma_estimate*np.pi
                    else: # Gaussian
                        sigma_estimate = fwhm_estimate/2.35482
                        amplitude_estimate = height_estimate*sigma_estimate*np.sqrt(2*np.pi)
                    peak_bounds = (np.amin(x)-0.5*(np.amax(x)-np.amin(x)), np.amax(x)+0.5*(np.amax(x)-np.amin(x)))
                else:
                    background_estimate = values['bkg_c']
                    sigma_estimate = values['p0_sigma']
                    amplitude_estimate = values['p0_amplitude']
                    center_estimates = [values['p{}_center'.format(i)] for i in range(len(center_estimates))]
                model = ConstantModel(prefix='bkg_')
                params = model.make_params()
                params['bkg_c'].set(background_estimate, min=-np.inf)
                for i, center_estimate in enumerate(center_estimates):
                    peak, pars = self.add_peak('p{}_'.format(i), function_name, 
                                               center_estimate, peak_bounds, 
                                               amplitude_estimate, sigma_estimate)
                    model = model + peak
                    params.update(pars)
                result = model.fit(y, params, x=x)
                values = result.best_values
                print('Fit finished...')
                
                # Save fit values in dictionary
                peak_values['background_constant'].append(values['bkg_c'])
                for i in range(n_peaks):
                    peak_values['p{}_center'.format(i)].append(values['p{}_center'.format(i)])
                    if function_name == 'Lorentzian':    
                        best_fwhm = 2*values['p{}_sigma'.format(i)]
                        best_height = values['p{}_amplitude'.format(i)]/(np.pi*values['p{}_sigma'.format(i)])
                    else:
                        best_fwhm = 2.35482*values['p{}_sigma'.format(i)]
                        best_height = values['p{}_amplitude'.format(i)]/(np.sqrt(2*np.pi)*values['p{}_sigma'.format(i)])
                    peak_values['p{}_height'.format(i)].append(best_height)
                    peak_values['p{}_fwhm'.format(i)].append(best_fwhm)
                self.axes.plot(x, result.best_fit+line_index*self.offset, 'k--')
            self.canvas.draw()
            
            # Plot peak center positions as dots in main canvas
            for i in range(n_peaks):
                if self.orientation == 'vertical':
                    self.parent.axes.scatter(self.z[line_indices,0], peak_values['p{}_center'.format(i)], c='r')
                elif self.orientation == 'horizontal':
                    self.parent.axes.scatter(peak_values['p{}_center'.format(i)], self.z[line_indices,0], c='r')
            self.parent.canvas.draw()
            
            # Calculate properties of peaks
            peak_values['mean_height'] = [np.mean([peak_values['p{}_height'.format(i)][j] 
                                                   for i in range(n_peaks)]) for j in range(self.number)]
            peak_values['mean_fwhm'] = [np.mean([peak_values['p{}_fwhm'.format(i)][j] 
                                                 for i in range(n_peaks)]) for j in range(self.number)]
            for index, parity in enumerate(['even','odd']):
                peak_values['mean_spacing_'+parity] = [np.mean([np.abs(peak_values['p{}_center'.format(i)][j] - 
                                                                       peak_values['p{}_center'.format(i+1)][j]) 
                                                                for i in range(index,n_peaks-1,2)]) for j in range(self.number)]
                peak_values['mean_height_'+parity] = [np.mean([peak_values['p{}_height'.format(i)][j] 
                                                               for i in range(index,n_peaks,2)]) for j in range(self.number)]
                peak_values['mean_fwhm_'+parity] = [np.mean([peak_values['p{}_fwhm'.format(i)][j] 
                                                             for i in range(index,n_peaks,2)]) for j in range(self.number)]

            # Save dictionary with peak properties to file    
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Peak Fitting Data As...', os.path.splitext(self.parent.filepath)[0]+'_fitdata', '*.npy')
            if filename:
                np.save(filename, peak_values)
                print('Pead fitting data saved!')
        else: 
            self.y_fit = np.copy(self.y)
            if self.manual_box.checkState():
                p0 = None
            else:
                p0 = [float(par) for par in self.guess_edit.text().split()]
            if self.multiple:
                self.fit_parameters = []
                for index in range(int(self.number_line_edit.text())):
                    try:
                        popt = fits.fit_data(function_name=function_name, xdata=self.x[index], 
                                             ydata=self.y[index], p0=p0)
                        self.fit_parameters.append(list(popt))
                        self.y_fit[index] = fits.get_function(function_name)(self.x[index], *popt)
                    except RuntimeError:
                        print('Curve with index '+str(index)+' could not be fitted...')
                        nans = [np.nan]*len(fits.get_names(function_name).split(','))
                        self.fit_parameters.append(nans)
                        self.y_fit[index] = np.nan
            else:
                try:
                    self.fit_parameters = fits.fit_data(function_name=function_name, xdata=self.x, 
                                                        ydata=self.y, p0=p0)
                    self.y_fit = fits.get_function(function_name)(self.x, *self.fit_parameters)
                except RuntimeError:
                    print('Curve could not be fitted...')
                    self.fit_parameters = [np.nan]*len(fits.get_names(function_name).split(','))
                    self.y_fit = np.nan            
            if self.multiple:
                self.draw_plots()
            else:
                self.draw_plot(self.x, self.y)
            self.draw_fits()
            self.plot_parameters()
        
    def add_peak(self, prefix, shape, center, bounds, amplitude, sigma):
        if shape == "Lorentzian":
            peak = LorentzianModel(prefix=prefix)
        else:
            peak = GaussianModel(prefix=prefix)
        pars = peak.make_params()
        pars[prefix + 'center'].set(center, min=bounds[0], max=bounds[1])
        pars[prefix + 'amplitude'].set(amplitude, min=0)
        pars[prefix + 'sigma'].set(sigma, min=0)
        return peak, pars
        
    def plot_parameters(self):
        if self.multiple:
            self.pfit = np.array(self.fit_parameters).transpose()
            rows, cols = self.pfit.shape
            self.pars_window = []
            for index in range(rows):
                ylabel = fits.get_names(self.fit_box.currentText()).split(',')[index]
                self.pars_window.append(ParametersWindow(
                        self.z[:,0], self.pfit[index], self.zlabel, ylabel))
                self.pars_window[-1].show()
        else:
            self.axes.text(0.02,0.95,'Position: %.4g' % self.fit_parameters[2], transform=self.axes.transAxes)
            self.axes.text(0.02,0.9,'Height: %.4g' % self.fit_parameters[1], transform=self.axes.transAxes)
            self.axes.text(0.02,0.85,'Width: %.4g' % self.fit_parameters[0], transform=self.axes.transAxes)
            self.axes.text(0.02,0.8,'Background: %.4g' % self.fit_parameters[3], transform=self.axes.transAxes)
            self.canvas.draw()
    
    def colormap_type_edited(self):
        self.colormap_box.currentIndexChanged.disconnect(self.draw_plots)
        self.colormap_box.clear()
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentText()])
        self.colormap_box.currentIndexChanged.connect(self.draw_plots)
        self.draw_plots()
         
    def draw_plot(self, x, y):
        self.running = True
        self.x, self.y = x, y
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.image = self.axes.plot(x, y, 'C3', linewidth=self.linewidth)
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()

    def draw_plots(self):
        self.running = True
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.offset = float(self.offset_line_edit.text())
        selected_colormap = cm.get_cmap(self.colormap_box.currentText())
        
        if self.multiple_files:
            self.x = np.stack([p.processed_data[0] for p in self.all_parents])
            self.y = np.stack([p.processed_data[1] for p in self.all_parents])
            self.labels = [p.filename for p in self.all_parents] # todo
            self.number = len(self.all_parents)
        else:
            rows, cols = self.parent.processed_data[0].shape
            self.number = int(self.number_line_edit.text())
            if self.orientation == 'horizontal':
                if self.check_specify_lines.checkState() == 0:
                    indices = np.linspace(0, cols-1, self.number, dtype=int)
                else:
                    try:
                        if ':' in self.specify_lines_edit.text():
                            min_value = float(self.specify_lines_edit.text().split(':')[0])
                            max_value = float(self.specify_lines_edit.text().split(':')[-1])
                            min_index = np.argmin(np.abs(self.parent.processed_data[1][0,:]-min_value))
                            max_index = np.argmin(np.abs(self.parent.processed_data[1][0,:]-max_value))
                            indices = list(range(min_index,max_index+1))
                        else:
                            indices = [np.argmin(np.abs(self.parent.processed_data[1][0,:]-float(s))) for s in self.specify_lines_edit.text().split(',')]
                        self.number = len(indices)
                    except:
                        indices = []
                        self.number = 0
                self.x = self.parent.processed_data[0][:,indices].transpose()
                self.y = self.parent.processed_data[2][:,indices].transpose()
                self.z = self.parent.processed_data[1][:,indices].transpose()
            elif self.orientation == 'vertical':
                if self.check_specify_lines.checkState() == 0:
                    indices = np.linspace(0, rows-1, self.number, dtype=int)
                else:
                    try:
                        if ':' in self.specify_lines_edit.text():
                            min_value = float(self.specify_lines_edit.text().split(':')[0])
                            max_value = float(self.specify_lines_edit.text().split(':')[-1])
                            min_index = np.argmin(np.abs(self.parent.processed_data[0][:,0]-min_value))
                            max_index = np.argmin(np.abs(self.parent.processed_data[0][:,0]-max_value))
                            indices = list(range(min_index,max_index+1))
                        else:
                            indices = [np.argmin(np.abs(self.parent.processed_data[0][:,0]-float(s))) for s in self.specify_lines_edit.text().split(',')]
                        self.number = len(indices)
                    except:
                        indices = []
                        self.number = 0
    
                self.x = self.parent.processed_data[1][indices,:]
                self.y = self.parent.processed_data[2][indices,:]
                self.z = self.parent.processed_data[0][indices,:]
            self.labels = ['{:.5g}'.format(self.z[i,0]) for i in range(self.number)]            
        
        line_colors = selected_colormap(np.linspace(0.1,0.9,self.number))
        for index in range(self.number):
            self.axes.plot(self.x[index], self.y[index]+index*self.offset, 
                           color=line_colors[index], linewidth=self.linewidth, 
                           label=self.labels[index])
        if self.check_legend.checkState():
            self.axes.legend(title=self.zlabel)
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()
              
    def draw_fits(self):
        if self.multiple:
            for index in range(self.number):
                self.axes.plot(self.x[index], self.y_fit[index]+index*self.offset, 
                               'k--', linewidth=self.linewidth)
        else:
            self.axes.plot(self.x, self.y_fit, 'k--', linewidth=self.linewidth)            
        self.canvas.draw()
    
    def apply_plot_settings(self):
        self.axes.set_xlabel(self.xlabel, size='xx-large')
        self.axes.set_ylabel(self.ylabel, size='xx-large')
        self.axes.tick_params(labelsize='x-large')
        self.axes.set_title(self.title, size='x-large')
        self.canvas.draw()
        
    def clicked_all_lines(self):
        rows, cols = self.parent.processed_data[0].shape
        if self.orientation == 'horizontal':
            number = cols
        elif self.orientation == 'vertical':
            number = rows
        self.number_line_edit.setText(str(number))
        self.draw_plots()
        
    def closeEvent(self, event):
        self.running = False
        
    def save_data(self):
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Data As...')
        if self.multiple:
            data = np.array([self.z.flatten(), self.x.flatten(), self.y.flatten()])
        else:
            data = np.array([self.x, self.y])
        np.savetxt(filename, data.T)
        
    def save_image(self):
        formats = 'Portable Network Graphic (*.png);;Adobe Acrobat (*.pdf)'
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Figure As...', '', formats)
        if filename:
            print('Save Figure as '+filename+' ...')
            self.figure.savefig(filename)
            print('Saved!')
    
    def copy_image(self):
        self.cursor.horizOn = False
        self.cursor.vertOn = False            
        self.canvas.draw()            
        buf = io.BytesIO()
        self.figure.savefig(buf, dpi=300, transparent=True, bbox_inches='tight')
        QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
        buf.close()
        self.cursor.horizOn = True
        self.cursor.vertOn = True                       
        self.canvas.draw()
            
    def mouse_scroll_canvas(self, event):
        if event.inaxes:
            if len(self.parent.columns) == 3:
                data_shape = self.parent.processed_data[0].shape
                if self.parent.orientation == 'horizontal':
                    new_index = self.parent.selected_indices[1]+int(event.step)
                    if new_index >= 0 and new_index < data_shape[1]:
                        self.parent.selected_indices[1] = new_index
                elif self.parent.orientation == 'vertical':
                    new_index = self.parent.selected_indices[0]+int(event.step)
                    if new_index >= 0 and new_index < data_shape[0]:
                        self.parent.selected_indices[0] = new_index
                self.parent.update_linecut()
                self.parent.canvas.draw()


class ParametersWindow(QtWidgets.QWidget):
    def __init__(self, x, y, xlabel, ylabel):
        super(self.__class__, self).__init__()
        self.resize(600, 600)
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.x = x
        self.y = y
        self.axes.plot(self.x, self.y,'.')
        self.axes.set_xlabel(xlabel, size=DEFAULT_PLOT_SETTINGS['labelsize'])
        self.axes.set_ylabel(ylabel, size=DEFAULT_PLOT_SETTINGS['labelsize'])
        self.axes.tick_params(labelsize=DEFAULT_PLOT_SETTINGS['ticksize'])
        self.canvas.draw()
        self.figure.tight_layout(pad=2)
        self.canvas.draw()
        self.save_button = QtWidgets.QPushButton('Save')
        self.save_button.clicked.connect(self.save_data)
        self.save_image_button = QtWidgets.QPushButton('Save Image')
        self.save_image_button.clicked.connect(self.save_image)
        self.vertical_layout.addWidget(self.navi_toolbar)
        self.vertical_layout.addWidget(self.canvas)
        self.vertical_layout.addWidget(self.save_button)
        self.vertical_layout.addWidget(self.save_image_button)
        self.setLayout(self.vertical_layout)
        
    def save_data(self):
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Data As...')
        data = np.array([self.x, self.y])
        np.savetxt(filename, data.T)

    def save_image(self):
        formats = 'Portable Network Graphic (*.png);;Adobe Acrobat (*.pdf)'
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Figure As...', '', formats)
        if filename:
            print('Save Figure as '+filename+' ...')
            self.figure.savefig(filename)
            print('Saved!')   
  
            
class FFTWindow(QtWidgets.QWidget):
    def __init__(self, fftdata):
        super(self.__class__, self).__init__()
        self.resize(600, 600)
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.fft = np.absolute(fftdata).transpose()
        self.image = self.axes.pcolormesh(self.fft, shading='auto', norm=LogNorm(vmin=self.fft.min(), vmax=self.fft.max()))
        self.cbar = self.figure.colorbar(self.image, orientation='vertical')
        self.figure.tight_layout(pad=2)
        self.axes.tick_params(labelsize=DEFAULT_PLOT_SETTINGS['ticksize'])
        self.canvas.draw()
        self.vertical_layout.addWidget(self.navi_toolbar)
        self.vertical_layout.addWidget(self.canvas)
        self.setLayout(self.vertical_layout)
    

class NoScrollQComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super(NoScrollQComboBox, self).__init__(*args, **kwargs)

    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QtWidgets.QComboBox.wheelEvent(self, *args, **kwargs)      
        
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


class NavigationToolbarMod(NavigationToolbar):
    #without save button
    NavigationToolbar.toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'))
  

class DraggablePoint:
    lock = None #  only one can be animated at a time
    def __init__(self, parent, x, y, draw_line=True, draw_circle=False):
        self.parent = parent
        self.axes = parent.axes
        self.draw_line = draw_line
        self.draw_circle = draw_circle
        
        x_lb, x_ub = self.axes.get_xlim()
        y_lb, y_ub = self.axes.get_ylim()
        
        self.point = patches.Ellipse((x, y), (x_ub-x_lb)*0.01, (y_ub-y_lb)*0.01, fc='k', alpha=1, edgecolor='k')
        self.x = x
        self.y = y
        self.axes.add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()
        
        if self.parent.list_points and self.draw_line:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]
            self.line = Line2D(line_x, line_y, color=self.parent.settings['linecolor'], alpha=1.0, linestyle='dashed', linewidth=1)
            self.axes.add_line(self.line)
        if len(self.parent.list_points) > 1 and self.draw_circle:
            x0, y0 = self.parent.list_points[0].x, self.parent.list_points[0].y
            x1, y1 = self.parent.list_points[1].x, self.y
            self.circle = patches.Ellipse((x0, y0), 2*(x1-x0), 2*(y1-y0), fc='none', alpha=None, 
                                          linestyle='dashed', linewidth=1, edgecolor='k')
            self.axes.add_patch(self.circle)

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        if len(self.parent.list_points) > 1 and self.draw_line:
            if self == self.parent.list_points[1]:
                self.line.set_animated(True)
            else:
                self.parent.list_points[1].line.set_animated(True)
        if len(self.parent.list_points) > 2 and self.draw_circle:
            if self == self.parent.list_points[2]:
                self.circle.set_animated(True)
            else:
                self.parent.list_points[2].circle.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)
        axes.draw_artist(self.point)
        canvas.blit(axes.bbox)


    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        if self.draw_circle:
            if self == self.parent.list_points[1]:
                dy = 0
            elif self == self.parent.list_points[2]:
                dx = 0
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
        canvas = self.point.figure.canvas
        axes = self.point.axes
        canvas.restore_region(self.background)
        axes.draw_artist(self.point)
        if len(self.parent.list_points) > 1 and self.draw_line:
            if self == self.parent.list_points[1]:
                axes.draw_artist(self.line)
            else:
                self.parent.list_points[1].line.set_animated(True)
                axes.draw_artist(self.parent.list_points[1].line)
        if len(self.parent.list_points) > 2 and self.draw_circle:
            if self == self.parent.list_points[2]:
                axes.draw_artist(self.circle)
            else:
                self.parent.list_points[2].circle.set_animated(True)
                axes.draw_artist(self.parent.list_points[2].circle)
        self.x = self.point.center[0]
        self.y = self.point.center[1]
        if len(self.parent.list_points) > 1 and self.draw_line:
            if self == self.parent.list_points[1]:
                line_x = [self.parent.list_points[0].x, self.x]
                line_y = [self.parent.list_points[0].y, self.y]
                self.line.set_data(line_x, line_y)
            else:
                line_x = [self.x, self.parent.list_points[1].x]
                line_y = [self.y, self.parent.list_points[1].y]
                self.parent.list_points[1].line.set_data(line_x, line_y)
        elif len(self.parent.list_points) > 2 and self.draw_circle:
            if self == self.parent.list_points[2]:
                self.circle.height = 2*(self.y-self.parent.list_points[0].y)
            elif self == self.parent.list_points[1]:
                self.parent.list_points[2].circle.width = 2*(self.x-self.parent.list_points[0].x)
            else:
                self.parent.list_points[2].circle.set_center((self.x, self.y))
        canvas.blit(axes.bbox)


    def on_release(self, event):
        if DraggablePoint.lock is not self:
            return
        self.press = None
        DraggablePoint.lock = None
        self.point.set_animated(False)
        if len(self.parent.list_points) > 1 and self.draw_line:
            if self == self.parent.list_points[1]:
                self.line.set_animated(False)
            else:
                self.parent.list_points[1].line.set_animated(False)
        if len(self.parent.list_points) > 2 and self.draw_circle:
            if self == self.parent.list_points[2]:
                self.circle.set_animated(False)
            else:
                self.parent.list_points[2].circle.set_animated(False)
            if self == self.parent.list_points[0]:
                circle = self.parent.list_points[2].circle
                self.parent.list_points[1].point.center = (circle.center[0]+0.5*circle.width, circle.center[1])
                self.parent.list_points[2].point.center = (circle.center[0],circle.center[1]+0.5*circle.height)
                self.parent.list_points[1].x = self.parent.list_points[1].point.center[0]
                self.parent.list_points[1].y = self.parent.list_points[1].point.center[1]
                self.parent.list_points[2].x = self.parent.list_points[2].point.center[0]
                self.parent.list_points[2].y = self.parent.list_points[2].point.center[1]
        self.background = None
        self.point.figure.canvas.draw()
        self.x = self.point.center[0]
        self.y = self.point.center[1]
        if len(self.parent.list_points) > 1:
            self.parent.update_linecut()
            self.parent.linecut_window.activateWindow()

    def disconnect(self):
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    app.lastWindowClosed.connect(app.quit)
    edit_window = Editor()
    edit_window.show()
    app.exec_()

if __name__ == '__main__':
    main()