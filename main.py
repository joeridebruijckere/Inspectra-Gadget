# -*- coding: utf-8 -*-
"""
Inspectra-Gadget

Author: Joeri de Bruijckere

Last updated on Nov 18 2020

"""

from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os
import copy
import io
from stat import ST_CTIME
from datetime import datetime
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
from cycler import cycler
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
try:
    import qdarkstyle # pip install qdarkstyle
    qdarkstyle_imported = True
except ModuleNotFoundError:
    qdarkstyle_imported = False
try:
    import qcodes as qc
    qcodes_imported = True
except ModuleNotFoundError:
    qcodes_imported = False

import design
import filters
import fits

# UI settings
DARK_THEME = True
AUTO_REFRESH_INTERVAL_2D = 1
AUTO_REFRESH_INTERVAL_3D = 30

# List of custom presets
PRESETS = [{'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5',
            'titlesize': '9',
            'canvas_bounds': (0.425,0.4,0.575,0.6), # (left, bottom, right, top)
            'show_meta_settings': False},
           {'title': '<label>', 'labelsize': '16', 'ticksize': '16', 
            'spinewidth': '0.8', 'titlesize': '16', 
            'canvas_bounds': (0.425,0.4,0.575,0.6), # (left, bottom, right, top)
            'show_meta_settings': True},
           {'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5'},
           {'title': '', 'labelsize': '9', 'ticksize': '9', 'spinewidth': '0.5'}]

# Matplotlib settings; font type is chosen such that text (labels, ticks, ...) 
# can be recognized by Illustrator
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.cursive'] = ['Arial']
rcParams['mathtext.fontset'] = 'custom'
if DARK_THEME and qdarkstyle_imported:
    DARK_COLOR = '#19232D'
    GREY_COLOR = '#505F69'
    LIGHT_COLOR = '#F0F0F0'
    BLUE_COLOR = '#148CD2'
    rcParams['figure.facecolor'] = DARK_COLOR
    rcParams['axes.facecolor'] = DARK_COLOR
    rcParams['axes.edgecolor'] = GREY_COLOR
    rcParams['text.color'] = LIGHT_COLOR
    rcParams['xtick.color'] = LIGHT_COLOR
    rcParams['ytick.color'] = LIGHT_COLOR
    rcParams['axes.labelcolor'] = LIGHT_COLOR
    rcParams['savefig.facecolor'] = 'white'
    color_cycle = [BLUE_COLOR, 'ff7f0e', '2ca02c', 'd62728', '9467bd', 
                   '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']
    rcParams['axes.prop_cycle'] = cycler('color', color_cycle)

# Colormaps
cmaps = OrderedDict()
cmaps['Uniform'] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential'] = ['Greys','Purples','Blues','Greens','Oranges','Reds',
                       'YlOrBr','YlOrRd','OrRd','PuRd','RdPu','BuPu','GnBu', 
                       'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential (2)'] = ['binary','gist_yarg','gist_gray','gray','bone',
                           'pink','spring','summer','autumn','winter','cool',
                           'Wistia','hot','afmhot','gist_heat','copper']
cmaps['Diverging'] = ['PiYG','PRGn','BrBG','PuOr','RdGy','RdBu','RdYlBu',
                      'RdYlGn','Spectral','coolwarm','bwr','seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = ['Pastel1','Pastel2','Paired','Accent','Dark2','Set1',
                        'Set2','Set3','tab10','tab20','tab20b','tab20c']
cmaps['Miscellaneous'] = ['flag','prism','ocean','gist_earth','terrain',
                          'gist_stern','gnuplot','gnuplot2','CMRmap',
                          'cubehelix','brg','gist_rainbow','rainbow','jet',
                          'nipy_spectral','gist_ncar']

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
    cmaps[cmap_type][:] = [cmap for cmap in cmaps[cmap_type] 
                           if cmap in plt.colormaps()]
    if cmaps[cmap_type] == []:
        del cmaps[cmap_type]
        
FONT_SIZES = ['8', '9', '10', '12', '14', '16', '18', '24']
SETTINGS_MENU_OPTIONS = OrderedDict()
SETTINGS_MENU_OPTIONS['title'] = [' ','<label>']
SETTINGS_MENU_OPTIONS['xlabel'] = ['Gate voltage (V)', 
                                   '$V_{\mathrm{g}}$ (V)',
                                   'Bias voltage (mV)', 
                                   '$V$ (mV)',
                                   'Magnetic Field (T)', 
                                   '$B$ (T)', 
                                   'Angle (degrees)']
SETTINGS_MENU_OPTIONS['ylabel'] = ['Bias voltage (mV)', 
                                   '$V$ (mV)', 
                                   'Gate voltage (V)', 
                                   '$V_{\mathrm{g}}$ (V)', 
                                   'd$I$/d$V$ (μS)', 
                                   'd$I$/d$V$ $(e^{2}/h)$', 
                                   'Angle (degrees)', 
                                   'Temperature (mK)']
SETTINGS_MENU_OPTIONS['clabel'] = ['$I$ (nA)', 
                                   '$I$ (a.u.)', 
                                   'Current (nA)', 
                                   'd$I$/d$V$ (μS)', 
                                   'd$I$/d$V$ ($G_0$)', 
                                   'd$I$/d$V$ (a.u.)', 
                                   'd$I$/d$V$ $(e^{2}/h)$', 
                                   'log$^{10}$(d$I$/d$V$ $(e^{2}/h)$)', 
                                   'd$^2I$/d$V^2$ (a.u.)', 
                                   '|d$^2I$/d$V^2$| (a.u.)']
SETTINGS_MENU_OPTIONS['titlesize'] = FONT_SIZES
SETTINGS_MENU_OPTIONS['labelsize'] = FONT_SIZES
SETTINGS_MENU_OPTIONS['ticksize'] = FONT_SIZES
SETTINGS_MENU_OPTIONS['colorbar'] = ['True', 'False']
SETTINGS_MENU_OPTIONS['columns'] = ['0,1,2','0,1,3','0,2,3','1,2,4']
SETTINGS_MENU_OPTIONS['minorticks'] = ['True','False']
SETTINGS_MENU_OPTIONS['delimiter'] = [' ',',']
SETTINGS_MENU_OPTIONS['linecolor'] = ['black', 'red', 'white', 
                                        'blue', 'green']
SETTINGS_MENU_OPTIONS['maskcolor'] = ['black','white']
SETTINGS_MENU_OPTIONS['lut'] = ['128','256','512','1024']
SETTINGS_MENU_OPTIONS['rasterized'] = ['True','False']
SETTINGS_MENU_OPTIONS['dpi'] = ['figure','300']
SETTINGS_MENU_OPTIONS['transparent'] = ['True', 'False']


class Editor(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.window_title = 'Inspectra Gadget'
        self.window_title_auto_refresh = ''
        self.setupUi(self)
        self.init_plot_settings()
        self.init_view_settings()
        self.init_filters()
        self.init_connections()
        self.init_canvas()
        self.linked_folder = None
        self.linked_files = []
    
    def init_plot_settings(self):
        self.settings_table.setColumnCount(2)
        self.settings_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        for col in range(2):
            h = self.settings_table.horizontalHeader()
            h.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.settings_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.settings_table.customContextMenuRequested.connect(self.open_plot_settings_menu)
    
    def init_view_settings(self):
        self.cmaps = cmaps
        for cmap_type in self.cmaps:    
            self.colormap_type_box.addItem(cmap_type)
        self.colormap_box.addItems(list(self.cmaps.values())[0])
        self.min_line_edit.setAlignment(QtCore.Qt.AlignRight | 
                                        QtCore.Qt.AlignVCenter)
        self.max_line_edit.setAlignment(QtCore.Qt.AlignRight | 
                                        QtCore.Qt.AlignVCenter)
        self.mid_line_edit.setAlignment(QtCore.Qt.AlignRight | 
                                        QtCore.Qt.AlignVCenter)
    
    def init_filters(self):                
        self.filters_combobox.addItem('<Add Filter>')
        self.filters_combobox.addItems(Filter.DEFAULT_SETTINGS.keys())
        self.filters_table.setColumnCount(4)
        self.filters_table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        h = self.filters_table.horizontalHeader()
        h.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in range(1,4):
            h.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        
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
        self.track_button.clicked.connect(self.track_button_clicked)
        self.action_open_files_from_folder.triggered.connect(self.open_files_from_folder)
        self.action_preset_0.triggered.connect(lambda: self.apply_preset(0))
        self.action_preset_1.triggered.connect(lambda: self.apply_preset(1))
        self.action_preset_2.triggered.connect(lambda: self.apply_preset(2))
        self.action_preset_3.triggered.connect(lambda: self.apply_preset(3))
        self.action_refresh_stop.setEnabled(False)
        self.action_link_to_folder.triggered.connect(lambda: self.update_link_to_folder(new_folder=True))
        self.action_unlink_folder.triggered.connect(self.unlink_folder)
        self.refresh_file_button.clicked.connect(self.update_plots)
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

    def open_files(self, filepaths=None):
        self.file_list.itemChanged.disconnect(self.file_checked)
        if not filepaths:
            filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, 'Open File', '', 'Data Files (*.dat *.npy *.db)')
        if filepaths:
            for filepath in filepaths:
                try:
                    print(f'Open {filepath}...')
                    filename, extension = os.path.splitext(filepath)
                    if extension == '.npy': # Numpy files (saved session)
                        dataset_list = np.load(filepath, allow_pickle=True)
                        for dataset in dataset_list:
                            try:
                                item = DataItem(NumpyData(filepath, self.canvas, dataset))
                                self.file_list.addItem(item)
                            except Exception as e:
                                print(f'Failed to add NumPy dataset '
                                      f'{dataset["File Name"]}...', e)
                    
                    elif extension == '.db': # QCoDeS files
                        if qcodes_imported:
                            qc.initialise_or_create_database_at(filepath)
                            datasets = qc.load_last_experiment().data_sets()
                            for dataset in datasets:
                                try:
                                    item = DataItem(qcodes_extension.QCodesData(filepath, self.canvas, dataset))
                                    self.file_list.addItem(item)
                                except Exception as e:
                                    print(f'Failed to add QCoDes dataset '
                                          f'#{dataset.captured_run_id}...', e)
                        else:
                            print('QCoDeS module not imported!')
                    
                    elif (os.path.basename(filepath) == 'data.dat' and # Matlab qd files
                          os.path.isfile(os.path.dirname(filepath)+'/meta.json')):
                        metapath = os.path.dirname(filepath)+'/meta.json'
                        item = DataItem(qd_extension.QdData(filepath, self.canvas, metapath))
                        self.file_list.addItem(item)
                    
                    else: # bare column-based data file
                        item = DataItem(BaseClassData(filepath, self.canvas))
                        self.file_list.addItem(item)
                except Exception as e:
                    print(f'Failed to open {filepath}...', e)
                    raise

            if self.file_list.count() > 0:
                last_item = self.file_list.item(self.file_list.count()-1)
                self.file_list.setCurrentItem(last_item)
                for item_index in range(self.file_list.count()-1):
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
                last_item.setCheckState(QtCore.Qt.Checked)
                self.file_checked(last_item)
        self.file_list.itemChanged.connect(self.file_checked)
    
    def remove_files(self, which='current'):
        update_plots = False
        if self.file_list.count() > 0:
            if which == 'current':
                items = [self.file_list.currentItem()]
            elif which == 'all':
                items = [self.file_list.item(n) for n in range(self.file_list.count())]
            for item in items: 
                if (item.data.filepath in self.linked_files 
                    and not hasattr(item, 'duplicate')):
                    self.linked_files.remove(item.data.filepath)
                if item.checkState() == 2:
                    update_plots = True
                index = self.file_list.row(item)
                self.file_list.takeItem(index)
                del item
        self.show_current_all()
        if update_plots: 
            self.update_plots()
    
    def file_checked(self, item):
        if item.checkState() == 2:
            self.file_list.setCurrentItem(item)
        self.update_plots()
    
    def file_clicked(self):
        self.show_current_all()
            
    def file_double_clicked(self, item):
        self.file_list.itemChanged.disconnect(self.file_checked)
        for item_index in range(self.file_list.count()):
            self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
        item.setCheckState(QtCore.Qt.Checked)
        self.file_list.itemChanged.connect(self.file_checked)
        self.update_plots()
    
    def update_plots(self, update_data=True):
        self.figure.clear()
        checked_items = self.get_checked_items()
        if checked_items:
            rows, cols = self.subplot_grid[len(checked_items)-1]
            for index, item in enumerate(checked_items):
                try:
                    if update_data:
                        item.data.prepare_data_for_plot()
                    item.data.figure = self.figure
                    item.data.axes = item.data.figure.add_subplot(rows, cols, index+1)
                    item.data.add_plot(dim=len(item.data.get_columns()))
                    if hasattr(item.data, 'linecut_window'):
                        item.data.linecut_window.update()
                    if hasattr(item.data, 'multiple_linecuts_window'):
                        if item.data.multiple_linecuts_window.isVisible():
                            item.data.multiple_linecuts_window.update()
                except Exception as e:
                    print(f'Could not plot {item.data.filepath}...', e)
                    raise
        self.show_current_all()
        self.canvas.draw()
          
    def refresh_files(self):
        current_item = self.file_list.currentItem()
        if current_item:
            current_item.data.prepare_data_for_plot(reload=True)
            self.update_plots()
        if self.linked_folder:
            old_number_of_items = self.file_list.count()
            self.update_link_to_folder(new_folder=False)
            if self.file_list.count() > old_number_of_items:
                last_item = self.file_list.item(self.file_list.count()-1)
                self.file_double_clicked(last_item)
                self.file_list.setCurrentItem(last_item)
                self.track_button_clicked() # TODO: is this the desired behavior?
            
    def to_next_file(self):
        checked_items, indices = self.get_checked_items(return_indices=True)
        if (len(checked_items) == 1 and self.file_list.count() > 1 and 
            indices[0]+1 < self.file_list.count()):
            item = checked_items[0]
            next_item = self.file_list.item(indices[1]+1)
            self.file_list.itemChanged.disconnect(self.file_checked)
            item.setCheckState(QtCore.Qt.Unchecked)
            next_item.setCheckState(QtCore.Qt.Checked)
            self.file_list.setCurrentItem(next_item)
            self.file_list.itemChanged.connect(self.file_checked)
            self.update_plots()
        
    def to_previous_file(self):
        checked_items, indices = self.get_checked_items(return_indices=True)
        if (len(checked_items) == 1 and self.file_list.count() > 1 
            and indices[0] > 0):
            item = checked_items[0]
            previous_item = self.file_list.item(indices[0]-1)
            self.file_list.itemChanged.disconnect(self.file_checked)
            item.setCheckState(QtCore.Qt.Unchecked)
            previous_item.setCheckState(QtCore.Qt.Checked)
            self.file_list.setCurrentItem(previous_item)
            self.file_list.itemChanged.connect(self.file_checked)
            self.update_plots()
            
    def get_checked_items(self, return_indices = False):
        indices = [index for index in range(self.file_list.count()) 
                   if self.file_list.item(index).checkState() == 2]
        checked_items = [self.file_list.item(index) for index in indices]
        if return_indices:    
            return checked_items, indices
        else:
            return checked_items
        
    def track_button_clicked(self):
        if self.track_button.text() == 'Track':
            last_item = self.file_list.item(self.file_list.count()-1)
            if len(last_item.data.get_columns()) == 2: # if file is 2D
                self.start_auto_refresh(AUTO_REFRESH_INTERVAL_2D)
            else: # if file is 3D
                self.start_auto_refresh(AUTO_REFRESH_INTERVAL_3D)
        elif self.track_button.text() == 'Stop':
            self.stop_auto_refresh()
        
    def start_auto_refresh(self, time_interval):
        self.track_button.setText('Stop')
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
        checked_items = self.get_checked_items()       
        # Refresh all checked items
        for item in checked_items:
            item.data.prepare_data_for_plot()
        self.update_plots()
        self.window_title_auto_refresh = ' - Auto-Refreshing Enabled'
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)        
        
        # Update progress of last item (if checked)
        last_item = self.file_list.item(self.file_list.count()-1)
        if last_item.checkState() == 2 and last_item.data.last_modified_time:
            if (datetime.now().timestamp() - 
                last_item.data.last_modified_time > 600 
                or last_item.data.progress_fraction == 1):
                print('Stop auto refresh...')
                last_item.data.remaining_time_string = ''
                self.stop_auto_refresh()
            else:
                self.remaining_time_label.setText(last_item.data.remaining_time_string)
        else:
            self.remaining_time_label.setText('')
            
    def stop_auto_refresh(self):
        self.track_button.setText('Track')
        self.auto_refresh_timer.stop()
        self.action_refresh_stop.setEnabled(False)
        self.window_title_auto_refresh = ''
        self.setWindowTitle(self.window_title+self.window_title_auto_refresh)
        self.remaining_time_label.setText('')
        
    def move_file(self, direction):
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
                if (current_item.checkState() == 2 and 
                    self.file_list.item(new_row).checkState() == 2):
                    update_canvas = True
                else:
                    update_canvas = False
                self.file_list.takeItem(current_row)
                self.file_list.insertItem(new_row, current_item)
                self.file_list.setCurrentRow(new_row)
                if update_canvas:
                    self.update_plots()
        
    def show_current_all(self):
        self.show_current_plot_settings()
        self.show_current_view_settings()
        self.show_current_filters()
    
    def show_current_plot_settings(self):
        current_item = self.file_list.currentItem()
        if current_item:
            self.settings_table.itemChanged.disconnect(self.plot_setting_edited)
            self.settings_table.setRowCount(0)
            settings = current_item.data.settings
            for key, value in list(settings.items()):
                row = self.settings_table.rowCount()
                self.settings_table.insertRow(row)
                property_item = QtWidgets.QTableWidgetItem(key)
                property_item.setFlags(QtCore.Qt.ItemIsSelectable | 
                                       QtCore.Qt.ItemIsEnabled)
                self.settings_table.setItem(row, 0, property_item)
                self.settings_table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
            self.settings_table.itemChanged.connect(self.plot_setting_edited)
            
    def show_current_view_settings(self):
        current_item = self.file_list.currentItem()
        if current_item:
            settings = current_item.data.view_settings
            self.min_line_edit.setText(f'{settings["Minimum"]:.4g}')
            self.max_line_edit.setText(f'{settings["Maximum"]:.4g}')
            self.mid_line_edit.setText(f'{settings["Midpoint"]:.4g}')
            if settings['Locked']:
                self.lock_checkbox.setCheckState(QtCore.Qt.Checked)
            else:
                self.lock_checkbox.setCheckState(QtCore.Qt.Unchecked)
            if settings['MidLock']:
                self.mid_checkbox.setCheckState(QtCore.Qt.Checked)
            else:
                self.mid_checkbox.setCheckState(QtCore.Qt.Unchecked)
            self.colormap_type_box.currentIndexChanged.disconnect(self.colormap_type_edited)
            self.colormap_type_box.setCurrentText(settings['Colormap Type'])
            self.colormap_type_box.currentIndexChanged.connect(self.colormap_type_edited)
            self.fill_colormap_box()
            self.colormap_box.currentIndexChanged.disconnect(self.colormap_edited)
            self.colormap_box.setCurrentText(settings['Colormap'])
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
        self.filters_table.setRowCount(0)
        current_item = self.file_list.currentItem()
        if current_item:
            for _ in current_item.data.filters:
                self.append_filter_to_table()
    
    def plot_setting_edited(self):
        current_item = self.file_list.currentItem()
        if current_item:
            current_item.data.old_settings = current_item.data.settings.copy()
            row = self.settings_table.currentRow()
            setting_name = self.settings_table.item(row, 0).text()
            value = self.settings_table.item(row, 1).text()
            current_item.data.settings[setting_name] = value
            self.settings_table.clearFocus()
            try:
                if setting_name == 'columns' or setting_name == 'delimiter':
                    current_item.data.prepare_data_for_plot(reload=True)
                    self.update_plots()           
                elif setting_name == 'linecolor':
                    for line in current_item.data.axes.get_lines():
                        line.set_color(value)
                    self.canvas.draw()
                elif setting_name == 'maskcolor' or setting_name == 'lut':
                    current_item.data.apply_colormap()
                elif (setting_name == 'rasterized' or setting_name == 'colorbar'
                      or setting_name == 'minorticks'):
                    self.update_plots()
                current_item.data.extension_setting_edited(self, setting_name)
                current_item.data.apply_plot_settings()
                self.canvas.draw()
            except Exception as e: # if invalid value is typed: reset to previous settings
                print('Invalid value of plot setting!', e)
                self.paste_plot_settings(which='old')
    
    def view_setting_edited(self, edited_setting):
        current_item = self.file_list.currentItem()
        view_settings = current_item.data.view_settings
        current_item.data.old_view_settings = view_settings.copy()
        if current_item:
            try:
                if edited_setting == 'Minimum' or edited_setting == 'Maximum':
                    if edited_setting == 'Minimum':
                        text_box = self.min_line_edit
                    else:
                        text_box = self.max_line_edit
                    new_value = float(text_box.text())
                    view_settings[edited_setting] = new_value
                    text_box.setText(f'{new_value:.4g}')
                    text_box.clearFocus()
                    current_item.data.reset_midpoint()
                    self.mid_line_edit.setText(f'{view_settings["Midpoint"]:.4g}')
                elif edited_setting == 'Midpoint':
                    if self.mid_line_edit.text():
                        new_value = float(self.mid_line_edit.text())
                        view_settings[edited_setting] = new_value
                    else:
                        current_item.data.reset_midpoint()
                        new_value = view_settings[edited_setting]
                    self.mid_line_edit.setText(f'{new_value:.4g}')
                    self.mid_line_edit.clearFocus()
                elif edited_setting == 'Locked':
                    if self.lock_checkbox.isChecked():
                        view_settings[edited_setting] = True
                    else:
                        view_settings[edited_setting] = False
                elif edited_setting == 'MidLock':
                    if self.mid_checkbox.isChecked():
                        view_settings[edited_setting] = True
                    else:
                        view_settings[edited_setting] = False
                current_item.data.apply_view_settings()
                self.canvas.draw()
            except Exception as e:
                print('Invalid value of view setting!', e)
                self.paste_view_settings(which='old')
                
    def fill_colormap_box(self):
        self.colormap_box.currentIndexChanged.disconnect(self.colormap_edited)
        self.colormap_box.clear()
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentText()])
        self.colormap_box.currentIndexChanged.connect(self.colormap_edited)
    
    def colormap_type_edited(self):
        self.fill_colormap_box()
        self.colormap_edited()
        
    def colormap_edited(self):
        current_item = self.file_list.currentItem()
        if current_item:
            settings = current_item.data.view_settings
            settings['Colormap Type'] = self.colormap_type_box.currentText()
            settings['Colormap'] = self.colormap_box.currentText()
            settings['Reverse'] = self.reverse_colors_box.isChecked()
            if current_item.checkState():
                current_item.data.apply_colormap()
                self.canvas.draw()
    
    def filters_table_edited(self, item):
        current_item = self.file_list.currentItem()
        current_item.data.old_filters = copy.deepcopy(current_item.data.filters)
        if current_item:   
            try:
                row = item.row()
                filt = current_item.data.filters[row]
                filter_item = self.filters_table.item(row, 0)
                filt.method = self.filters_table.cellWidget(row, 1).currentText()
                filt.settings = [self.filters_table.item(row, 2).text(), 
                                 self.filters_table.item(row, 3).text()]
                filt.checkstate = filter_item.checkState()
                self.filters_table.clearFocus()
                current_item.data.apply_all_filters()
                current_item.data.reset_view_settings()
                if current_item.checkState():
                    self.update_plots()
                    self.show_current_filters()
                    self.show_current_view_settings()
            except Exception as e:
                print('Invalid value of filter!', e)
                self.paste_filters(which='old')
    
    def copy_plot_settings(self):
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_settings = current_item.data.settings.copy()
    
    def copy_filters(self):
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_filters = copy.deepcopy(current_item.data.filters)
            
    def copy_view_settings(self):
        current_item = self.file_list.currentItem()
        if current_item:
            self.copied_view_settings = current_item.data.view_settings.copy()
    
    def paste_plot_settings(self, which='copied'):
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'copied':
                if self.copied_settings:
                    current_item.data.settings = self.copied_settings.copy()
            elif which == 'default':
                current_item.data.settings = current_item.data.DEFAULT_PLOT_SETTINGS.copy()
            elif which == 'old':
                current_item.data.settings = current_item.data.old_settings.copy()
            self.show_current_plot_settings()
            if current_item.checkState():
                current_item.data.apply_plot_settings()
                self.canvas.draw()
    
    def paste_filters(self, which='copied'):
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'copied':
                if self.copied_filters:
                    current_item.data.filters = copy.deepcopy(self.copied_filters)
            elif which == 'old':
                current_item.data.filters = copy.deepcopy(current_item.data.old_filters)
            self.show_current_filters()
            current_item.data.apply_all_filters()
            self.show_current_view_settings()
            if current_item.checkState():
                self.update_plots()
                self.canvas.draw()

    def paste_view_settings(self, which='copied'):
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'copied':
                if self.copied_view_settings:
                    current_item.data.view_settings = self.copied_view_settings.copy()
            elif which == 'old':
                current_item.data.view_settings = current_item.data.old_view_settings.copy()
            self.show_current_view_settings()
            if current_item.checkState():
                current_item.data.apply_view_settings()
                current_item.data.apply_colormap()
                self.canvas.draw()

    def open_item_menu(self):
        current_item = self.file_list.currentItem()
        if current_item:
            checked_items = self.get_checked_items()
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
        current_item = self.file_list.currentItem()
        if current_item:
            if signal.text() == 'Duplicate...':
                self.open_files(filepaths=[current_item.data.filepath])
                item = self.file_list.currentItem()
                item.data.label = f'*{item.data.label}'
                item.setText(item.data.label)
                item.duplicate = True
                item.data.settings = current_item.data.settings.copy()
                item.data.view_settings = current_item.data.view_settings.copy()
                item.data.filters = copy.deepcopy(current_item.data.filters)              
                self.update_plots()
            elif signal.text() == 'Check all...':
                self.file_list.itemChanged.disconnect(self.file_checked)
                for item_index in range(self.file_list.count()):                        
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Checked)
                    self.file_list.itemChanged.connect(self.file_checked)
                self.update_plots()
            elif signal.text() == 'Combine plots...':
                try:
                    self.combine_plots()
                except Exception as e:
                    print('Cannot combine these plots...', e)
                
    def combine_plots(self): # TODO fix
        checked_items = self.get_checked_items()
        if checked_items:
            title = '[combined] '
            data_list = []
            for item in checked_items:
                split_text = item.text().split(' ')
                title += f'{split_text[0]} {split_text[1]}'
                data_list.append(item.data)
            self.multi_plot_window = LineCutWindow(parent=data_list, multiple=True)
            self.multi_plot_window.title = title
            self.multi_plot_window.xlabel = data_list[0].settings['xlabel']
            self.multi_plot_window.ylabel = data_list[0].settings['ylabel']
            self.multi_plot_window.zlabel = None
            self.multi_plot_window.draw_plot()
            self.multi_plot_window.show()
    
    def open_plot_settings_menu(self):
        row = self.settings_table.currentRow()
        column = self.settings_table.currentColumn()
        if column == 1:
            setting_name = self.settings_table.item(row, 0).text()
            menu = QtWidgets.QMenu(self)
            settings = SETTINGS_MENU_OPTIONS.copy()
            current_item = self.file_list.currentItem()
            if current_item and hasattr(current_item.data, 'settings_menu_options'):
                settings.update(current_item.data.settings_menu_options)
            if setting_name in settings.keys():
                for entry in settings[setting_name]:
                    action = QtWidgets.QAction(entry, self)
                    menu.addAction(action)
                menu.triggered[QtWidgets.QAction].connect(self.replace_plot_setting)
                menu.popup(QtGui.QCursor.pos())
   
    def replace_plot_setting(self, signal):
        item = self.settings_table.currentItem()
        item.setText(signal.text())

    def reset_color_limits(self):
        current_item = self.file_list.currentItem()
        if current_item:
            current_item.data.reset_view_settings(overrule=True)
            self.show_current_view_settings()
            if current_item.checkState():
                current_item.data.apply_view_settings()
                self.canvas.draw()
    
    def filters_box_changed(self):
        current_item = self.file_list.currentItem()
        if current_item:
            filt = Filter(self.filters_combobox.currentText())
            current_item.data.filters.append(filt)
            if current_item.checkState() and filt.checkstate:
                self.update_plots()
            else:
                self.append_filter_to_table()
        self.filters_combobox.currentIndexChanged.disconnect(self.filters_box_changed)
        self.filters_combobox.setCurrentIndex(0)
        self.filters_combobox.clearFocus()
        self.filters_combobox.currentIndexChanged.connect(self.filters_box_changed)
        
    
    def append_filter_to_table(self):
        current_item = self.file_list.currentItem()
        if current_item:
            row = self.filters_table.rowCount()
            filt = current_item.data.filters[row]
            self.filters_table.itemChanged.disconnect(self.filters_table_edited)
            self.filters_table.insertRow(row) 
            filter_item = QtWidgets.QTableWidgetItem(filt.name)
            filter_item.setFlags(QtCore.Qt.ItemIsSelectable | 
                                 QtCore.Qt.ItemIsEnabled | 
                                 QtCore.Qt.ItemIsUserCheckable)
            filter_item.setText(filt.name)
            filter_item.setCheckState(filt.checkstate)
            method_box = NoScrollQComboBox()
            method_box.addItems(filt.method_list)
            method_box.setCurrentIndex(filt.method_list.index(filt.method))
            setting_1 = QtWidgets.QTableWidgetItem(filt.settings[0])
            setting_2 = QtWidgets.QTableWidgetItem(filt.settings[1])
            method_box.currentIndexChanged.connect(lambda: self.filters_table_edited(setting_1))
            self.filters_table.setItem(row, 0, filter_item)
            self.filters_table.setCellWidget(row, 1, method_box)
            self.filters_table.setItem(row, 2, setting_1)
            self.filters_table.setItem(row, 3, setting_2)
            self.filters_table.item(row, 2).setTextAlignment(int(QtCore.Qt.AlignRight) | 
                                                             int(QtCore.Qt.AlignVCenter))
            self.filters_table.item(row, 3).setTextAlignment(int(QtCore.Qt.AlignRight) | 
                                                             int(QtCore.Qt.AlignVCenter))
            self.filters_table.setCurrentCell(row, 0)
            self.filters_table.itemChanged.connect(self.filters_table_edited)
    
    def remove_filters(self, which='current'):
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'current':
                filter_row = self.filters_table.currentRow()
                if filter_row != -1:
                    self.filters_table.removeRow(filter_row)
                    del current_item.data.filters[filter_row]
                    current_item.data.apply_all_filters()
                    current_item.data.reset_view_settings()
            elif which == 'all':
                self.filters_table.setRowCount(0)
                current_item.data.filters = []
                current_item.data.apply_all_filters()
                current_item.data.reset_view_settings()
            if current_item.checkState():
                current_item.data.apply_view_settings()
                self.update_plots()
                self.show_current_view_settings()
     
    def move_filter(self, to):
        current_item = self.file_list.currentItem()
        if current_item:
            filters = current_item.data.filters
            row = self.filters_table.currentRow()
            if ((row > 0 and to == -1) or
                (row < self.filters_table.rowCount()-1 and to == 1)):
                filters[row], filters[row+to] = filters[row+to], filters[row]
                self.show_current_filters()
                self.filters_table.setCurrentCell(row+to, 0)
                if (self.filters_table.item(row,0).checkState() and 
                    self.filters_table.item(row+to,0).checkState()):
                    current_item.data.apply_all_filters()
                    self.update_plots()
                    self.show_current_view_settings()

    def save_image(self):
        current_item = self.file_list.currentItem()
        if current_item:
            data_name, _ = os.path.splitext(current_item.data.filepath)
            formats = 'Adobe Acrobat (*.pdf);;Portable Network Graphic (*.png)'
            filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Figure As...', data_name.replace(':',''), formats)
            if filename:
                print('Save Figure as ', filename)                    
                if current_item.data.settings['dpi'] == 'figure':
                    dpi = 'figure'
                else:
                    dpi = current_item.data.settings['dpi'] 
                if DARK_THEME and qdarkstyle_imported:             
                    rcParams_to_light_theme()
                    self.update_plots(update_data=False)
                transparent = current_item.data.settings['transparent']=='True'
                self.figure.savefig(filename, dpi=dpi, transparent=transparent,
                                    bbox_inches='tight')
                if DARK_THEME and qdarkstyle_imported:
                    rcParams_to_dark_theme()
                    self.update_plots(update_data=False)
                print('Saved!')   
           
    def save_filters(self):
        current_item = self.file_list.currentItem()
        if current_item:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Filters As...', '', '.npy')
            np.save(filename, current_item.data.filters)
            
    def load_filters(self):
        current_item = self.file_list.currentItem()
        if current_item:
            filename, _ = QtWidgets.QFileDialog.getOpenFileNames(
                    self, 'Open Filters File...', '', '*.npy')
            loaded_filters = np.load(filename[0], allow_pickle=True)
            current_item.data.filters = copy.deepcopy(loaded_filters)
            self.show_current_filters()
            current_item.data.apply_all_filters()
            self.update_plots()
            self.show_current_view_settings()
    
    def save_session(self, which='current'):
        current_item = self.file_list.currentItem()
        if current_item:
            if which == 'current':
                suggested_filename = os.path.splitext(current_item.data.filepath)[0].replace(':','')
                filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', suggested_filename, '*.npy')
                items = [current_item]
            elif which == 'all':
                filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', '', '*.npy')
                items = [self.file_list.item(n) for n in range(self.file_list.count())]
            elif which == 'checked':
                filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', '', '*.npy')
                items = [self.file_list.item(n) for n in range(self.file_list.count()) 
                         if self.file_list.item(n).checkState() == 2]                             
            dictionary_list = []
            for item in items:
                item_dictionary = {'Label': item.data.label, 
                                   'File Path': filepath,
                                   'Settings': item.data.settings, 
                                   'Filters': item.data.filters, 
                                   'View Settings': item.data.view_settings, 
                                   'Raw Data': item.data.raw_data}
                dictionary_list.append(item_dictionary)
            np.save(filepath, dictionary_list)
            print('Saved!')
    
    def open_files_from_folder(self): 			
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
        self.open_files(filepaths)
        
    def update_link_to_folder(self, new_folder=True):
        if new_folder:
            self.linked_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.linked_folder:
            self.window_title = f'Inspectra Gadget - Linked to folder {self.linked_folder}'
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
                self.file_list.di
                for new_file in new_files:
                    try:
                        print('Open', new_file[1])
                        self.add_file(new_file[1], do_load_data=False)
                        self.linked_files.append(new_file[1])
                    except Exception as e:
                        print('Could not open', new_file[1], e)
                        continue
                if new_folder:
                    self.file_list.itemChanged.disconnect(self.file_checked)
                    for index in range(self.file_list.count()):
                        self.file_list.item(index).setCheckState(QtCore.Qt.Unchecked)
                    last_item = self.file_list.item(self.file_list.count()-1)
                    self.file_list.setCurrentItem(last_item)
                    last_item.setCheckState(QtCore.Qt.Checked)
                    self.file_list.itemChanged.connect(self.file_checked)
                    self.file_checked(last_item)
                    
    def unlink_folder(self):
        if self.linked_folder:
            self.linked_folder = None
            self.window_title = 'Inspectra Gadget'
            self.setWindowTitle(self.window_title+
                                self.window_title_auto_refresh)
        
    def mouse_click_canvas(self, event):
        if self.navi_toolbar.mode == '': # If not using the navigation toolbar tools
            if event.inaxes:
                x, y = event.xdata, event.ydata
                self.plot_in_focus = [self.file_list.item(index) for index 
                                      in range(self.file_list.count()) 
                                      if self.file_list.item(index).data.axes == event.inaxes]
                if self.plot_in_focus:
                    data = self.plot_in_focus[0].data
                    data.selected_x, data.selected_y = x, y
                    
                    if ((event.button == 1 or event.button == 2) and 
                        len(data.get_columns()) == 3 and 
                        not hasattr(data, 'linecut_points')):
                        index_x = np.argmin(np.abs(data.processed_data[0][:,0]-x))
                        index_y = np.argmin(np.abs(data.processed_data[1][0,:]-y))
                        data.selected_indices = [int(index_x), int(index_y)]
                        if not hasattr(data, 'linecut_window'):
                            data.linecut_window = LineCutWindow(data)
                        if event.button == 1:
                            data.linecut_window.orientation = 'horizontal'
                        elif event.button == 2:
                            data.linecut_window.orientation = 'vertical'
                        data.linecut_window.running = True
                        data.linecut_window.update()
                        self.canvas.draw()
                        data.linecut_window.activateWindow()
                        
                    elif event.button == 3:
                        rightclick_menu = QtWidgets.QMenu(self)
                        if len(data.get_columns()) == 3:
                            index_x = np.argmin(np.abs(data.processed_data[0][:,0]-x))
                            index_y = np.argmin(np.abs(data.processed_data[1][0,:]-y))
                            z = data.processed_data[2][index_x,index_y]
                            coordinates = (f'x = {x:.4g}, y = {y:.4g}, z = {z:.4g}'
                                           f' ({index_x}, {index_y})')
                            action = QtWidgets.QAction(coordinates, self)
                            action.setEnabled(False)
                            rightclick_menu.addAction(action)
                            rightclick_menu.addSeparator()
                            
                        # Add actions from extension modules
                        data.add_extension_actions(self, rightclick_menu)

                        if len(data.get_columns()) == 3:
                            actions = []
                            actions.append(QtWidgets.QAction('Draw diagonal linecut...', self))
                            #data.linecut_from = [x, y]
                            actions.append(QtWidgets.QAction('Draw circular linecut...', self))
                            actions.append(QtWidgets.QAction('Plot vertical linecuts...', self))
                            actions.append(QtWidgets.QAction('Plot horizontal linecuts...', self))
                            actions.append(QtWidgets.QAction('FFT vertical...', self))
                            actions.append(QtWidgets.QAction('FFT horizontal...', self))
                        for action in actions:
                            rightclick_menu.addAction(action)
                        rightclick_menu.triggered[QtWidgets.QAction].connect(self.popup_canvas)
                        rightclick_menu.popup(QtGui.QCursor.pos())
                
                else: # if colorbar in focus
                    checked_items = self.get_checked_items()
                    self.cbar_in_focus = [checked_item for checked_item in checked_items
                                          if checked_item.data.cbar.ax == event.inaxes]
                    if self.cbar_in_focus:
                        data = self.cbar_in_focus[0].data
                        if event.button == 1:
                            data.view_settings['Minimum'] = y
                            data.reset_midpoint()
                        elif event.button == 2:
                            data.view_settings['Midpoint'] = y
                        elif event.button == 3:
                            data.view_settings['Maximum'] = y
                            data.reset_midpoint()
                        data.apply_view_settings()
                        self.canvas.draw()
                        self.show_current_view_settings()
    
    def popup_canvas(self, signal):
        data = self.plot_in_focus[0].data
        if (signal.text() == 'Plot horizontal linecuts...' or
            signal.text() == 'Plot vertical linecuts...'):
            data.multi_linecuts_window = MultipleLineCutsWindow(data) 
            if signal.text() == 'Plot horizontal linecuts...':
                data.multi_linecuts_window.orientation = 'horizontal'
            elif signal.text() == 'Plot vertical linecuts...':
                data.multi_linecuts_window.orientation = 'vertical'
            data.multi_linecuts_window.update()
        elif signal.text() == 'Draw diagonal linecut...':
            if hasattr(data, 'linecut_points'):
                data.hide_linecuts()
            x, y = data.selected_x, data.selected_y
            data.linecut_points = [DraggablePoint(data, x, y)]            
            left, right = data.axes.get_xlim() 
            bottom, top = data.axes.get_ylim()
            x_mid, y_mid = 0.5*(left+right), 0.5*(top+bottom)
            data.linecut_points.append(DraggablePoint(data, x_mid, y_mid, 
                                                      draw_line=True))
            if not hasattr(data, 'linecut_window'):
                data.linecut_window = LineCutWindow(data)
            data.linecut_window.orientation = 'diagonal'
            data.linecut_window.running = True
            data.linecut_window.update()
            self.canvas.draw()
            data.linecut_window.activateWindow()
        elif signal.text() == 'Draw circular linecut...':
            if hasattr(data, 'linecut_points'):
                data.hide_linecuts()
            left, right = data.axes.get_xlim() 
            bottom, top = data.axes.get_ylim()                
            x, y = data.selected_x, data.selected_y
            data.xr, data.yr = 0.1*(right-left), 0.1*(top-bottom)
            data.linecut_points = [DraggablePoint(data, x, y),
                                   DraggablePoint(data, x+data.xr, y)]
            data.linecut_points.append(DraggablePoint(data, x, y+data.yr,
                                                      draw_circle=True))
            if not hasattr(data, 'linecut_window'):
                data.linecut_window = LineCutWindow(data)
            data.linecut_window.running = True
            data.linecut_window.orientation = 'circular'
            data.linecut_window.update()
            self.canvas.draw()
            data.linecut_window.activateWindow()
        elif signal.text() == 'FFT vertical...':
            data.fft_orientation = 'vertical'
            data.open_fft_window()
        elif signal.text() == 'FFT horizontal...':
            data.fft_orientation = 'horizontal'
            data.open_fft_window()
        else:
            data.do_extension_actions(self, signal)
            
    def copy_canvas_to_clipboard(self):
        checked_items = self.get_checked_items()
        for item in checked_items:
            item.data.cursor.horizOn = False
            item.data.cursor.vertOn = False            
        self.canvas.draw()
        if DARK_THEME and qdarkstyle_imported:
            rcParams_to_light_theme()
            self.update_plots(update_data=False)
        buf = io.BytesIO()
        if item.data.settings['dpi'] == 'figure':
            dpi = 'figure'
        else:
            dpi = int(item.data.settings['dpi'])
        self.figure.savefig(buf, dpi=dpi, bbox_inches='tight')
        QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
        buf.close()
        for item in checked_items:
            item.data.cursor.horizOn = True
            item.data.cursor.vertOn = True                       
        self.canvas.draw()
        if DARK_THEME and qdarkstyle_imported:
            rcParams_to_dark_theme()
            self.update_plots(update_data=False)
            
    def mouse_scroll_canvas(self, event):
        if event.inaxes:
            y = event.ydata
            self.plot_in_focus = [self.file_list.item(index) for index 
                                  in range(self.file_list.count()) 
                                  if self.file_list.item(index).data.axes == event.inaxes]
            if self.plot_in_focus:
                data = self.plot_in_focus[0].data
                if len(data.get_columns()) == 3:
                    if hasattr(data, 'linecut_window'):
                        data_shape = data.processed_data[0].shape
                        if data.linecut_window.orientation == 'horizontal':
                            new_index = data.selected_indices[1]+int(event.step)
                            if new_index >= 0 and new_index < data_shape[1]:
                                data.selected_indices[1] = new_index
                        elif data.linecut_window.orientation == 'vertical':
                            new_index = data.selected_indices[0]+int(event.step)
                            if new_index >= 0 and new_index < data_shape[0]:
                                data.selected_indices[0] = new_index
                        data.linecut_window.update()
                        self.canvas.draw()
            else:
                checked_items = self.get_checked_items()
                self.cbar_in_focus = [checked_item for checked_item in checked_items
                                      if checked_item.data.cbar.ax == event.inaxes]
                if self.cbar_in_focus:
                    data = self.cbar_in_focus[0].data
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
            speed = 0.03
            lb, rb, tb, bb = 0.15*width, 0.85*width, 0.85*height, 0.15*height
            if (event.x < lb and event.y > bb and event.y < tb):
                if (event.step > 0 or 
                    (event.step < 0 and self.figure.subplotpars.left > 0.07)):
                    self.figure.subplots_adjust(left=(1+speed*event.step)*
                                                self.figure.subplotpars.left)
            elif (event.x > rb and event.y > bb and event.y < tb):
                  if (event.step < 0 or 
                      (event.step > 0 and self.figure.subplotpars.right < 0.97)):
                      self.figure.subplots_adjust(right=(1+speed*0.5*event.step)*
                                                  self.figure.subplotpars.right)
            elif (event.y < bb and event.x > lb and event.x < rb):
                  if (event.step > 0 or 
                      (event.step < 0 and self.figure.subplotpars.bottom > 0.07)):
                      self.figure.subplots_adjust(bottom=(1+speed*event.step)*
                                                  self.figure.subplotpars.bottom)
            elif (event.y > tb and event.x > lb and event.x < rb):
                  if (event.step < 0 or 
                      (event.step > 0 and self.figure.subplotpars.top < 0.94)):
                      self.figure.subplots_adjust(top=(1+speed*0.5*event.step)*
                                                  self.figure.subplotpars.top)
            else:
                self.figure.subplots_adjust(wspace=(1+speed*event.step)*self.figure.subplotpars.wspace)
            self.canvas.draw()
            
    def keyPressEvent(self, event): 
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            self.copy_canvas_to_clipboard()
        elif event.key() == QtCore.Qt.Key_T and event.modifiers() == QtCore.Qt.ControlModifier:
            if not self.action_refresh_stop.isEnabled():
                self.start_auto_refresh(1)
                print('Start live tracking...')
            else:
                self.stop_auto_refresh()
                print('Stop live tracking...')
            
    def merge_files(self, raw_data=True): # TODO improve
        try:
            checked_items = self.get_checked_items()
            if len(checked_items) > 1:
                filepaths = [item.data.filepath for item in checked_items]
                time_stamps = [item.data.label.split('_')[0] for item in checked_items]
                device_name = checked_items[0].data.label.split('_')[1] # TODO make more general
                measurement_name = checked_items[0].data.label.split('_')[2]
                output_name = '_'.join(time_stamps)+'_'+device_name+'_'+measurement_name+'_merged.dat'
                output_dir = os.path.dirname(checked_items[0].data.filepath)
                output_filepath = output_dir+'/'+output_name
                if raw_data:
                    with open(output_filepath,'w') as out_file:
                        for filepath in filepaths:
                            with open(filepath) as in_file:
                                for line in in_file:
                                    out_file.write(line)
                else:
                    data = [item.data.processed_data for item in checked_items]
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
                    data0 = checked_items[0].data
                    view_settings = {
                            'Minimum': np.min(g_data), 'Maximum': np.max(g_data), 
                            'Midpoint': 0.5*(np.min(g_data)+np.max(g_data)),
                            'Colormap': 'magma', 'Colormap Type': 'Uniform',
                            'Locked': 0, 'MidLock': 0, 'Reverse': 2}
                    dictionary_list = [{'File Name': output_name, 'Settings': data0.default_settings,
                                      'Filters': [], 'View Settings': view_settings,
                                      'Raw Data': [x_g,y_g,g_data]}]
                    np.save(output_filepath, dictionary_list)
                print('Merged files into file '+output_filepath)
        except Exception as e:
            print('Cannot merge these files...', e)
            
            
    def apply_preset(self, preset_number): # TODO improve
        checked_items = self.get_checked_items()
        if checked_items:
            for item in checked_items:
                for preset_item in PRESETS[preset_number].items():
                    if preset_item[0] in item.data.settings.keys():
                        item.data.settings[preset_item[0]] = preset_item[1]
                    elif preset_item[0] == 'canvas_bounds':
                        b = preset_item[1] # (left, bottom, right, top)
                        item.data.figure.subplots_adjust(b[0], b[1], b[2], b[3])
                    elif preset_item[0] == 'show_meta_settings':
                        item.data.show_settings = preset_item[1]
            self.update_plots()

class DataItem(QtWidgets.QListWidgetItem):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        self.setCheckState(QtCore.Qt.Unchecked)
        self.setText(self.data.label)

class BaseClassData:    
    # Set default plot settings
    DEFAULT_PLOT_SETTINGS = {}
    DEFAULT_PLOT_SETTINGS['title'] = '<label>'
    DEFAULT_PLOT_SETTINGS['xlabel'] = ''
    DEFAULT_PLOT_SETTINGS['ylabel'] = ''
    DEFAULT_PLOT_SETTINGS['clabel'] = ''
    DEFAULT_PLOT_SETTINGS['titlesize'] = '12'
    DEFAULT_PLOT_SETTINGS['labelsize'] = '14' 
    DEFAULT_PLOT_SETTINGS['ticksize'] = '14'
    DEFAULT_PLOT_SETTINGS['linewidth'] = '1.5'
    DEFAULT_PLOT_SETTINGS['spinewidth'] = '0.8'
    DEFAULT_PLOT_SETTINGS['columns'] = '0,1,2'
    DEFAULT_PLOT_SETTINGS['colorbar'] = 'True'
    DEFAULT_PLOT_SETTINGS['minorticks'] = 'False'
    DEFAULT_PLOT_SETTINGS['delimiter'] = ''
    DEFAULT_PLOT_SETTINGS['linecolor'] = 'black'
    DEFAULT_PLOT_SETTINGS['maskcolor'] = 'black'
    DEFAULT_PLOT_SETTINGS['lut'] = '512'
    DEFAULT_PLOT_SETTINGS['rasterized'] = 'True'
    DEFAULT_PLOT_SETTINGS['dpi'] = '300'
    DEFAULT_PLOT_SETTINGS['transparent'] = 'False'
    
    # Set default view settings
    DEFAULT_VIEW_SETTINGS = {}
    DEFAULT_VIEW_SETTINGS['Minimum'] = 0
    DEFAULT_VIEW_SETTINGS['Maximum'] = 0
    DEFAULT_VIEW_SETTINGS['Midpoint'] = 0
    DEFAULT_VIEW_SETTINGS['Colormap'] = 'magma'
    DEFAULT_VIEW_SETTINGS['Colormap Type'] = 'Uniform'
    DEFAULT_VIEW_SETTINGS['Locked'] = False
    DEFAULT_VIEW_SETTINGS['MidLock'] = False
    DEFAULT_VIEW_SETTINGS['Reverse'] = False  
    
    def __init__(self, filepath, canvas):
        self.filepath = filepath
        self.canvas = canvas
        self.label = os.path.basename(self.filepath)
        
        self.settings = self.DEFAULT_PLOT_SETTINGS.copy()
        self.view_settings = self.DEFAULT_VIEW_SETTINGS.copy()
        self.filters = []

        try: # on Windows
            self.creation_time = os.path.getctime(filepath)
        except Exception:
            try: # on Mac
                self.creation_time = os.stat(filepath).st_birthtime
            except Exception:
                self.creation_time = None
        
    def get_column_data(self):
        column_data = np.genfromtxt(self.filepath, delimiter=self.settings['delimiter'])
        self.measured_data_points = column_data.shape[0]
        return column_data
    
    def get_columns(self):
        return [int(col) for col in self.settings['columns'].split(',')]
    
    def load_and_reshape_data(self):
        column_data = self.get_column_data()
        if column_data.ndim == 1: # if empty array or single-row array
            self.raw_data = None
        else:
            # Determine the number of unique values in the first column to determine the shape of the data
            columns = self.get_columns()
            unique_values, unique_indices = np.unique(column_data[:,columns[0]], 
                                                      return_index=True)
            if len(unique_values) > 1:
                sorted_indices = sorted(unique_indices)
                if len(column_data[sorted_indices[-1]::,0]) < sorted_indices[1]:
                    # Ignore the data from the last sweep if unfinished
                    data_shape = (len(unique_values)-1, sorted_indices[1])
                else:
                    data_shape = (len(unique_values), sorted_indices[1])
            else:
                data_shape = (1, column_data.shape[0])

            if data_shape[0] > 1: # If two or more sweeps are finished
        
                # Check if second column also has unique values at the same 
                # indices as the first column; if yes, skip that column.
                # Relevant for measurements where two parameters are swept simultaneously
                _, next_unique_indices = np.unique(column_data[:,columns[1]], 
                                                   return_index=True)
                if (np.array_equal(unique_indices, next_unique_indices) or
                    np.array_equal(unique_indices, next_unique_indices[::-1])):
                    columns[1] += 1
                    if len(columns) > 2 and columns[1] == columns[2]:
                        columns[2] += 1
                
                # Determine if file is 2D or 3D by checking if first two values in first column are repeated
                if column_data[1,columns[0]] != column_data[0,columns[0]] or len(columns) == 2:
                    self.raw_data = [column_data[:,x] for x in range(column_data.shape[1])]            
                    columns = columns[0,1]
                else: 
                    # flip if first column is sorted from high to low 
                    if unique_values[1] < unique_values[0]: 
                        column_data = np.flipud(column_data)
                    self.raw_data = [np.reshape(column_data[:data_shape[0]*data_shape[1],x], data_shape) 
                                     for x in range(column_data.shape[1])]
                    # flip if second column is sorted from high to low
                    if self.raw_data[1][0,0] > self.raw_data[1][0,1]: 
                        self.raw_data = [np.fliplr(self.raw_data[x]) for x in range(column_data.shape[1])]
                        
            elif data_shape[0] == 1: # if first two sweeps are not finished -> duplicate data of first sweep to enable 3D plotting
                self.raw_data = [np.tile(column_data[:data_shape[1],x], (2,1)) for x in range(column_data.shape[1])]    
                if len(unique_values) > 1: # if first sweep is finished -> set second x-column to second x-value
                    self.raw_data[columns[0]][0,:] = unique_values[0]
                    self.raw_data[columns[0]][1,:] = unique_values[1]
                else: # if first sweep is not finished -> set duplicate x-columns to +1 and -1 of actual value
                    self.raw_data[columns[0]][0,:] = unique_values[0]-1
                    self.raw_data[columns[0]][1,:] = unique_values[0]+1
            self.settings['columns'] = ','.join([str(i) for i in columns])
                   
    def copy_raw_to_processed_data(self):
        self.processed_data = [np.copy(self.raw_data[x]) for x in self.get_columns()]

    def prepare_data_for_plot(self, reload=False):
        if not hasattr(self, 'raw_data') or reload:
            self.load_and_reshape_data()
        if self.raw_data:
            self.copy_raw_to_processed_data()
            self.apply_all_filters()
        else:
            self.processed_data = None

    def add_plot(self, dim):
        if self.processed_data:
            cmap_str = self.view_settings['Colormap']
            if self.view_settings['Reverse']:
                cmap_str += '_r'
            cmap = cm.get_cmap(cmap_str, lut=int(self.settings['lut']))
            cmap.set_bad(self.settings['maskcolor'])
            if dim == 2:
                self.image = self.axes.plot(self.processed_data[0], 
                                            self.processed_data[1], color=cmap(0.5))
            elif dim == 3:
                norm = MidpointNormalize(vmin=self.view_settings['Minimum'], 
                                         vmax=self.view_settings['Maximum'], 
                                         midpoint=self.view_settings['Midpoint'])
                self.image = self.axes.pcolormesh(self.processed_data[0], 
                                                  self.processed_data[1], 
                                                  self.processed_data[2], 
                                                  shading='auto', norm=norm, cmap=cmap,
                                                  rasterized=self.settings['rasterized'])
                if self.settings['colorbar'] == 'True':
                    self.cbar = self.figure.colorbar(self.image, orientation='vertical')
            self.cursor = Cursor(self.axes, useblit=True, 
                                 color=self.settings['linecolor'], linewidth=0.5)
            self.apply_plot_settings()

    def reset_view_settings(self, overrule=False):
        if not self.view_settings['Locked'] or overrule:
            minimum = np.min(self.processed_data[-1])
            maximum = np.max(self.processed_data[-1])
            self.view_settings['Minimum'] = minimum
            self.view_settings['Maximum'] = maximum
            self.view_settings['Midpoint'] = 0.5*(minimum+maximum)
            self.view_settings['MidLock'] = False
            
    def reset_midpoint(self):
        if self.view_settings['MidLock'] == False:
            self.view_settings['Midpoint'] = 0.5*(self.view_settings['Minimum']+
                                                  self.view_settings['Maximum'])
                    
    def apply_plot_settings(self):
        self.axes.set_xlabel(self.settings['xlabel'], 
                             size=self.settings['labelsize'])
        self.axes.set_ylabel(self.settings['ylabel'], 
                             size=self.settings['labelsize'])
        if isinstance(self.image, list):
            self.image[0].set_linewidth(float(self.settings['linewidth']))
        for axis in ['top','bottom','left','right']:
            self.axes.spines[axis].set_linewidth(float(self.settings['spinewidth']))
        self.axes.tick_params(labelsize=self.settings['ticksize'], 
                              width=float(self.settings['spinewidth']), 
                              color=rcParams['axes.edgecolor'])
        if self.settings['minorticks'] == 'True':
            self.axes.minorticks_on()
        if self.settings['title'] == '<label>':
            self.axes.set_title(self.label, size=self.settings['titlesize'])
        else:
            self.axes.set_title(self.settings['title'], 
                                size=self.settings['titlesize'])
        if self.settings['colorbar'] == 'True' and len(self.get_columns()) == 3:
            self.cbar.ax.set_title(self.settings['clabel'], 
                                   size=self.settings['labelsize'])
            self.cbar.ax.tick_params(labelsize=self.settings['ticksize'], 
                                     color=rcParams['axes.edgecolor']) 
            self.cbar.outline.set_linewidth(float(self.settings['spinewidth']))

    def apply_view_settings(self):
        if len(self.get_columns()) == 3:
            norm = MidpointNormalize(vmin=self.view_settings['Minimum'], 
                                     vmax=self.view_settings['Maximum'], 
                                     midpoint=self.view_settings['Midpoint'])
            self.image.set_norm(norm)
            if self.settings['colorbar'] == 'True':
                self.cbar.update_normal(self.image)
                self.cbar.ax.set_title(self.settings['clabel'], 
                                       size=self.settings['labelsize'])
                self.cbar.ax.tick_params(labelsize=self.settings['ticksize'], 
                                         color=rcParams['axes.edgecolor'])          
    
    def apply_colormap(self):
        cmap_str = self.view_settings['Colormap']
        if self.view_settings['Reverse']:
            cmap_str += '_r'
        cmap = cm.get_cmap(cmap_str, lut=int(self.settings['lut']))
        cmap.set_bad(self.settings['maskcolor'])
        if len(self.get_columns()) == 3:
            self.image.set_cmap(cmap)
        else:
            self.image[0].set_color(cmap(0.5))            

    def apply_filter(self, filt, update_color_limits=True):
        if filt.checkstate:
            self.processed_data = filt.function(self.processed_data, 
                                                filt.method,
                                                filt.settings[0], 
                                                filt.settings[1]) 
            if update_color_limits:
                self.reset_view_settings()
                self.apply_view_settings()
                
    def apply_all_filters(self, update_color_limits=True):        
        for filt in self.filters:
            if filt.checkstate:
                self.processed_data = filt.function(self.processed_data, 
                                                    filt.method,
                                                    filt.settings[0], 
                                                    filt.settings[1])
        if update_color_limits:
            self.reset_view_settings()
            if hasattr(self, 'image'):
                self.apply_view_settings()
       
    def extension_setting_edited(self, editor, setting_name):
        pass
        
    def add_extension_actions(self, editor, menu):
        pass
    
    def do_extension_actions(self, editor, menu):
        pass
        
    def hide_linecuts(self):
        if hasattr(self, 'linecut_window'):
            self.linecut_window.running = False
        for line in reversed(self.axes.get_lines()):
            line.remove()
            del line
        for patch in reversed(self.axes.patches):
            patch.remove()
            del patch
        if hasattr(self, 'linecut_points'):
            del self.linecut_points
        self.canvas.draw()
    
    def open_fft_window(self):
        if self.fft_orientation == 'vertical':
            self.fft = np.fft.rfft(self.processed_data[-1], axis=1)
        elif self.fft_orientation == 'horizontal':
            self.fft = np.fft.rfft(self.processed_data[-1], axis=0)
        self.fft_window = FFTWindow(self.fft)
        self.fft_window.show()


class NumpyData(BaseClassData):
    def __init__(self, filepath, canvas, dataset):
        super().__init__(filepath, canvas)
        self.dataset = dataset
        self.label = self.dataset['Label']

    def setup_plot_settings(self):
        for setting, value in self.dataset['Settings'].items():
            if setting in self.settings:
                self.settings[setting] = value        

    def setup_filters(self):
        self.filters = self.dataset['Filters']
        
    def setup_view_settings(self):
        self.view_settings = self.dataset['View Settings']       
        
    def setup_raw_data(self):
        self.raw_data = self.dataset['Raw Data']

    def prepare_data_for_plot(self, reload=False):
        self.copy_raw_to_processed_data()
        self.apply_all_filters()


class Filter:   
    DEFAULT_SETTINGS = {'Derivative': {'Method': ['Mid'],
                                       'Settings': ['0', '1'],
                                       'Function': filters.derivative,
                                       'Checkstate': 2},
                        'Smoothen': {'Method': ['Gauss', 'Median'],
                                     'Settings': ['0', '2'],
                                     'Function': filters.smooth,
                                     'Checkstate': 2},
                        'Sav-Gol': {'Method': ['Y','X','dY','dX','ddY','ddX'],
                                    'Settings': ['7', '2'],
                                    'Function': filters.sav_gol,
                                    'Checkstate': 2},                               
                        'Crop X': {'Method': ['Abs', 'Rel'],
                                   'Settings': ['-1', '1'],
                                   'Function': filters.crop_x,
                                   'Checkstate': 0},                              
                        'Crop Y': {'Method': ['Abs', 'Rel'],
                                   'Settings': ['-1', '1'],
                                   'Function': filters.crop_y,
                                   'Checkstate': 0},
                        'Roll X': {'Method': ['Index'],
                                   'Settings': ['0', '0'],
                                   'Function': filters.roll_x,
                                   'Checkstate': 0},                             
                        'Roll Y': {'Method': ['Index'],
                                   'Settings': ['0', '0'],
                                   'Function': filters.roll_y,
                                   'Checkstate': 0}, 
                        'Cut X': {'Method': ['Index'],
                                  'Settings': ['0', '0'],
                                  'Function': filters.cut_x,
                                  'Checkstate': 0},                               
                        'Cut Y': {'Method': ['Index'],
                                  'Settings': ['0', '0'],
                                  'Function': filters.cut_y,
                                  'Checkstate': 0},                                
                        'Swap XY': {'Method': [''],
                                    'Settings': ['', ''],
                                    'Function': filters.swap_xy,
                                    'Checkstate': 2}, 
                        'Flip': {'Method': ['L-R','U-D'],
                                 'Settings': ['', ''],
                                 'Function': filters.flip,
                                 'Checkstate': 2}, 
                        'Normalize': {'Method': ['Max', 'Min', 'Point'],
                                      'Settings': ['', ''],
                                      'Function': filters.normalize,
                                      'Checkstate': 0}, 
                        'Offset': {'Method': ['X','Y','Z'],
                                   'Settings': ['0', ''],
                                   'Function': filters.offset,
                                   'Checkstate': 0},               
                        'Absolute': {'Method': [''],
                                     'Settings': ['', ''],
                                     'Function': filters.absolute,
                                     'Checkstate': 2},                 
                        'Multiply': {'Method': ['X','Y','Z'],
                                     'Settings': ['1', ''],
                                     'Function': filters.multiply,
                                     'Checkstate': 2}, 
                        'Slope': {'Method': [''],
                                  'Settings': ['0', '-1'],
                                  'Function': filters.add_slope,
                                  'Checkstate': 0}, 
                        'Logarithm': {'Method': ['Mask','Shift','Abs'],
                                      'Settings': ['', ''],
                                      'Function': filters.logarithm,
                                      'Checkstate': 2}, 
                        'Interp': {'Method': ['linear','cubic','quintic'],
                                   'Settings': ['800', '600'],
                                   'Function': filters.interpolate,
                                   'Checkstate': 0},
                        'Subtract': {'Method': ['Ver', 'Hor'],
                                     'Settings': ['0', ''],
                                     'Function': filters.subtract_trace,
                                     'Checkstate': 0}, 
                        'Divide': {'Method': ['X','Y','Z'],
                                   'Settings': ['1', ''],
                                   'Function': filters.divide,
                                   'Checkstate': 0}, 
                        'Invert': {'Method': ['X','Y','Z'],
                                   'Settings': ['', ''],
                                   'Function': filters.invert,
                                   'Checkstate': 0}} 
    
    def __init__(self, name, method=None, settings=None, checkstate=None):
        self.name = name
        default_settings = self.DEFAULT_SETTINGS.copy()
        self.method_list = default_settings[name]['Method']
        if method:
            self.method = method
        else:
            self.method = self.method_list[0]
        if settings:
            self.settings = settings
        else:
            self.settings = default_settings[name]['Settings']
        if checkstate:
            self.checkstate = checkstate
        else:
            self.checkstate = default_settings[name]['Checkstate']
        self.function = default_settings[name]['Function']
        

class LineCutWindow(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True
        self.init_widgets()
        self.init_canvas()
        self.init_connections()
        self.init_layouts()
        self.set_main_layout()
        
    def init_widgets(self):
        self.setWindowTitle('Inspectra Gadget - Linecut Window')
        self.resize(600, 600)
        self.save_button = QtWidgets.QPushButton('Save Data')
        self.save_image_button = QtWidgets.QPushButton('Save Image')
        self.copy_image_button = QtWidgets.QPushButton('Copy Image')
        self.orientation_button = QtWidgets.QPushButton('Hor./Vert.')
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.fit_button = QtWidgets.QPushButton('Fit')
        self.guess_checkbox = QtWidgets.QCheckBox('Initial guess')
        self.guess_edit = QtWidgets.QLineEdit()
        self.fit_box = QtWidgets.QComboBox()
        self.fit_box.addItems(fits.get_names())
        self.fit_box.setCurrentIndex(0)        
        self.pars_label = QtWidgets.QLabel(fits.get_names(parameters=self.fit_box.currentText()))
        
    def init_connections(self):
        self.save_button.clicked.connect(self.save_data)
        self.save_image_button.clicked.connect(self.save_image)
        self.copy_image_button.clicked.connect(self.copy_image)
        self.orientation_button.clicked.connect(self.change_orientation)
        self.clear_button.clicked.connect(self.clear_lines)
        self.fit_box.currentIndexChanged.connect(self.fit_type_changed)
        self.fit_button.clicked.connect(self.start_fitting)
        
    def init_canvas(self):
        self.figure = Figure(tight_layout={'pad':2})
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.scroll_event_id = self.canvas.mpl_connect('scroll_event', 
                                                       self.mouse_scroll_canvas)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)        
    
    def init_layouts(self):
        self.top_buttons_layout = QtWidgets.QHBoxLayout()
        self.fit_layout = QtWidgets.QHBoxLayout()
        self.bottom_buttons_layout = QtWidgets.QHBoxLayout()
        self.main_layout = QtWidgets.QVBoxLayout()
        
        self.top_buttons_layout.addWidget(self.save_button)
        self.top_buttons_layout.addWidget(self.save_image_button)
        self.top_buttons_layout.addWidget(self.copy_image_button)
        self.top_buttons_layout.addWidget(self.orientation_button)
        self.top_buttons_layout.addStretch()
        
        self.bottom_buttons_layout.addStretch()
        self.bottom_buttons_layout.addWidget(self.fit_button)
        self.bottom_buttons_layout.addWidget(self.clear_button)        

        self.fit_layout.addStretch()
        self.fit_layout.addWidget(self.guess_checkbox)
        self.fit_layout.addWidget(self.guess_edit)
        self.fit_layout.addWidget(self.pars_label) 
        self.fit_layout.addWidget(self.fit_box)

    def set_main_layout(self):
        self.main_layout.addLayout(self.top_buttons_layout)
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.navi_toolbar)
        self.main_layout.addLayout(self.fit_layout)
        self.main_layout.addLayout(self.bottom_buttons_layout)
        self.setLayout(self.main_layout)        
             
    def change_orientation(self):
        if self.orientation == 'horizontal':
            self.orientation = 'vertical'
        elif self.orientation == 'vertical':
            self.orientation = 'horizontal'
        self.update()
  
    def update(self):
        if self.running:
            try:
                self.parent.linecut.remove()
                del self.parent.linecut
            except:
                pass
            self.ylabel = self.parent.settings['clabel']
            self.draw_plot()
            self.parent.canvas.draw()
            self.show()
               
    def clear_lines(self):
        for line in reversed(self.axes.get_lines()):
            if line.get_linestyle() == '--':
                line.remove()
                del line
        if hasattr(self, 'peak_estimates'):
            self.peak_estimates = []
        self.canvas.draw()
    
    def detect_peaks(self):
        self.clear_lines()
        if self.reverse_checkbox.checkState(): 
            index_trace = np.argmax(self.z[:,0])
        else:
            index_trace = np.argmin(self.z[:,0])
        peaks, _ = find_peaks(self.y[index_trace,:], width=5)
        if not hasattr(self, 'peak_estimates'):
            self.peak_estimates = []
        for peak in peaks:
            self.peak_estimates.append(self.axes.axvline(self.x[index_trace,peak], 
                                                         linestyle='dashed'))
        self.canvas.draw()
    
    def fit_type_changed(self):
        self.pars_label.setText(fits.get_names(parameters=self.fit_box.currentText()))
    
    def start_fitting(self):
        function_name = self.fit_box.currentText()
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
        self.draw_plot(self.x, self.y)
        self.draw_fits()
        self.plot_parameters()
        
        
    def plot_parameters(self):
        self.axes.text(0.02,0.95,'Position: {self.fit_parameters[2]:.4g}', 
                       transform=self.axes.transAxes)
        self.axes.text(0.02,0.9,'Height: {self.fit_parameters[1]:.4g}', 
                       transform=self.axes.transAxes)
        self.axes.text(0.02,0.85,'Width: {self.fit_parameters[0]:.4g}', 
                       transform=self.axes.transAxes)
        self.axes.text(0.02,0.8,'Background: {self.fit_parameters[3]:.4g}', 
                       transform=self.axes.transAxes)
        self.canvas.draw()
         
    def draw_plot(self):
        self.running = True
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        
        if self.orientation == 'horizontal':
            x = self.parent.processed_data[0][:,self.parent.selected_indices[1]]
            y = self.parent.processed_data[2][:,self.parent.selected_indices[1]]
            z = self.parent.processed_data[1][0,self.parent.selected_indices[1]]
            xlabel = self.parent.settings['xlabel']
            title = f'{self.parent.settings["ylabel"]} = {z}'
            self.parent.linecut = self.parent.axes.axhline(y=z, linestyle='dashed', linewidth=1, 
                                                           color=self.parent.settings['linecolor'])
        elif self.orientation == 'vertical':
            x = self.parent.processed_data[1][self.parent.selected_indices[0],:]
            y = self.parent.processed_data[2][self.parent.selected_indices[0],:]
            z = self.parent.processed_data[0][self.parent.selected_indices[0],0]
            xlabel = self.parent.settings['ylabel']
            title = f'{self.parent.settings["xlabel"]} = {z}'
            self.parent.linecut = self.parent.axes.axvline(x=z, linestyle='dashed', linewidth=1, 
                                                           color=self.parent.settings['linecolor'])
        elif self.orientation == 'diagonal' or self.orientation == 'circular':
            x0 = self.parent.linecut_points[0].x 
            y0 = self.parent.linecut_points[0].y
            x1 = self.parent.linecut_points[1].x 
            y1 = self.parent.linecut_points[1].y                
            l_x, l_y = self.parent.processed_data[0].shape
            x_min = np.amin(self.parent.processed_data[0][:,0])
            x_max = np.amax(self.parent.processed_data[0][:,0])
            y_min = np.amin(self.parent.processed_data[1][0,:])
            y_max = np.amax(self.parent.processed_data[1][0,:])
            i_x0 = (l_x-1)*(x0-x_min)/(x_max-x_min)
            i_y0 = (l_y-1)*(y0-y_min)/(y_max-y_min)
            i_x1 = (l_x-1)*(x1-x_min)/(x_max-x_min)
            i_y1 = (l_y-1)*(y1-y_min)/(y_max-y_min)
            if self.orientation == 'diagonal':
                n = int(np.sqrt((i_x1-i_x0)**2+(i_y1-i_y0)**2))
                x_diag = np.linspace(i_x0, i_x1, n), 
                y_diag = np.linspace(i_y0, i_y1, n)
                y = map_coordinates(self.parent.processed_data[-1], 
                                    np.vstack((x_diag, y_diag)))
                x = map_coordinates(self.parent.processed_data[0], 
                                    np.vstack((x_diag, y_diag)))                
                xlabel = self.parent.settings['xlabel']
            elif self.orientation == 'circular':
                n = int(8*np.sqrt((i_x0-i_x1)**2+(i_y0-i_y1)**2))
                theta = np.linspace(0, 2*np.pi, n)
                i_x_circ = i_x0+(i_x1-i_x0)*np.cos(theta) 
                i_y_circ = i_y0+(i_y1-i_y0)*np.sin(theta)
                y = map_coordinates(self.parent.processed_data[-1], 
                                    np.vstack((i_x_circ, i_y_circ)))
                x = theta
                xlabel = 'Angle (rad)'
            title = ''
        ylabel = self.parent.settings['clabel']
        self.image = self.axes.plot(x, y, linewidth=self.parent.settings['linewidth'])
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.axes.set_xlabel(xlabel, size='xx-large')
        self.axes.set_ylabel(ylabel, size='xx-large')
        self.axes.tick_params(labelsize='x-large', color=rcParams['axes.edgecolor'])
        self.axes.set_title(title, size='x-large')
        self.canvas.draw()
        self.parent.canvas.draw()
              
    def draw_fits(self):
        self.axes.plot(self.x, self.y_fit, 'k--', 
                       linewidth=self.parent.settings['linewidth'])            
        self.canvas.draw()
        
    def closeEvent(self, event):
        self.parent.hide_linecuts()
        self.running = False
        
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
            if DARK_THEME and qdarkstyle_imported:
                rcParams_to_light_theme()
                self.update()
            self.figure.savefig(filename)
            if DARK_THEME and qdarkstyle_imported:
                rcParams_to_dark_theme()
                self.update()
            print('Saved!')
    
    def copy_image(self):
        self.cursor.horizOn = False
        self.cursor.vertOn = False            
        self.canvas.draw()
        if DARK_THEME and qdarkstyle_imported:
            rcParams_to_light_theme()
            self.parent.update()           
        buf = io.BytesIO()
        self.figure.savefig(buf, dpi=300, transparent=True, bbox_inches='tight')
        QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buf.getvalue()))
        buf.close()
        self.cursor.horizOn = True
        self.cursor.vertOn = True                       
        self.canvas.draw()
        if DARK_THEME and qdarkstyle_imported:
            rcParams_to_dark_theme()
            self.update() 
            
    def mouse_scroll_canvas(self, event):
        if event.inaxes:
            data_shape = self.parent.processed_data[0].shape
            if self.orientation == 'horizontal':
                new_index = self.parent.selected_indices[1]+int(event.step)
                if new_index >= 0 and new_index < data_shape[1]:
                    self.parent.selected_indices[1] = new_index
            elif self.orientation == 'vertical':
                new_index = self.parent.selected_indices[0]+int(event.step)
                if new_index >= 0 and new_index < data_shape[0]:
                    self.parent.selected_indices[0] = new_index
            self.update()
            self.parent.canvas.draw()
            
            
class MultipleLineCutsWindow(LineCutWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Inspectra Gadget - Multiple Linecuts Window')
 
    def init_widgets(self):
        super().init_widgets()
        self.colormap_box = QtWidgets.QComboBox()
        self.colormap_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colormap_type_box = QtWidgets.QComboBox()
        self.colormap_type_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)  
        for cmap_type in cmaps:    
            self.colormap_type_box.addItem(cmap_type)
        self.colormap_box.addItems(list(cmaps.values())[0])
        
        self.number_label = QtWidgets.QLabel('Number of lines:')
        self.number_line_edit = QtWidgets.QLineEdit('5')
        self.number_line_edit.setFixedSize(40,20)
        self.all_lines_button = QtWidgets.QPushButton('All')
        self.all_lines_button.setFixedSize(35,22)
        self.check_specify_lines = QtWidgets.QCheckBox('Specify lines:')
        self.check_specify_lines.clicked.connect(self.draw_plot)
        self.specify_lines_edit = QtWidgets.QLineEdit('')
        self.offset_label = QtWidgets.QLabel('Offset')
        self.offset_line_edit = QtWidgets.QLineEdit('0')
        self.offset_line_edit.setFixedSize(70,20)
        self.check_legend = QtWidgets.QCheckBox('Legend')

        self.reverse_checkbox = QtWidgets.QCheckBox('Reverse Fit Order')
        self.reverse_checkbox.setCheckState(QtCore.Qt.Checked)
        self.detect_peaks_button = QtWidgets.QPushButton('Detect Peaks')

    def init_connections(self):
        super().init_connections()
        self.colormap_type_box.currentIndexChanged.connect(self.colormap_type_edited)
        self.colormap_box.currentIndexChanged.connect(self.draw_plot)
        self.number_line_edit.editingFinished.connect(self.draw_plot)
        self.all_lines_button.clicked.connect(self.clicked_all_lines)
        self.specify_lines_edit.editingFinished.connect(self.draw_plot)
        self.offset_line_edit.editingFinished.connect(self.draw_plot)
        self.check_legend.clicked.connect(self.draw_plot)
        self.detect_peaks_button.clicked.connect(self.detect_peaks)
        
    def init_canvas(self):
        super().init_canvas()
        self.canvas.mpl_disconnect(self.scroll_event_id) 
        self.canvas.mpl_connect('button_press_event', self.mouse_click_canvas)
               
    def init_layouts(self):
        super().init_layouts()
        
        self.top_buttons_layout.addWidget(self.colormap_type_box)
        self.top_buttons_layout.addWidget(self.colormap_box)
        
        self.lines_layout = QtWidgets.QHBoxLayout()
        self.lines_layout.addWidget(self.number_label)
        self.lines_layout.addWidget(self.number_line_edit)
        self.lines_layout.addWidget(self.all_lines_button)
        self.lines_layout.addWidget(self.check_specify_lines)
        self.lines_layout.addWidget(self.specify_lines_edit)
        self.lines_layout.addWidget(self.offset_label)
        self.lines_layout.addWidget(self.offset_line_edit)
        self.lines_layout.addWidget(self.check_legend)
        
        self.bottom_buttons_layout.addWidget(self.detect_peaks_button)
        self.bottom_buttons_layout.addWidget(self.reverse_checkbox)
    
    def set_main_layout(self):
        self.main_layout.addLayout(self.top_buttons_layout)
        self.main_layout.addLayout(self.lines_layout)
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.navi_toolbar)
        self.main_layout.addLayout(self.fit_layout)
        self.main_layout.addLayout(self.bottom_buttons_layout)
        self.setLayout(self.main_layout)
    
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
    
    def update(self):
        self.draw_plot()
        self.show()
        
    def closeEvent(self, event):
        self.running = False
    
    def start_fitting(self):
        function_name = self.fit_box.currentText()
        if hasattr(self, 'peak_estimates') and len(self.peak_estimates) > 1:
            center_estimates = [p.get_xdata()[0] for p in self.peak_estimates]
            n_peaks = len(center_estimates)
            self.clear_lines()
            line_indices = list(range(self.z.shape[0]))
            if np.argmin(self.z[:,0]) == 0 and self.reverse_checkbox.checkState():  
                line_indices.reverse()
            elif np.argmax(self.z[:,0]) == 0 and not self.reverse_checkbox.checkState(): 
                line_indices.reverse()
            
            # Initialize objects
            values = None
            peak_values = {}
            peak_values['background_constant'] = []
            peak_values['mean_height'] = []
            peak_values['mean_fwhm'] = []
            peak_values['coordinates'] = self.z[line_indices,0]
            for i in range(len(center_estimates)):
                peak_values[f'p{i}_center'] = [] 
                peak_values[f'p{i}_height'] = [] 
                peak_values[f'p{i}_fwhm'] = []

            # Fit routine for every trace
            for counter, line_index in enumerate(line_indices):
                x, y, z = self.x[line_index,:], self.y[line_index,:], self.z[line_index,0]
                print(f'Fitting peaks; index = {line_index}, value = {z}...')
                if counter == 0:
                    background_estimate = np.amin(y)
                    height_estimate = np.amax(y) - np.amin(y)
                    fwhm_estimate = 0.25*min([abs(center_estimates[0]-c_est) 
                                              for c_est in center_estimates[1:]])
                    if function_name == 'Lorentzian':
                        sigma_estimate = fwhm_estimate/2
                        amplitude_estimate = height_estimate*sigma_estimate*np.pi
                    else: # Gaussian
                        sigma_estimate = fwhm_estimate/2.35482
                        amplitude_estimate = height_estimate*sigma_estimate*np.sqrt(2*np.pi)
                    peak_bounds = (np.amin(x)-0.5*(np.amax(x)-np.amin(x)), 
                                   np.amax(x)+0.5*(np.amax(x)-np.amin(x)))
                else:
                    background_estimate = values['bkg_c']
                    sigma_estimate = values['p0_sigma']
                    amplitude_estimate = values['p0_amplitude']
                    center_estimates = [values[f'p{i}_center'] for i 
                                        in range(len(center_estimates))]
                model = ConstantModel(prefix='bkg_')
                params = model.make_params()
                params['bkg_c'].set(background_estimate, min=-np.inf)
                for i, center_estimate in enumerate(center_estimates):
                    peak, pars = self.add_peak(f'p{i}_', function_name, 
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
                    peak_values[f'p{i}_center'].append(values[f'p{i}_center'])
                    if function_name == 'Lorentzian':    
                        best_fwhm = 2*values[f'p{i}_sigma']
                        best_height = (values[f'p{i}_amplitude']/
                                       (np.pi*values[f'p{i}_sigma']))
                    else:
                        best_fwhm = 2.35482*values[f'p{i}_sigma']
                        best_height = (values[f'p{i}_amplitude']/
                                       (np.sqrt(2*np.pi)*values[f'p{i}_sigma']))
                    peak_values[f'p{i}_height'].append(best_height)
                    peak_values[f'p{i}_fwhm'].append(best_fwhm)
                self.axes.plot(x, result.best_fit+line_index*self.offset, 'k--')
            self.canvas.draw()
            
            # Plot peak center positions as dots in main canvas
            for i in range(n_peaks):
                if self.orientation == 'vertical':
                    self.parent.axes.scatter(self.z[line_indices,0], 
                                             peak_values[f'p{i}_center'], c='r')
                elif self.orientation == 'horizontal':
                    self.parent.axes.scatter(peak_values[f'p{i}_center'], 
                                             self.z[line_indices,0], c='r')
            self.parent.canvas.draw()
            
            # Calculate properties of peaks
            peak_values['mean_height'] = [np.mean([peak_values[f'p{i}_height'][j] 
                                                   for i in range(n_peaks)]) 
                                          for j in range(self.number)]
            peak_values['mean_fwhm'] = [np.mean([peak_values[f'p{i}_fwhm'][j] 
                                                 for i in range(n_peaks)]) 
                                        for j in range(self.number)]
            for index, parity in enumerate(['even','odd']):
                peak_values['mean_spacing_'+parity] = [np.mean([np.abs(peak_values[f'p{i}_center'][j] - 
                                                                       peak_values[f'p{i+1}_center'][j]) 
                                                                for i in range(index,n_peaks-1,2)]) 
                                                       for j in range(self.number)]
                peak_values['mean_height_'+parity] = [np.mean([peak_values[f'p{i}_height'][j] 
                                                               for i in range(index,n_peaks,2)]) 
                                                      for j in range(self.number)]
                peak_values['mean_fwhm_'+parity] = [np.mean([peak_values[f'p{i}_fwhm'][j] 
                                                             for i in range(index,n_peaks,2)]) 
                                                    for j in range(self.number)]

            # Save dictionary with peak properties to file    
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save peak-fitting data as...', 
                    os.path.splitext(self.parent.filepath)[0]+'_fitdata', '*.npy')
            if filename:
                np.save(filename, peak_values)
                print('Peak fitting data saved!')
        
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
        self.pfit = np.array(self.fit_parameters).transpose()
        rows, cols = self.pfit.shape
        self.pars_window = []
        for index in range(rows):
            ylabel = fits.get_names(self.fit_box.currentText()).split(',')[index]
            self.pars_window.append(ParametersWindow(
                    self.z[:,0], self.pfit[index], self.zlabel, ylabel))
            self.pars_window[-1].show()
    
    def colormap_type_edited(self):
        self.colormap_box.currentIndexChanged.disconnect(self.draw_plot)
        self.colormap_box.clear()
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentText()])
        self.colormap_box.currentIndexChanged.connect(self.draw_plot)
        self.draw_plot()
         
    def draw_plot(self):
        self.running = True
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        try:
            self.offset = float(self.offset_line_edit.text())
        except Exception as e:
            print('Invalid offset value!', e)
            self.offset = 0
            self.offset_line_edit.setText('0')
        selected_colormap = cm.get_cmap(self.colormap_box.currentText())
        
        rows, cols = self.parent.processed_data[0].shape
        self.number = int(self.number_line_edit.text())
        if self.orientation == 'horizontal':
            if not self.check_specify_lines.checkState():
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
                        indices = [np.argmin(np.abs(self.parent.processed_data[1][0,:]-float(s))) 
                                   for s in self.specify_lines_edit.text().split(',')]
                    self.number = len(indices)
                except Exception as e:
                    print('Invalid lines specification!', e)
                    indices = []
                    self.number = 0
            x = self.parent.processed_data[0][:,indices].transpose()
            y = self.parent.processed_data[2][:,indices].transpose()
            z = self.parent.processed_data[1][:,indices].transpose()
            xlabel = self.parent.settings['xlabel']
            zlabel = self.parent.settings['ylabel']
        elif self.orientation == 'vertical':
            if not self.check_specify_lines.checkState():
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
                        indices = [np.argmin(np.abs(self.parent.processed_data[0][:,0]-float(s))) 
                                   for s in self.specify_lines_edit.text().split(',')]
                    self.number = len(indices)
                except Exception as e:
                    print('Invalid lines specification!', e)
                    indices = []
                    self.number = 0

            x = self.parent.processed_data[1][indices,:]
            y = self.parent.processed_data[2][indices,:]
            z = self.parent.processed_data[0][indices,:]
            xlabel = self.parent.settings['ylabel']
            zlabel = self.parent.settings['xlabel']
        self.labels = [f'{z[i,0]:.5g}' for i in range(self.number)]            
        ylabel = self.parent.settings['clabel']
        title = ''
        line_colors = selected_colormap(np.linspace(0.1,0.9,self.number))
        for index in range(self.number):
            self.axes.plot(x[index], y[index]+index*self.offset, 
                           color=line_colors[index], 
                           linewidth=self.parent.settings['linewidth'], 
                           label=self.labels[index])
        if self.check_legend.checkState():
            self.axes.legend(title=zlabel)
        self.cursor = Cursor(self.axes, useblit=True, 
                             color='grey', linewidth=0.5)
        self.axes.set_xlabel(xlabel, size='xx-large')
        self.axes.set_ylabel(ylabel, size='xx-large')
        self.axes.tick_params(labelsize='x-large', color=rcParams['axes.edgecolor'])
        self.axes.set_title(title, size='x-large')
        self.canvas.draw()
           
    def draw_fits(self):
        for index in range(self.number):
            self.axes.plot(self.x[index], self.y_fit[index]+index*self.offset, 
                           'k--', linewidth=self.linewidth)         
        self.canvas.draw()
          
    def clicked_all_lines(self):
        rows, cols = self.parent.processed_data[0].shape
        if self.orientation == 'horizontal':
            number = cols
        elif self.orientation == 'vertical':
            number = rows
        self.number_line_edit.setText(str(number))
        self.draw_plot()
        
    def save_data(self):
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save Data As...')
        data = np.array([self.z.flatten(), self.x.flatten(), self.y.flatten()])
        np.savetxt(filename, data.T)
            
        
class ParametersWindow(QtWidgets.QWidget):
    def __init__(self, x, y, xlabel, ylabel):
        super().__init__()
        self.resize(600, 600)
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.button_layout = QtWidgets.QHBoxLayout()
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.x = x
        self.y = y
        self.axes.plot(self.x, self.y,'.')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.tick_params(color=rcParams['axes.edgecolor'])
        self.canvas.draw()
        self.figure.tight_layout(pad=2)
        self.canvas.draw()
        self.save_button = QtWidgets.QPushButton('Save')
        self.save_button.clicked.connect(self.save_data)
        self.save_image_button = QtWidgets.QPushButton('Save Image')
        self.save_image_button.clicked.connect(self.save_image)
        self.vertical_layout.addWidget(self.navi_toolbar)
        self.vertical_layout.addWidget(self.canvas)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.save_image_button)
        self.vertical_layout.addLayout(self.button_layout)
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
        super().__init__()
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
        self.axes.tick_params(color=rcParams['axes.edgecolor'])
        self.canvas.draw()
        self.vertical_layout.addWidget(self.navi_toolbar)
        self.vertical_layout.addWidget(self.canvas)
        self.setLayout(self.vertical_layout)
    

class NoScrollQComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

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
    def __init__(self, parent, x, y, draw_line=False, draw_circle=False):
        self.parent = parent
        
        x_lb, x_ub = self.parent.axes.get_xlim()
        y_lb, y_ub = self.parent.axes.get_ylim()
        
        self.point = patches.Ellipse((x, y), (x_ub-x_lb)*0.02, 
                                     (y_ub-y_lb)*0.02, fc='k', 
                                     alpha=1, edgecolor='k')
        self.x = x
        self.y = y
        self.parent.axes.add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()
        
        if draw_line:
            line_x = [self.parent.linecut_points[0].x, self.x]
            line_y = [self.parent.linecut_points[0].y, self.y]
            self.line = Line2D(line_x, line_y, 
                               color=self.parent.settings['linecolor'], 
                               alpha=1.0, linestyle='dashed', linewidth=1)
            self.parent.axes.add_line(self.line)
        if draw_circle:
            x0, y0 = self.parent.linecut_points[0].x, self.parent.linecut_points[0].y
            x1, y1 = self.parent.linecut_points[1].x, self.y
            self.circle = patches.Ellipse((x0, y0), 2*(x1-x0), 2*(y1-y0), 
                                          fc='none', alpha=None, linestyle='dashed', 
                                          linewidth=1, edgecolor='k')
            self.parent.axes.add_patch(self.circle)

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.parent.cursor.horizOn = False
        self.parent.cursor.vertOn = False 
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        if hasattr(self.parent.linecut_points[1], 'line'):
            if self == self.parent.linecut_points[1]:
                self.line.set_animated(True)
            else:
                self.parent.linecut_points[1].line.set_animated(True)
        if (len(self.parent.linecut_points) > 2 and 
            hasattr(self.parent.linecut_points[2], 'circle')):
            if self == self.parent.linecut_points[2]:
                self.circle.set_animated(True)
            else:
                self.parent.linecut_points[2].circle.set_animated(True)
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
        if (len(self.parent.linecut_points) > 2 and 
            hasattr(self.parent.linecut_points[2], 'circle')):
            if self == self.parent.linecut_points[1]:
                dy = 0
            elif self == self.parent.linecut_points[2]:
                dx = 0
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
        self.x = self.point.center[0]
        self.y = self.point.center[1]
        canvas = self.point.figure.canvas
        axes = self.point.axes
        canvas.restore_region(self.background)
        axes.draw_artist(self.point)
        if hasattr(self.parent.linecut_points[1], 'line'):
            if self == self.parent.linecut_points[1]:
                self.line.set_animated(True)
                axes.draw_artist(self.line)
                line_x = [self.parent.linecut_points[0].x, self.x]
                line_y = [self.parent.linecut_points[0].y, self.y]
                self.line.set_data(line_x, line_y)
            else:
                self.parent.linecut_points[1].line.set_animated(True)
                axes.draw_artist(self.parent.linecut_points[1].line)
                line_x = [self.x, self.parent.linecut_points[1].x]
                line_y = [self.y, self.parent.linecut_points[1].y]
                self.parent.linecut_points[1].line.set_data(line_x, line_y)
        if (len(self.parent.linecut_points) > 2 and 
            hasattr(self.parent.linecut_points[2], 'circle')):
            if self == self.parent.linecut_points[2]:
                self.circle.set_animated(True)
                axes.draw_artist(self.circle)
                self.circle.height = 2*(self.y-self.parent.linecut_points[0].y)
            elif self == self.parent.linecut_points[1]:
                self.parent.linecut_points[2].circle.set_animated(True)
                axes.draw_artist(self.parent.linecut_points[2].circle)
                self.parent.linecut_points[2].circle.width = 2*(self.x-self.parent.linecut_points[0].x)
            else:
                self.parent.linecut_points[2].circle.set_animated(True)
                axes.draw_artist(self.parent.linecut_points[2].circle)
                self.parent.linecut_points[2].circle.set_center((self.x, self.y))
        canvas.blit(axes.bbox)


    def on_release(self, event):
        if DraggablePoint.lock is not self:
            return
        self.parent.cursor.horizOn = True
        self.parent.cursor.vertOn = True
        self.press = None
        DraggablePoint.lock = None
        self.point.set_animated(False)
        if hasattr(self.parent.linecut_points[1], 'line'):
            if self == self.parent.linecut_points[1]:
                self.line.set_animated(False)
            else:
                self.parent.linecut_points[1].line.set_animated(False)
        if (len(self.parent.linecut_points) > 2 and 
            hasattr(self.parent.linecut_points[2], 'circle')):
            if self == self.parent.linecut_points[2]:
                self.circle.set_animated(False)
            elif self == self.parent.linecut_points[1]:
                self.parent.linecut_points[2].circle.set_animated(False) 
            else:
                self.parent.linecut_points[2].circle.set_animated(False)
            if self == self.parent.linecut_points[0]:
                circle = self.parent.linecut_points[2].circle
                self.parent.linecut_points[1].point.center = (circle.center[0]+0.5*circle.width, circle.center[1])
                self.parent.linecut_points[2].point.center = (circle.center[0],circle.center[1]+0.5*circle.height)
                self.parent.linecut_points[1].x = self.parent.linecut_points[1].point.center[0]
                self.parent.linecut_points[1].y = self.parent.linecut_points[1].point.center[1]
                self.parent.linecut_points[2].x = self.parent.linecut_points[2].point.center[0]
                self.parent.linecut_points[2].y = self.parent.linecut_points[2].point.center[1]
        self.background = None
        self.point.figure.canvas.draw()
        self.x = self.point.center[0]
        self.y = self.point.center[1]
        self.update()
        self.parent.linecut_window.activateWindow()

    def disconnect(self):
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
 
def rcParams_to_dark_theme():
    rcParams['text.color'] = LIGHT_COLOR
    rcParams['xtick.color'] = LIGHT_COLOR
    rcParams['ytick.color'] = LIGHT_COLOR
    rcParams['axes.facecolor'] = DARK_COLOR
    rcParams['axes.edgecolor'] = GREY_COLOR
    rcParams['axes.labelcolor'] = LIGHT_COLOR
    
def rcParams_to_light_theme():
    rcParams['text.color'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.labelcolor'] = 'black' 

import qd_extension
import qcodes_extension
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    app.lastWindowClosed.connect(app.quit)
    
    edit_window = Editor()
    
    if DARK_THEME and qdarkstyle_imported:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    edit_window.show()
    app.exec_()

if __name__ == '__main__':
    main()