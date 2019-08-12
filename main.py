# -*- coding: utf-8 -*-
"""
Inspectra-Gadget

Author: Joeri de Bruijckere (J.deBruijckere@tudelft.nl)

Last updated on Aug 6 2019
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import sys, os, copy, json
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from matplotlib import cm
from matplotlib.widgets import Cursor
from matplotlib import rcParams
from textwrap import wrap
import design
import filters
import fits

DEFAULT_COLUMNS = '0,1,2'
CONVERT_MICROSIEMENS_TO_ESQUAREDH = True
PRINT_FUNCTION_CALLS = False # print function commands in terminal when called
DEFAULT_VALUE_RCFILTER_CORRECT = False # only for meta.json files; applies rc-filters by default upon opening if True

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.cursive'] = ['Arial']
rcParams['mathtext.fontset'] = 'custom'

class Editor(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.init_plot_settings()
        self.init_view_settings()
        self.init_filters()
        self.init_connections()
        self.init_canvas()
        
    def init_plot_settings(self):
        font_sizes = ['xx-small','x-small','small','medium','large','x-large','xx-large']
        self.settings_menu_list = {
                'title': ['<filename>','<metadataname>'], 
                'xlabel': ['Gate Voltage (V)', '$V_{\mathrm{g}}$ (V)','Magnetic Field (T)', '$B$ (T)', 'Angle (degrees)'], 
                'ylabel': ['Bias Voltage (mV)', '$V$ (mV)', 'Gate Voltage (V)',
                           'Angle (degrees)','Temperature (mK)'],
                'clabel': ['$I$ (nA)', '$I$ (a.u.)', 'Current (nA)', 'd$I$/d$V$ ($\mu$S)', 
                        'd$I$/d$V$ ($G_0$)', 'd$I$/d$V$ (a.u.)', 'd$I$/d$V$ $(e^{2}/h)$', 
                        'd$^2$I/d$V^2$ (a.u.)', '|d$^2$I/d$V^2$| (a.u.)'],
                'titlesize': font_sizes, 'labelsize': font_sizes, 'ticksize': font_sizes,
                'colorbar': ['True', 'False'], 'columns': ['0,1,2','0,1,3','0,2,3','1,2,4'], '2D': [],
                'minorticks': ['True','False'], 'delimiter': ['',','],
                'lut': ['128','256','512','1024'], 'rasterized': ['False','True'], 
                'dpi': ['figure'], 'transparent': ['True', 'False'], 'frameon': ['False','True']}
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
        self.cmaps = [('Uniform', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']),
         ('Hybrid', ['bone+magma_r'])]
        for n in range(len(self.cmaps)):    
            self.colormap_type_box.addItem(self.cmaps[n][0])
        self.colormap_box.addItems(self.cmaps[0][1])
        self.min_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.max_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.mid_line_edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.copied_view_settings = None
        for hybrid_cmap in self.cmaps[-1][-1]:
            n_colors = 512
            top = cm.get_cmap(hybrid_cmap.split('+')[1], n_colors)
            bottom = cm.get_cmap(hybrid_cmap.split('+')[0], n_colors)
            newcolors = np.vstack((top(np.linspace(0, 1, n_colors)),
                                   bottom(np.linspace(0, 1, n_colors))))
            newcolors_r = newcolors[::-1]
            newcmp = ListedColormap(newcolors, name=hybrid_cmap)
            newcmp_r = ListedColormap(newcolors_r, name=hybrid_cmap+'_r')
            cm.register_cmap(cmap=newcmp)
            cm.register_cmap(cmap=newcmp_r)
    
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
                'Logarithm': {'Name': 'Logarithm', 'Method': 'log10',
                              'Setting 1': '', 'Setting 2': '', 'Checked': 2},
                'Curvature': {'Name': 'Curvature', 'Method': 'Y',
                             'Setting 1': '1', 'Setting 2': '', 'Checked': 2},
                'Band cut': {'Name': 'Band cut', 'Method': 'Y',
                             'Setting 1': '20', 'Setting 2': '25', 'Checked': 0},
                'Interp': {'Name': 'Interp', 'Method': 'linear',
                           'Setting 1': '800', 'Setting 2': '600', 'Checked': 0},
                'Subtract': {'Name': 'Subtract', 'Method': 'Vertical',
                             'Setting 1': '0', 'Setting 2': '', 'Checked': 0},
                'Divide': {'Name': 'Divide', 'Method': 'Z',
                             'Setting 1': '1', 'Setting 2': '', 'Checked': 2}}                  
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
        self.action_save_canvas.triggered.connect(self.save_image)
        self.load_filters_button.clicked.connect(self.load_filters)
        self.action_filters.triggered.connect(self.save_filters)
        self.action_current_file.triggered.connect(lambda: self.save_session('current'))
        self.action_all_files.triggered.connect(lambda: self.save_session('all'))
        self.action_merge_raw.triggered.connect(lambda: self.merge_files(raw_data=True))
        self.action_merge_processed.triggered.connect(lambda: self.merge_files(raw_data=False))
        self.action_refresh_30s.triggered.connect(lambda: self.start_auto_refresh(time_interval=30))
        self.action_refresh_5m.triggered.connect(lambda: self.start_auto_refresh(time_interval=300))
        self.action_refresh_30m.triggered.connect(lambda: self.start_auto_refresh(time_interval=1800))
        self.action_refresh_stop.triggered.connect(self.stop_auto_refresh)
        self.action_refresh_stop.setEnabled(False)
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
        self.subplot_grid = [(1,1),(1,2),(2,2),(2,2),(2,3),(2,3),(2,4),(2,4),(3,4),(3,4),(3,4),(3,4)]

    def open_files(self, filenames=None):
        if PRINT_FUNCTION_CALLS:
            print('open_files')
        try:
            if not filenames:
                filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
                        self, 'Open File', '', 'Data Files (*.dat *.npy)')
            if filenames:
                self.file_list.itemChanged.disconnect(self.file_checked)
                for filename in filenames:
                    print('Open '+filename)
                    if filename.split('.')[-1] == 'dat':
                        self.add_file(filename)
                    elif filename.split('.')[-1] == 'npy':
                        loaded_session = np.load(filename, allow_pickle=True)
                        for session_item in loaded_session:
                            self.add_file(session_item['File Name'], data=session_item['Raw Data'])
                            item = self.file_list.item(self.file_list.count()-1)
                            data = item.data(QtCore.Qt.UserRole)
                            data.settings = session_item['Settings']
                            data.filters = session_item['Filters']
                            data.view_settings = session_item['View Settings']
                            data.apply_all_filters(update_color_limits=False)
                last_item = self.file_list.item(self.file_list.count()-1)
                self.file_list.setCurrentItem(last_item)
                for item_index in range(self.file_list.count()-1):
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
                self.show_current_all()
                self.file_list.itemChanged.connect(self.file_checked)
                last_item.setCheckState(QtCore.Qt.Checked)
        except:
            print('Could not open file(s)...')
            
    def add_file(self, file, data=None):
        if PRINT_FUNCTION_CALLS:
            print('add_file')
        item = QtWidgets.QListWidgetItem()
        item.setText(os.path.basename(file))
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Unchecked)
        self.file_list.addItem(item)
        item.setData(QtCore.Qt.UserRole, Data3D(filepath=file, existing_data=data))
        if item.data(QtCore.Qt.UserRole).meta_data:
            item.setText(item.data(QtCore.Qt.UserRole).meta_data_name)
        
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
            self.update_plots()
            if item.checkState() == 2:
                self.file_list.setCurrentItem(item)
        except:
            print('Could not update plot(s)...')
    
    def file_clicked(self):
        if PRINT_FUNCTION_CALLS:
            print('file_clicked')
        try:
            self.show_current_all()
        except:
            print('Could not show current settings...')
    
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
                data = item.data(QtCore.Qt.UserRole)
                data.figure = self.figure
                data.axes = data.figure.add_subplot(
                        self.subplot_rows, self.subplot_cols, index+1)
                if data.settings['2D'] == 'True':
                    data.add_plot_2d()
                else:
                    data.add_plot()
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
        self.canvas.draw()
          
    def refresh_plot(self):
        if PRINT_FUNCTION_CALLS:
            print('refresh_plot')
        current_item = self.file_list.currentItem()
        if current_item:
            data = current_item.data(QtCore.Qt.UserRole)
            data.refresh_data(update_color_limits=False, refresh_unit_conversion=False)
            self.update_plots()
        
    def start_auto_refresh(self, time_interval):
        if PRINT_FUNCTION_CALLS:
            print('start_auto_refresh')
        self.auto_refresh_timer = QtCore.QTimer()
        self.auto_refresh_timer.setInterval(time_interval*1000)
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_call)
        self.action_refresh_stop.setEnabled(True)
        self.auto_refresh_timer.start()
        self.setWindowTitle('InSpectra Gadget - Auto-Refreshing Enabled')
        
    def auto_refresh_call(self):
        self.setWindowTitle('InSpectra Gadget - Auto-Refreshing Enabled (Auto-Refreshing...)')
        file_list = self.file_list
        checked_items = [file_list.item(index) for index in range(file_list.count()) 
                        if file_list.item(index).checkState() == 2]
        if checked_items:
            for index, item in enumerate(checked_items):
                data = item.data(QtCore.Qt.UserRole)
                data.refresh_data(update_color_limits=False, refresh_unit_conversion=False)
            self.update_plots()
        self.setWindowTitle('InSpectra Gadget - Auto-Refreshing Enabled')
            
    def stop_auto_refresh(self):
        if PRINT_FUNCTION_CALLS:
            print('stop_auto_refresh')
        self.auto_refresh_timer.stop()
        self.action_refresh_stop.setEnabled(False)
        self.setWindowTitle('InSpectra Gadget')
        
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
                    data.columns = [int(value.split(',')[i]) for i in range(3)]
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
                    self.update_plots()
                    self.show_current_all()            
                elif setting_name == 'delimiter':
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
                    self.update_plots()
                    self.show_current_all() 
                elif setting_name == '2D':
                    self.paste_plot_settings(which='old')
                elif setting_name == 'rc-filter':
                    data.refresh_data(update_color_limits=True, refresh_unit_conversion=False)
                    self.update_plots()
                    self.show_current_all()
                elif setting_name == 'lut':
                    data.apply_colormap()
                    self.canvas.draw()
                elif setting_name == 'rasterized' or setting_name == 'colorbar':
                    self.update_plots()
                elif setting_name == 'minorticks':
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
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentIndex()][1])
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
            menu = QtWidgets.QMenu(self)
            actions = ['Check only this item...','Check all...']
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
            if signal.text() == 'Check only this item...':
                self.file_list.itemChanged.disconnect(self.file_checked)
                for item_index in range(self.file_list.count()):
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Unchecked)
                current_item.setCheckState(QtCore.Qt.Checked)
                self.file_list.itemChanged.connect(self.file_checked)
                self.update_plots()
            elif signal.text() == 'Check all...':
                self.file_list.itemChanged.disconnect(self.file_checked)
                for item_index in range(self.file_list.count()):                        
                    self.file_list.item(item_index).setCheckState(QtCore.Qt.Checked)
                    self.file_list.itemChanged.connect(self.file_checked)
                self.update_plots()
    
    def open_plot_settings_menu(self):
        if PRINT_FUNCTION_CALLS:
            print('open_plot_settings_menu')
        table = self.settings_table
        row = table.currentRow()
        column = table.currentColumn()
        if column == 1:
            setting_name = table.item(row, 0).text()
            menu = QtWidgets.QMenu(self)
            if self.settings_menu_list[setting_name]:
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
            table.item(row, 2).setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            table.item(row, 3).setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
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
                    self, 'Save Figure As...', data_name, formats)
            if filename:
                print('Save Figure as '+filename+' ...')
                try:
                    dpi = int(data.settings['dpi'])
                except:
                    dpi = 'figure'
                self.figure.savefig(filename, dpi=dpi,
                                    transparent=data.settings['transparent']=='True',
                                    frameon=data.settings['frameon']=='True', bbox_inches='tight')
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
                    self, 'Save Session As...', os.path.splitext(data.filepath)[0], '*.npy')
                dictionary_list = [{'File Name': data.filename, 'File Path': data.filepath, 
                                   'Settings': data.settings, 'Filters': data.filters, 
                                   'View Settings': data.view_settings, 'Raw Data': data.raw_data}]
            elif which == 'all':
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, 'Save Session As...', '', '*.npy')
                dictionary_list = []
                items = [self.file_list.item(n) for n in range(self.file_list.count())]
                for item in items:
                    data = item.data(QtCore.Qt.UserRole)
                    item_dictionary = {'File Name': data.filename, 'File Path': data.filepath,
                        'Settings': data.settings, 'Filters': data.filters, 
                        'View Settings': data.view_settings, 'Raw Data': data.raw_data}
                    dictionary_list.append(item_dictionary)
            np.save(filename, dictionary_list)
                
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
                    data = plot_data.processed_data
                    index_x = np.argmin(np.absolute(data[0][:,0]-x))
                    index_y = np.argmin(np.absolute(data[1][0,:]-y))
                    if (event.button == 1 or event.button == 2) and plot_data.settings['2D'] == 'False':
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
                            plot_data.linecut_window = LineCutWindow()
                        plot_data.linecut_window.running = True
                        plot_data.update_linecut()
                        self.canvas.draw()
                        plot_data.linecut_window.activateWindow()
                    elif event.button == 3:
                        if PRINT_FUNCTION_CALLS:
                            print('rightmouseclickplot')
                        menu = QtWidgets.QMenu(self)
                        if plot_data.settings['2D'] == 'False':
                            z_value = data[2][index_x,index_y]
                            coordinates = 'x = %.4g, y = %.4g, z = %.4g (%d, %d)' % (x, y, z_value, index_x, index_y)
                            action = QtWidgets.QAction(coordinates, self)
                            action.setEnabled(False)
                            menu.addAction(action)
                            menu.addSeparator()
                        if plot_data.channels:
                            channel_menu = menu.addMenu('Change channel to...')
                            for channel in plot_data.channels[2-(plot_data.settings['2D']=='True'):]:
                                action = QtWidgets.QAction(channel, self)
                                channel_menu.addAction(action)
                        try:
                            plot_data.settings['rc-filter']
                        except KeyError:
                            pass
                        else:
                            if plot_data.settings['rc-filter'] != '':
                                if plot_data.rcfilter_correct == True:
                                    action = QtWidgets.QAction('Disable RC-filter correction...', self)                            
                                else:
                                    action = QtWidgets.QAction('Enable RC-filter correction...', self)
                                menu.addAction(action)
                        if plot_data.settings['2D'] == 'False':
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
                                plot_data.linecut_to = [int(index_x), int(index_y)]
                            else:
                                action = QtWidgets.QAction('Diagonal linecut from...', self)
                                plot_data.linecut_from = [int(index_x), int(index_y)]
                            menu.addAction(action)
                            entries = ['Hide linecuts...',
                                       'Refresh plot...',
                                       'Plot vertical linecuts...',
                                       'Plot horizontal linecuts...',
                                       'FFT vertical...',
                                       'FFT horizontal...',
                                       'Copy canvas to clipboard...']
                        elif plot_data.settings['2D'] == 'True':
                            entries = ['Refresh plot...',
                                       'Copy canvas to clipboard...']
                        for entry in entries:
                            action = QtWidgets.QAction(entry, self)
                            menu.addAction(action)
                        menu.triggered[QtWidgets.QAction].connect(self.popup_canvas)
                        menu.popup(QtGui.QCursor.pos())
                else:
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
                            print('leftmouseclickcbar')
                            data.view_settings['Minimum'] = new_value
                            data.reset_midpoint()
                        elif event.button == 2:
                            print('middlemouseclickcbar')
                            data.view_settings['Midpoint'] = new_value
                        elif event.button == 3:
                            print('rightmouseclickcbar')
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
            try:
                plot_data.linecut.remove()
                del plot_data.linecut
                plot_data.linecut_window.running = False
                self.canvas.draw()
            except:
                pass
        elif signal.text() == 'Refresh plot...':
            plot_data.refresh_data(update_color_limits=False, refresh_unit_conversion=False)
            self.update_plots()
        elif (signal.text() == 'Plot horizontal linecuts...' or
            signal.text() == 'Plot vertical linecuts...'):
            if signal.text() == 'Plot horizontal linecuts...':
                plot_data.multi_orientation = 'horizontal'
            elif signal.text() == 'Plot vertical linecuts...':
                plot_data.multi_orientation = 'vertical'
            plot_data.multi_linecuts_window = LineCutWindow(multiple=True)
            
            plot_data.update_multiple_linecuts()
            print(plot_data.multi_linecuts_window.isVisible())
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
            x1, y1 = plot_data.linecut_from
            plot_data.drawing_diagonal_linecut = True
            self.canvas.draw()
        elif signal.text() == 'Diagonal linecut to...':
            plot_data.orientation = 'diagonal'
            try:
                plot_data.linecut_window
            except AttributeError:
                plot_data.linecut_window = LineCutWindow()
            plot_data.linecut_window.running = True
            plot_data.update_linecut()
            self.canvas.draw()
            plot_data.drawing_diagonal_linecut = False
            plot_data.linecut_window.activateWindow()            
        elif signal.text() == 'FFT vertical...':
            plot_data.fft_orientation = 'vertical'
            plot_data.open_fft_window()
        elif signal.text() == 'FFT horizontal...':
            plot_data.fft_orientation = 'horizontal'
            plot_data.open_fft_window()
        elif signal.text() in plot_data.channels:
            channel_index = plot_data.channels.index(signal.text())
            plot_data.settings['columns'] = plot_data.settings['columns'][:-1]+str(channel_index)
            if plot_data.settings['2D'] == 'True':
                plot_data.columns[1] = channel_index
                plot_data.settings['ylabel'] = plot_data.channels[plot_data.columns[1]]
            else:                
                plot_data.columns[2] = channel_index
                plot_data.settings['clabel'] = plot_data.channels[plot_data.columns[2]]
            plot_data.refresh_data(update_color_limits=True, refresh_unit_conversion=True)
            self.update_plots()
            self.show_current_all()
        elif signal.text() == 'Copy canvas to clipboard...':
            del plot_data.cursor
            self.canvas.draw()
            self.clipboard = QtWidgets.QApplication.clipboard()
            self.pixmap_canvas = QtGui.QPixmap(self.canvas.grab())
            self.clipboard.setPixmap(self.pixmap_canvas)
            plot_data.cursor = Cursor(plot_data.axes, useblit=True, color='grey', linewidth=0.5)
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
                if data.settings['2D'] == 'False':
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
                    if y > 0.5:    
                        new_max = max_map + event.step*(max_map-min_map)*0.02
                        data.view_settings['Maximum'] = new_max
                    else:
                        new_min = min_map + event.step*(max_map-min_map)*0.02
                        data.view_settings['Minimum'] = new_min
                    data.reset_midpoint()
                    data.apply_view_settings()
                    self.canvas.draw()
                    self.show_current_view_settings()
                    
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

    
class Data3D:
    def __init__(self, filepath, existing_data=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.default_settings = {
                'title': '', 'xlabel': 'Gate Voltage (V)',
                'ylabel': 'Bias Voltage (mV)', 'clabel': 'd$I$/d$V$ ($\mu$S)',
                'titlesize': 'x-large', 'labelsize': 'xx-large', 'ticksize': 'x-large', 
                'columns': DEFAULT_COLUMNS, 'colorbar': 'True', 'minorticks': 'False', 
                'delimiter': '', 'lut': '512', 'rasterized': 'True', 'dpi': '300', 
                'transparent': 'False', 'frameon': 'False', '2D': 'False', 'rc-filter': ''}
        self.default_filters = []
        self.filters = copy.deepcopy(self.default_filters)
        self.settings = self.default_settings.copy()
        self.old_settings = self.default_settings.copy()
        self.columns = [int(self.default_settings['columns'].split(',')[i]) for i in range(3)]
        if existing_data:
            self.from_npy_file = True
            self.raw_data = existing_data
        else:
            self.from_npy_file = False
            self.load_data_from_file(filepath)
        self.processed_to_raw()
        meta_file_exists = os.path.isfile(os.path.dirname(self.filepath)+'/meta.json')
        if self.filename == 'data.dat' and meta_file_exists and self.from_npy_file == False: # Copenhagen meta data file
            self.interpret_meta_file()
            self.rcfilter_correct = DEFAULT_VALUE_RCFILTER_CORRECT
            self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=True)
        else:
            self.channels = None
            self.meta_data = None
            self.meta_data_name = None
            self.rcfilter_correct = False
            self.apply_all_filters(update_color_limits=False, refresh_unit_conversion=False)
        min_map = np.min(self.processed_data[2])
        max_map = np.max(self.processed_data[2])
        mid_map = 0.5*(min_map+max_map)
        self.default_view_settings = {
                'Minimum': min_map, 'Maximum': max_map, 'Midpoint': mid_map,
                'Color Map': 'magma', 'Color Map Type': 'Uniform',
                'Norm': MidpointNormalize(vmin=min_map, vmax=max_map, midpoint=mid_map),
                'Locked': 0, 'MidLock': 0, 'Reverse': 2}
        self.view_settings = self.default_view_settings.copy()
        self.selected_indices = [0, 0]
        self.figure = None
        self.axes = None
        self.image = None
        self.cbar = None
        self.orientation = None
        self.multi_orientation = None
        self.cropping = False
        self.drawing_diagonal_linecut = False        
        
    def load_data_from_file(self, filepath):
        if PRINT_FUNCTION_CALLS:
            print('load_data_from_file')
        column_data = np.genfromtxt(filepath, delimiter=self.settings['delimiter'])
        unique_values, indices = np.unique(column_data[:,self.columns[0]], return_index=True)
        if len(indices) > 1: # if first column has more than one repeated value
            l0 = np.sort(indices)[1]
        else:
            l0 = len(column_data[:,self.columns[0]])
        l1 = len(indices) 
        if len(column_data[np.sort(indices)[-1]::,0]) < l0:
            l1 = l1-1
        if column_data[:,self.columns[0]][1] != column_data[:,self.columns[0]][0]: # if file is 2D
            self.settings['2D'] = 'True'
            xq, yq = np.meshgrid([-1.,0.,1.], column_data[:,self.columns[0]])
            zq = np.array([column_data[:,self.columns[1]]]*3)
            self.raw_data = [xq.transpose(), yq.transpose(), zq]
        else:
            self.settings['2D'] = 'False'
            if indices[1] > indices[0]:
                self.raw_data = [np.reshape(column_data[:l0*l1,x], (l1,l0)) for x in range(column_data.shape[1])]
            else:
                self.raw_data = [np.reshape(column_data[l0*l1-1::-1,x], (l1,l0)) for x in range(column_data.shape[1])]
            if self.raw_data[1][0,0] > self.raw_data[1][0,1]:
                self.raw_data = [np.fliplr(self.raw_data[x]) for x in range(column_data.shape[1])]

    def interpret_meta_file(self):
        if PRINT_FUNCTION_CALLS:
            print('interpret_meta_file')
        self.metaFile = os.path.dirname(self.filepath)+'/meta.json'
        with open(self.metaFile) as f:
            self.meta_data = json.load(f)
        self.channels = [channel['name'] for channel in self.meta_data['columns']]
        self.settings['xlabel'] = self.channels[self.columns[0]]
        self.settings['ylabel'] = self.channels[self.columns[1]]
        self.settings['clabel'] = self.channels[self.columns[2]]
        self.meta_data_name = self.meta_data['timestamp']+' '+self.meta_data['name']
        if 'source' in self.channels and 'dc_curr' in self.channels:
            DEFAULT_RC_FILTER = 8240
            self.settings['rc-filter'] = str(DEFAULT_RC_FILTER)
    
    def correct_for_rcfilters(self):
        if PRINT_FUNCTION_CALLS:
            print('correct_for_rcfilters')
        if 'source' in self.channels and 'dc_curr' in self.channels:
            source_index = self.channels.index('source')
            if source_index in self.columns[:3-(self.settings['2D']=='True')]:
                print('Correcting source for total rc-filter resistance of',self.settings['rc-filter'],'Ohm...')
                source_divider = self.meta_data['setup']['meta']['source_divider']
                curr_index = self.channels.index('dc_curr')                    
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                source_data = self.raw_data[source_index] / source_divider
                curr_data = self.raw_data[curr_index] / curr_amp
                source_corrected = (source_data - float(self.settings['rc-filter'])*curr_data)*source_divider
                self.processed_data[self.columns.index(source_index)] = np.copy(source_corrected)
        
        if 'lockin_curr/X' in self.channels:
            lockin_index = self.channels.index('lockin_curr/X')
            if lockin_index in self.columns[:3-(self.settings['2D']=='True')]:
                print('Correcting lockin_curr/X for total rc-filter resistance of',self.settings['rc-filter'],'Ohm...')
                for instrument in self.meta_data['register']['instruments']:
                    if instrument['name'] == 'lockin_curr':
                        break
                sine_amplitude = float(instrument['config']['SLVL'])
                lockin_divider = self.meta_data['setup']['meta']['lock_sig_divider']
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                conversion = curr_amp*sine_amplitude/lockin_divider # to Siemens (1/Ohm)
                lockin_data = self.raw_data[lockin_index]/conversion # in Siemens
                lockin_corrected = lockin_data/(1.0-float(self.settings['rc-filter'])*lockin_data)*conversion
                self.processed_data[self.columns.index(lockin_index)] = np.copy(lockin_corrected)                
    
    def meta_unit_conversion(self):
        if PRINT_FUNCTION_CALLS:
            print('meta_unit_conversion')
        if 'source' in self.channels:
            source_index = self.channels.index('source')
            if source_index in self.columns[:3-(self.settings['2D']=='True')]:
                source_divider = self.meta_data['setup']['meta']['source_divider']
                SOURCE_UNIT = 1e-3 # Convert V to mV  
                print('Converting source units to millivolts...')
                if source_divider*SOURCE_UNIT != 1.0:
                    divide = '%.3g' % (source_divider*SOURCE_UNIT)
                    axis = ['X','Y','Z'][self.columns.index(source_index)+(self.settings['2D']=='True')] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(source_index) == 1:
                    self.settings['ylabel'] = 'Bias Voltage (mV)'
                elif self.columns.index(source_index) == 0:
                    self.settings['xlabel'] = 'Bias Voltage (mV)'
        
        fine_gates = [channel for channel in self.channels if channel[0] == 'g' and channel[-1] == 'f']
        for gate in fine_gates:
            gate_index = self.channels.index(gate)
            if gate_index in self.columns[:3-(self.settings['2D']=='True')]:                              
                print('Converting '+gate+' units to volts...')
                DAC_FINE_DIVIDER = 200
                DAC_FINE_OFFSET = 0.05
                divide = '%.3g' % DAC_FINE_DIVIDER
                gate_axis = ['X','Y','Z'][self.columns.index(gate_index)+(self.settings['2D']=='True')] 
                self.filters.append({'Name': 'Divide', 'Method': gate_axis, 
                                     'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if DAC_FINE_OFFSET != 0.0:
                    fine_offset = '%.3g' % DAC_FINE_OFFSET
                    self.filters.append({'Name': 'Offset', 'Method': gate_axis, 
                                         'Setting 1': fine_offset, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(gate_index) == 1:
                    self.settings['ylabel'] = 'Gate Voltage '+gate[1:]+' (V)'
                elif self.columns.index(gate_index) == 0:
                    self.settings['xlabel'] = 'Gate Voltage '+gate[1:]+' (V)'
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
        
        coarse_gates = [channel for channel in self.channels if channel[0] == 'g' and len(channel) == 2]
        for gate in coarse_gates:
            gate_index = self.channels.index(gate)
            if gate_index in self.columns[:3-(self.settings['2D']=='True')]:
                if self.columns.index(gate_index) == 1:
                    self.settings['ylabel'] = 'Gate Voltage '+gate[1]+' (V)'
                elif self.columns.index(gate_index) == 0:
                    self.settings['xlabel'] = 'Gate Voltage '+gate[1]+' (V)'  
            
        if 'dc_curr' in self.channels:
            curr_index = self.channels.index('dc_curr')
            if curr_index in self.columns[:3-(self.settings['2D']=='True')]:
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                print('Converting dc_curr units to nano-amperes...')
                CURR_UNIT = 1e-9 # Ampere to nano-ampere
                if curr_amp*CURR_UNIT != 1.0:
                    divide = '%.3g' % (curr_amp*CURR_UNIT)
                    axis = ['X','Y','Z'][self.columns.index(curr_index)+(self.settings['2D']=='True')] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(curr_index) == 2:
                    self.settings['clabel'] = 'DC Current (nA)'
                elif self.settings['2D'] == 'True' and self.columns.index(curr_index) == 1:
                    self.settings['ylabel'] = 'DC Current (nA)'
        
        if 'lockin_curr/X' in self.channels:
            lockin_index = self.channels.index('lockin_curr/X')
            if lockin_index in self.columns[:3-(self.settings['2D']=='True')]:
                for instrument in self.meta_data['register']['instruments']:
                    if instrument['name'] == 'lockin_curr':
                        break
                sine_amplitude = float(instrument['config']['SLVL'])
                lockin_divider = self.meta_data['setup']['meta']['lock_sig_divider']
                curr_amp = self.meta_data['setup']['meta']['current_amp']
                
                print('Correcting lockin_curr/X units to microsiemens...')
                LOCKIN_UNIT = 1e-6 # Siemens to microsiemens
                conversion_factor = curr_amp*sine_amplitude/lockin_divider*LOCKIN_UNIT
                if conversion_factor != 1.0:
                    divide = '%.3g' % conversion_factor
                    axis = ['X','Y','Z'][self.columns.index(lockin_index)+(self.settings['2D']=='True')] 
                    self.filters.append({'Name': 'Divide', 'Method': axis, 
                                         'Setting 1': divide, 'Setting 2': '', 'Checked': 2})
                if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                    multiply_e2h = '%.6g' % 0.0258128
                    axis = ['X','Y','Z'][self.columns.index(lockin_index)+(self.settings['2D']=='True')] 
                    self.filters.append({'Name': 'Multiply', 'Method': axis, 
                                         'Setting 1': multiply_e2h, 'Setting 2': '', 'Checked': 2})
                if self.columns.index(lockin_index) == 2:
                    if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                        self.settings['clabel'] = 'd$I$/d$V$ ($e^2/h$)'
                    else:
                        self.settings['clabel'] = 'd$I$/d$V$ ($\mu$S)'
                elif self.settings['2D'] == 'True' and self.columns.index(lockin_index) == 1:
                    if CONVERT_MICROSIEMENS_TO_ESQUAREDH:
                        self.settings['clabel'] = 'd$I$/d$V$ ($e^2/h$)'
                    else:
                        self.settings['clabel'] = 'd$I$/d$V$ ($\mu$S)'
               
    def processed_to_raw(self):
        if PRINT_FUNCTION_CALLS:
            print('processed_to_raw')
        self.processed_data = [np.copy(self.raw_data[i]) for i in self.columns]
    
    def refresh_data(self, update_color_limits=False, refresh_unit_conversion=False):
        if PRINT_FUNCTION_CALLS:
            print('refresh_data')
        if self.from_npy_file:
            print('.npy files cannot be refreshed...')
        else:
            self.load_data_from_file(self.filepath)
            self.apply_all_filters(update_color_limits, refresh_unit_conversion)        
    
    def add_plot(self):
        if PRINT_FUNCTION_CALLS:
            print('add_plot')
        data = self.processed_data
        cmap = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap = cmap+'_r'
        self.image = self.axes.pcolormesh(data[0], data[1], data[2], 
                                  norm=self.view_settings['Norm'], 
                                  cmap=cm.get_cmap(cmap, lut=int(self.settings['lut'])),
                                  rasterized=self.settings['rasterized']=='True')
        if self.settings['colorbar'] == 'True':
            self.cbar = self.figure.colorbar(self.image, orientation='vertical')
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()
      
    def add_plot_2d(self):
        if PRINT_FUNCTION_CALLS:
            print('add_plot_2d')
        data = self.processed_data
        cmap = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap = cmap+'_r'
        self.image = self.axes.plot(data[1][0,:], data[2][0,:], color=cm.get_cmap(cmap)(0.5))
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()        
        
    def reset_view_settings(self, overrule=False):
        if PRINT_FUNCTION_CALLS:
            print('reset_view_settings')
        if self.view_settings['Locked'] == 0 or overrule == True:
            self.view_settings['Minimum'] = np.min(self.processed_data[2])
            self.view_settings['Maximum'] = np.max(self.processed_data[2])
            self.view_settings['Midpoint'] = 0.5*(np.min(self.processed_data[2])+
                             np.max(self.processed_data[2]))
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
        self.axes.tick_params(labelsize=settings['ticksize'])
        if settings['minorticks'] == 'True':    
            self.axes.minorticks_on()
        if settings['title'] == '<filename>':
            self.axes.set_title(self.filename, size=settings['titlesize'])
        elif settings['title'] == '<metadataname>':
            self.axes.set_title("\n".join(wrap(self.meta_data_name,80)), size=settings['titlesize'])
        else:
            self.axes.set_title(settings['title'], size=settings['titlesize'])
        if settings['colorbar'] == 'True' and settings['2D'] == 'False':
            self.cbar.ax.set_title(settings['clabel'], size=settings['labelsize'])
            self.cbar.ax.tick_params(labelsize=settings['ticksize'])
            

    def apply_view_settings(self):
        if PRINT_FUNCTION_CALLS:
            print('apply_view_settings')
        if self.settings['2D'] == 'False':
            self.view_settings['Norm'] = MidpointNormalize(
                    vmin=self.view_settings['Minimum'], 
                    vmax=self.view_settings['Maximum'], 
                    midpoint=self.view_settings['Midpoint'])
            self.image.set_norm(self.view_settings['Norm'])
            if self.settings['colorbar'] == 'True':
                self.cbar.update_bruteforce(self.image)
                self.cbar.ax.set_title(self.settings['clabel'], 
                                       size=self.settings['labelsize'])
                self.cbar.ax.tick_params(labelsize=self.settings['ticksize'])          
    
    def apply_colormap(self):
        if PRINT_FUNCTION_CALLS:
            print('apply_colormap '+self.view_settings['Color Map'])
        cmap = self.view_settings['Color Map']
        if self.view_settings['Reverse']:
            cmap = cmap+'_r'
        if self.settings['2D'] == 'False':
            self.image.set_cmap(cm.get_cmap(cmap, lut=int(self.settings['lut'])))
        else:
            self.image[0].set_color(cm.get_cmap(cmap)(0.5))
            
    
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
        if self.rcfilter_correct:
            self.correct_for_rcfilters()
        if self.meta_data and refresh_unit_conversion:
            self.filters = []
            self.meta_unit_conversion()
        for _, filter_settings in enumerate(self.filters):
            if filter_settings['Checked']:
                if PRINT_FUNCTION_CALLS:
                    print('Applying '+filter_settings['Name']+'...')
                self.processed_data = filters.apply(self.processed_data, filter_settings)
        if update_color_limits:
            self.reset_view_settings()
            try:
                self.apply_view_settings()
            except:
                pass
            
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
                        y=value, linestyle='dashed', linewidth=0.5, color='k')
            elif self.orientation == 'vertical':
                x = self.processed_data[1][self.selected_indices[0],:]
                y = self.processed_data[2][self.selected_indices[0],:]
                value = self.processed_data[0][self.selected_indices[0],0]
                self.linecut_window.xlabel = self.settings['ylabel']
                self.linecut_window.zlabel = self.settings['xlabel']
                self.linecut_window.title = self.settings['xlabel']+' = '+str(value)
                self.linecut = self.axes.axvline(
                        x=value, linestyle='dashed', linewidth=0.5, color='k')
            elif self.orientation == 'diagonal':
                i_x0, i_y0 = self.linecut_from
                i_x1, i_y1 = self.linecut_to
                n = 200
                x_diag, y_diag = np.linspace(i_x0, i_x1, n), np.linspace(i_y0, i_y1, n)
                y = map_coordinates(self.processed_data[2], np.vstack((x_diag, y_diag)))
                x = np.linspace(0, 1, n)
                self.linecut_window.xlabel = ''
                self.linecut_window.zlabel = ''
                self.linecut_window.title = ''
                self.linecut = self.axes.plot([self.processed_data[0][i_x0,0], self.processed_data[0][i_x1,0]],
                                              [self.processed_data[1][0,i_y0], self.processed_data[1][0,i_y1]],
                                              linestyle='dashed', linewidth=0.5, color='k')[0]
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
        self.multi_linecuts_window.data = self.processed_data
        self.multi_linecuts_window.draw_plots()
        self.multi_linecuts_window.show()
        
    def open_fft_window(self):
        if PRINT_FUNCTION_CALLS:
            print('open_fft_window')
        if self.fft_orientation == 'vertical':
            self.fft = np.fft.rfft(self.processed_data[2], axis=1)
        elif self.fft_orientation == 'horizontal':
            self.fft = np.fft.rfft(self.processed_data[2], axis=0)
        self.fft_window = FFTWindow(self.fft)
        self.fft_window.show()


class LineCutWindow(QtWidgets.QWidget):
    def __init__(self, multiple=False):
        super(self.__class__, self).__init__()
        self.setWindowTitle('Inspectra Gadget - Linecut Window')
        self.multiple = multiple
        self.running = True
        self.resize(600, 600)
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
        if multiple:
            self.init_multiple()
        self.vertical_layout.addWidget(self.canvas)
        self.vertical_layout.addWidget(self.navi_toolbar)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.save_image_button)
        self.button_layout.addStretch()
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
        self.offset_label = QtWidgets.QLabel('Offset')
        self.offset_line_edit = QtWidgets.QLineEdit('0')
        self.offset_line_edit.setFixedSize(70,20)
        self.offset_line_edit.editingFinished.connect(self.draw_plots)
        self.check_legend = QtWidgets.QCheckBox('Legend')
        self.check_legend.clicked.connect(self.draw_plots)
        self.control_layout.addWidget(self.number_label)
        self.control_layout.addWidget(self.number_line_edit)
        self.control_layout.addWidget(self.all_lines_button)
        self.control_layout.addWidget(self.offset_label)
        self.control_layout.addWidget(self.offset_line_edit)
        self.control_layout.addWidget(self.check_legend)
        self.cmaps = [('Uniform', [
                    'viridis', 'magma', 'plasma', 'inferno']),
                 ('Sequential', [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                 ('Sequential 2', [
                    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper']),
                 ('Diverging', [
                    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                 ('Qualitative', [
                    'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c']),
                 ('Miscellaneous', [
                    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
        self.colormap_box = QtWidgets.QComboBox()
        self.colormap_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colormap_type_box = QtWidgets.QComboBox()
        self.colormap_type_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        for n in range(6):    
            self.colormap_type_box.addItem(self.cmaps[n][0])
        self.colormap_box.addItems(self.cmaps[0][1])
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
        self.fit_layout.addWidget(self.manual_box)
        self.fit_layout.addWidget(self.guess_edit)
        self.fit_layout.addStretch()
        self.fit_layout.addWidget(self.pars_label)
        self.fit_layout.addWidget(self.fit_box)        
        self.fit_layout.addWidget(self.fit_button)
        self.vertical_layout.addLayout(self.fit_layout)
    
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
        if self.multiple:
            self.draw_plots()
        else:
            self.draw_plot(self.x, self.y)
        self.draw_fits()
        self.plot_parameters()
        
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
        self.colormap_box.addItems(self.cmaps[self.colormap_type_box.currentIndex()][1])
        self.colormap_box.currentIndexChanged.connect(self.draw_plots)
        self.draw_plots()
         
    def draw_plot(self, x, y):
        self.running = True
        self.x, self.y = x, y
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.image = self.axes.plot(x, y, 'C3', linewidth=0.8)
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()

    def draw_plots(self):
        self.running = True
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        rows, cols = self.data[0].shape
        self.number = int(self.number_line_edit.text())
        self.offset = float(self.offset_line_edit.text())
        selected_colormap = cm.get_cmap(self.colormap_box.currentText())
        line_colors = selected_colormap(np.linspace(0,1,self.number))
        if self.orientation == 'horizontal':
            indices = np.linspace(0, cols-1, self.number, dtype=int)
            self.x = self.data[0][:,indices].transpose()
            self.y = self.data[2][:,indices].transpose()
            self.z = self.data[1][:,indices].transpose()
        elif self.orientation == 'vertical':
            indices = np.linspace(0, rows-1, self.number, dtype=int)
            self.x = self.data[1][indices,:]
            self.y = self.data[2][indices,:]
            self.z = self.data[0][indices,:]
        for index in range(self.number):
            self.axes.plot(self.x[index], self.y[index]+index*self.offset, 
                           color=line_colors[index], linewidth=0.8, 
                           label=str(self.z[index][0]))
        if self.check_legend.checkState():
            self.axes.legend(title=self.zlabel)
        self.cursor = Cursor(self.axes, useblit=True, color='grey', linewidth=0.5)
        self.apply_plot_settings()
              
    def draw_fits(self):
        if self.multiple:
            for index in range(self.number):
                self.axes.plot(self.x[index], self.y_fit[index]+index*self.offset, 
                               'k--', linewidth=0.8)
        else:
            self.axes.plot(self.x, self.y_fit, 'k--', linewidth=0.8)            
        self.canvas.draw()
    
    def apply_plot_settings(self):
        #print('apply_plot_settings')
        self.axes.set_xlabel(self.xlabel, size='xx-large')
        self.axes.set_ylabel(self.ylabel, size='xx-large')
        self.axes.tick_params(labelsize='x-large')
        self.axes.set_title(self.title, size='x-large')
        self.canvas.draw()
        
    def clicked_all_lines(self):
        rows, cols = self.data[0].shape
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
        self.axes.plot(self.x, self.y,'C2.')
        self.axes.set_xlabel(xlabel, size='xx-large')
        self.axes.set_ylabel(ylabel, size='xx-large')
        self.axes.tick_params(labelsize='x-large')
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
        self.image = self.axes.pcolormesh(self.fft, norm=LogNorm(vmin=self.fft.min(), vmax=self.fft.max()))
        self.cbar = self.figure.colorbar(self.image, orientation='vertical')
        self.figure.tight_layout(pad=2)
        self.axes.tick_params(labelsize='x-large')
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
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    edit_window = Editor()
    edit_window.show()
    app.exec_()

if __name__ == '__main__':
    main()