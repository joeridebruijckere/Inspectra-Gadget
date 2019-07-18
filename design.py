# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1520, 880)
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.central_widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.control_layout = QtWidgets.QVBoxLayout()
        self.control_layout.setObjectName("control_layout")
        self.files_box = QtWidgets.QGroupBox(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.files_box.sizePolicy().hasHeightForWidth())
        self.files_box.setSizePolicy(sizePolicy)
        self.files_box.setObjectName("files_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.files_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.files_layout = QtWidgets.QGridLayout()
        self.files_layout.setObjectName("files_layout")
        self.up_file_button = QtWidgets.QPushButton(self.files_box)
        self.up_file_button.setMinimumSize(QtCore.QSize(110, 0))
        self.up_file_button.setObjectName("up_file_button")
        self.files_layout.addWidget(self.up_file_button, 4, 0, 1, 1)
        self.file_list = QtWidgets.QListWidget(self.files_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.file_list.sizePolicy().hasHeightForWidth())
        self.file_list.setSizePolicy(sizePolicy)
        self.file_list.setObjectName("file_list")
        self.files_layout.addWidget(self.file_list, 1, 0, 1, 3)
        self.down_file_button = QtWidgets.QPushButton(self.files_box)
        self.down_file_button.setMinimumSize(QtCore.QSize(110, 0))
        self.down_file_button.setObjectName("down_file_button")
        self.files_layout.addWidget(self.down_file_button, 4, 1, 1, 1)
        self.open_files_button = QtWidgets.QPushButton(self.files_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.open_files_button.sizePolicy().hasHeightForWidth())
        self.open_files_button.setSizePolicy(sizePolicy)
        self.open_files_button.setMinimumSize(QtCore.QSize(110, 0))
        self.open_files_button.setObjectName("open_files_button")
        self.files_layout.addWidget(self.open_files_button, 0, 0, 1, 1)
        self.delete_files_button = QtWidgets.QPushButton(self.files_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.delete_files_button.sizePolicy().hasHeightForWidth())
        self.delete_files_button.setSizePolicy(sizePolicy)
        self.delete_files_button.setMinimumSize(QtCore.QSize(110, 0))
        self.delete_files_button.setObjectName("delete_files_button")
        self.files_layout.addWidget(self.delete_files_button, 0, 1, 1, 1)
        self.clear_files_button = QtWidgets.QPushButton(self.files_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_files_button.sizePolicy().hasHeightForWidth())
        self.clear_files_button.setSizePolicy(sizePolicy)
        self.clear_files_button.setMinimumSize(QtCore.QSize(110, 0))
        self.clear_files_button.setObjectName("clear_files_button")
        self.files_layout.addWidget(self.clear_files_button, 0, 2, 1, 1)
        self.refresh_file_button = QtWidgets.QPushButton(self.files_box)
        self.refresh_file_button.setMinimumSize(QtCore.QSize(110, 0))
        self.refresh_file_button.setObjectName("refresh_file_button")
        self.files_layout.addWidget(self.refresh_file_button, 4, 2, 1, 1)
        self.gridLayout_2.addLayout(self.files_layout, 0, 0, 1, 1)
        self.control_layout.addWidget(self.files_box)
        self.settings_box = QtWidgets.QGroupBox(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settings_box.sizePolicy().hasHeightForWidth())
        self.settings_box.setSizePolicy(sizePolicy)
        self.settings_box.setObjectName("settings_box")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.settings_box)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.settings_layout = QtWidgets.QGridLayout()
        self.settings_layout.setObjectName("settings_layout")
        self.copy_settings_button = QtWidgets.QPushButton(self.settings_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.copy_settings_button.sizePolicy().hasHeightForWidth())
        self.copy_settings_button.setSizePolicy(sizePolicy)
        self.copy_settings_button.setMinimumSize(QtCore.QSize(110, 0))
        self.copy_settings_button.setObjectName("copy_settings_button")
        self.settings_layout.addWidget(self.copy_settings_button, 1, 0, 1, 1)
        self.settings_table = QtWidgets.QTableWidget(self.settings_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settings_table.sizePolicy().hasHeightForWidth())
        self.settings_table.setSizePolicy(sizePolicy)
        self.settings_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.settings_table.setObjectName("settings_table")
        self.settings_table.setColumnCount(0)
        self.settings_table.setRowCount(0)
        self.settings_table.horizontalHeader().setVisible(False)
        self.settings_table.horizontalHeader().setDefaultSectionSize(80)
        self.settings_table.horizontalHeader().setStretchLastSection(True)
        self.settings_table.verticalHeader().setVisible(False)
        self.settings_table.verticalHeader().setDefaultSectionSize(23)
        self.settings_layout.addWidget(self.settings_table, 0, 0, 1, 3)
        self.paste_settings_button = QtWidgets.QPushButton(self.settings_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.paste_settings_button.sizePolicy().hasHeightForWidth())
        self.paste_settings_button.setSizePolicy(sizePolicy)
        self.paste_settings_button.setMinimumSize(QtCore.QSize(110, 0))
        self.paste_settings_button.setObjectName("paste_settings_button")
        self.settings_layout.addWidget(self.paste_settings_button, 1, 1, 1, 1)
        self.reset_settings_button = QtWidgets.QPushButton(self.settings_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.reset_settings_button.sizePolicy().hasHeightForWidth())
        self.reset_settings_button.setSizePolicy(sizePolicy)
        self.reset_settings_button.setMinimumSize(QtCore.QSize(110, 0))
        self.reset_settings_button.setObjectName("reset_settings_button")
        self.settings_layout.addWidget(self.reset_settings_button, 1, 2, 1, 1)
        self.gridLayout_4.addLayout(self.settings_layout, 0, 0, 1, 1)
        self.control_layout.addWidget(self.settings_box)
        self.filters_box = QtWidgets.QGroupBox(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filters_box.sizePolicy().hasHeightForWidth())
        self.filters_box.setSizePolicy(sizePolicy)
        self.filters_box.setObjectName("filters_box")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.filters_box)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.filters_layout = QtWidgets.QGridLayout()
        self.filters_layout.setObjectName("filters_layout")
        self.delete_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.delete_filters_button.sizePolicy().hasHeightForWidth())
        self.delete_filters_button.setSizePolicy(sizePolicy)
        self.delete_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.delete_filters_button.setObjectName("delete_filters_button")
        self.filters_layout.addWidget(self.delete_filters_button, 0, 2, 1, 1)
        self.filters_combobox = QtWidgets.QComboBox(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filters_combobox.sizePolicy().hasHeightForWidth())
        self.filters_combobox.setSizePolicy(sizePolicy)
        self.filters_combobox.setObjectName("filters_combobox")
        self.filters_layout.addWidget(self.filters_combobox, 0, 0, 1, 2)
        self.clear_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_filters_button.sizePolicy().hasHeightForWidth())
        self.clear_filters_button.setSizePolicy(sizePolicy)
        self.clear_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.clear_filters_button.setObjectName("clear_filters_button")
        self.filters_layout.addWidget(self.clear_filters_button, 5, 2, 1, 1)
        self.up_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_filters_button.sizePolicy().hasHeightForWidth())
        self.up_filters_button.setSizePolicy(sizePolicy)
        self.up_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.up_filters_button.setObjectName("up_filters_button")
        self.filters_layout.addWidget(self.up_filters_button, 4, 0, 1, 1)
        self.filters_table = QtWidgets.QTableWidget(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filters_table.sizePolicy().hasHeightForWidth())
        self.filters_table.setSizePolicy(sizePolicy)
        self.filters_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.filters_table.setObjectName("filters_table")
        self.filters_table.setColumnCount(0)
        self.filters_table.setRowCount(0)
        self.filters_table.horizontalHeader().setVisible(False)
        self.filters_table.horizontalHeader().setCascadingSectionResizes(False)
        self.filters_table.horizontalHeader().setDefaultSectionSize(40)
        self.filters_table.horizontalHeader().setMinimumSectionSize(31)
        self.filters_table.horizontalHeader().setStretchLastSection(False)
        self.filters_table.verticalHeader().setVisible(False)
        self.filters_table.verticalHeader().setDefaultSectionSize(25)
        self.filters_table.verticalHeader().setMinimumSectionSize(25)
        self.filters_layout.addWidget(self.filters_table, 3, 0, 1, 3)
        self.down_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_filters_button.sizePolicy().hasHeightForWidth())
        self.down_filters_button.setSizePolicy(sizePolicy)
        self.down_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.down_filters_button.setObjectName("down_filters_button")
        self.filters_layout.addWidget(self.down_filters_button, 4, 1, 1, 1)
        self.copy_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.copy_filters_button.sizePolicy().hasHeightForWidth())
        self.copy_filters_button.setSizePolicy(sizePolicy)
        self.copy_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.copy_filters_button.setObjectName("copy_filters_button")
        self.filters_layout.addWidget(self.copy_filters_button, 5, 0, 1, 1)
        self.paste_filters_button = QtWidgets.QPushButton(self.filters_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.paste_filters_button.sizePolicy().hasHeightForWidth())
        self.paste_filters_button.setSizePolicy(sizePolicy)
        self.paste_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.paste_filters_button.setObjectName("paste_filters_button")
        self.filters_layout.addWidget(self.paste_filters_button, 5, 1, 1, 1)
        self.load_filters_button = QtWidgets.QPushButton(self.filters_box)
        self.load_filters_button.setMinimumSize(QtCore.QSize(110, 0))
        self.load_filters_button.setObjectName("load_filters_button")
        self.filters_layout.addWidget(self.load_filters_button, 4, 2, 1, 1)
        self.gridLayout_6.addLayout(self.filters_layout, 0, 0, 1, 1)
        self.control_layout.addWidget(self.filters_box)
        self.horizontalLayout.addLayout(self.control_layout)
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.vertical_layout.setObjectName("vertical_layout")
        self.graph_layout = QtWidgets.QGridLayout()
        self.graph_layout.setObjectName("graph_layout")
        self.vertical_layout.addLayout(self.graph_layout)
        self.view_layout = QtWidgets.QGridLayout()
        self.view_layout.setObjectName("view_layout")
        self.reset_limits_button = QtWidgets.QPushButton(self.central_widget)
        self.reset_limits_button.setObjectName("reset_limits_button")
        self.view_layout.addWidget(self.reset_limits_button, 0, 3, 1, 1)
        self.min_label = QtWidgets.QLabel(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.min_label.sizePolicy().hasHeightForWidth())
        self.min_label.setSizePolicy(sizePolicy)
        self.min_label.setObjectName("min_label")
        self.view_layout.addWidget(self.min_label, 0, 4, 1, 1)
        self.max_label = QtWidgets.QLabel(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_label.sizePolicy().hasHeightForWidth())
        self.max_label.setSizePolicy(sizePolicy)
        self.max_label.setObjectName("max_label")
        self.view_layout.addWidget(self.max_label, 0, 11, 1, 1)
        self.colormap_box = QtWidgets.QComboBox(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colormap_box.sizePolicy().hasHeightForWidth())
        self.colormap_box.setSizePolicy(sizePolicy)
        self.colormap_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.colormap_box.setObjectName("colormap_box")
        self.view_layout.addWidget(self.colormap_box, 0, 15, 1, 1)
        self.mid_line_edit = QtWidgets.QLineEdit(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mid_line_edit.sizePolicy().hasHeightForWidth())
        self.mid_line_edit.setSizePolicy(sizePolicy)
        self.mid_line_edit.setMinimumSize(QtCore.QSize(100, 0))
        self.mid_line_edit.setMaximumSize(QtCore.QSize(90, 16777215))
        self.mid_line_edit.setObjectName("mid_line_edit")
        self.view_layout.addWidget(self.mid_line_edit, 0, 8, 1, 3)
        self.max_line_edit = QtWidgets.QLineEdit(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_line_edit.sizePolicy().hasHeightForWidth())
        self.max_line_edit.setSizePolicy(sizePolicy)
        self.max_line_edit.setMinimumSize(QtCore.QSize(100, 0))
        self.max_line_edit.setMaximumSize(QtCore.QSize(90, 16777215))
        self.max_line_edit.setObjectName("max_line_edit")
        self.view_layout.addWidget(self.max_line_edit, 0, 12, 1, 1)
        self.reverse_colors_box = QtWidgets.QCheckBox(self.central_widget)
        self.reverse_colors_box.setObjectName("reverse_colors_box")
        self.view_layout.addWidget(self.reverse_colors_box, 0, 16, 1, 1)
        self.min_line_edit = QtWidgets.QLineEdit(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.min_line_edit.sizePolicy().hasHeightForWidth())
        self.min_line_edit.setSizePolicy(sizePolicy)
        self.min_line_edit.setMinimumSize(QtCore.QSize(100, 0))
        self.min_line_edit.setMaximumSize(QtCore.QSize(90, 16777215))
        self.min_line_edit.setObjectName("min_line_edit")
        self.view_layout.addWidget(self.min_line_edit, 0, 5, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.view_layout.addItem(spacerItem, 0, 0, 1, 1)
        self.lock_checkbox = QtWidgets.QCheckBox(self.central_widget)
        self.lock_checkbox.setObjectName("lock_checkbox")
        self.view_layout.addWidget(self.lock_checkbox, 0, 13, 1, 1)
        self.colormap_type_box = QtWidgets.QComboBox(self.central_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colormap_type_box.sizePolicy().hasHeightForWidth())
        self.colormap_type_box.setSizePolicy(sizePolicy)
        self.colormap_type_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.colormap_type_box.setObjectName("colormap_type_box")
        self.view_layout.addWidget(self.colormap_type_box, 0, 14, 1, 1)
        self.mid_checkbox = QtWidgets.QCheckBox(self.central_widget)
        self.mid_checkbox.setObjectName("mid_checkbox")
        self.view_layout.addWidget(self.mid_checkbox, 0, 7, 1, 1)
        self.paste_view_button = QtWidgets.QPushButton(self.central_widget)
        self.paste_view_button.setObjectName("paste_view_button")
        self.view_layout.addWidget(self.paste_view_button, 0, 2, 1, 1)
        self.copy_view_button = QtWidgets.QPushButton(self.central_widget)
        self.copy_view_button.setObjectName("copy_view_button")
        self.view_layout.addWidget(self.copy_view_button, 0, 1, 1, 1)
        self.vertical_layout.addLayout(self.view_layout)
        self.horizontalLayout.addLayout(self.vertical_layout)
        MainWindow.setCentralWidget(self.central_widget)
        self.menu_bar = QtWidgets.QMenuBar(MainWindow)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 1520, 26))
        self.menu_bar.setObjectName("menu_bar")
        self.menu_auto_refresh = QtWidgets.QMenu(self.menu_bar)
        self.menu_auto_refresh.setObjectName("menu_auto_refresh")
        self.menu_save_session = QtWidgets.QMenu(self.menu_bar)
        self.menu_save_session.setObjectName("menu_save_session")
        self.menu_merge_data = QtWidgets.QMenu(self.menu_bar)
        self.menu_merge_data.setObjectName("menu_merge_data")
        self.menu_save_image = QtWidgets.QMenu(self.menu_bar)
        self.menu_save_image.setObjectName("menu_save_image")
        MainWindow.setMenuBar(self.menu_bar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionNewWindow = QtWidgets.QAction(MainWindow)
        self.actionNewWindow.setObjectName("actionNewWindow")
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionClear_Window = QtWidgets.QAction(MainWindow)
        self.actionClear_Window.setObjectName("actionClear_Window")
        self.action_png = QtWidgets.QAction(MainWindow)
        self.action_png.setObjectName("action_png")
        self.action_pdf = QtWidgets.QAction(MainWindow)
        self.action_pdf.setObjectName("action_pdf")
        self.actionSession = QtWidgets.QAction(MainWindow)
        self.actionSession.setObjectName("actionSession")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.action_eps = QtWidgets.QAction(MainWindow)
        self.action_eps.setObjectName("action_eps")
        self.action_ps = QtWidgets.QAction(MainWindow)
        self.action_ps.setObjectName("action_ps")
        self.action_svg = QtWidgets.QAction(MainWindow)
        self.action_svg.setObjectName("action_svg")
        self.actionLoadSession = QtWidgets.QAction(MainWindow)
        self.actionLoadSession.setObjectName("actionLoadSession")
        self.action_save_image = QtWidgets.QAction(MainWindow)
        self.action_save_image.setObjectName("action_save_image")
        self.action_canvas_to_script = QtWidgets.QAction(MainWindow)
        self.action_canvas_to_script.setObjectName("action_canvas_to_script")
        self.actionEnable_Auto_Refresh = QtWidgets.QAction(MainWindow)
        self.actionEnable_Auto_Refresh.setObjectName("actionEnable_Auto_Refresh")
        self.actionEvery_10_s = QtWidgets.QAction(MainWindow)
        self.actionEvery_10_s.setObjectName("actionEvery_10_s")
        self.actionEvery_5_s = QtWidgets.QAction(MainWindow)
        self.actionEvery_5_s.setObjectName("actionEvery_5_s")
        self.action_refresh_5m = QtWidgets.QAction(MainWindow)
        self.action_refresh_5m.setObjectName("action_refresh_5m")
        self.actionEvery_5_minutes = QtWidgets.QAction(MainWindow)
        self.actionEvery_5_minutes.setObjectName("actionEvery_5_minutes")
        self.action_refresh_stop = QtWidgets.QAction(MainWindow)
        self.action_refresh_stop.setObjectName("action_refresh_stop")
        self.action_refresh_30s = QtWidgets.QAction(MainWindow)
        self.action_refresh_30s.setObjectName("action_refresh_30s")
        self.action_refresh_30m = QtWidgets.QAction(MainWindow)
        self.action_refresh_30m.setObjectName("action_refresh_30m")
        self.action_current_file = QtWidgets.QAction(MainWindow)
        self.action_current_file.setObjectName("action_current_file")
        self.action_all_files = QtWidgets.QAction(MainWindow)
        self.action_all_files.setObjectName("action_all_files")
        self.action_filters = QtWidgets.QAction(MainWindow)
        self.action_filters.setObjectName("action_filters")
        self.action_merge_raw = QtWidgets.QAction(MainWindow)
        self.action_merge_raw.setObjectName("action_merge_raw")
        self.action_merge_processed = QtWidgets.QAction(MainWindow)
        self.action_merge_processed.setObjectName("action_merge_processed")
        self.action_save_canvas = QtWidgets.QAction(MainWindow)
        self.action_save_canvas.setObjectName("action_save_canvas")
        self.action_save_raster = QtWidgets.QAction(MainWindow)
        self.action_save_raster.setObjectName("action_save_raster")
        self.menu_auto_refresh.addAction(self.action_refresh_30s)
        self.menu_auto_refresh.addAction(self.action_refresh_5m)
        self.menu_auto_refresh.addAction(self.action_refresh_30m)
        self.menu_auto_refresh.addAction(self.action_refresh_stop)
        self.menu_save_session.addAction(self.action_current_file)
        self.menu_save_session.addAction(self.action_all_files)
        self.menu_save_session.addAction(self.action_filters)
        self.menu_merge_data.addAction(self.action_merge_raw)
        self.menu_merge_data.addAction(self.action_merge_processed)
        self.menu_save_image.addAction(self.action_save_canvas)
        self.menu_bar.addAction(self.menu_save_image.menuAction())
        self.menu_bar.addAction(self.menu_save_session.menuAction())
        self.menu_bar.addAction(self.menu_auto_refresh.menuAction())
        self.menu_bar.addAction(self.menu_merge_data.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "InSpectra Gadget"))
        self.files_box.setTitle(_translate("MainWindow", "Opened Files"))
        self.up_file_button.setText(_translate("MainWindow", "Up"))
        self.down_file_button.setText(_translate("MainWindow", "Down"))
        self.open_files_button.setText(_translate("MainWindow", "Open"))
        self.delete_files_button.setText(_translate("MainWindow", "Remove"))
        self.clear_files_button.setText(_translate("MainWindow", "Clear"))
        self.refresh_file_button.setText(_translate("MainWindow", "Refresh"))
        self.settings_box.setTitle(_translate("MainWindow", "Settings Selected File"))
        self.copy_settings_button.setText(_translate("MainWindow", "Copy"))
        self.paste_settings_button.setText(_translate("MainWindow", "Paste"))
        self.reset_settings_button.setText(_translate("MainWindow", "Reset"))
        self.filters_box.setTitle(_translate("MainWindow", "Filters Selected File"))
        self.delete_filters_button.setText(_translate("MainWindow", "Remove"))
        self.clear_filters_button.setText(_translate("MainWindow", "Clear"))
        self.up_filters_button.setText(_translate("MainWindow", "Up"))
        self.down_filters_button.setText(_translate("MainWindow", "Down"))
        self.copy_filters_button.setText(_translate("MainWindow", "Copy"))
        self.paste_filters_button.setText(_translate("MainWindow", "Paste"))
        self.load_filters_button.setText(_translate("MainWindow", "Load"))
        self.reset_limits_button.setText(_translate("MainWindow", "Reset"))
        self.min_label.setText(_translate("MainWindow", "Min"))
        self.max_label.setText(_translate("MainWindow", "Max"))
        self.reverse_colors_box.setText(_translate("MainWindow", "Reverse"))
        self.lock_checkbox.setText(_translate("MainWindow", "Lock"))
        self.mid_checkbox.setText(_translate("MainWindow", "Mid"))
        self.paste_view_button.setText(_translate("MainWindow", "Paste"))
        self.copy_view_button.setText(_translate("MainWindow", "Copy"))
        self.menu_auto_refresh.setTitle(_translate("MainWindow", "Auto Refresh"))
        self.menu_save_session.setTitle(_translate("MainWindow", "Save Session"))
        self.menu_merge_data.setTitle(_translate("MainWindow", "Merge Data"))
        self.menu_save_image.setTitle(_translate("MainWindow", "Save Image"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionNewWindow.setText(_translate("MainWindow", "New Window..."))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File..."))
        self.actionClear_Window.setText(_translate("MainWindow", "Clear Window..."))
        self.action_png.setText(_translate("MainWindow", "PNG"))
        self.action_pdf.setText(_translate("MainWindow", "PDF"))
        self.actionSession.setText(_translate("MainWindow", "All Files..."))
        self.actionExit.setText(_translate("MainWindow", "Quit..."))
        self.actionQuit.setText(_translate("MainWindow", "Quit..."))
        self.action_eps.setText(_translate("MainWindow", "EPS"))
        self.action_ps.setText(_translate("MainWindow", "PS"))
        self.action_svg.setText(_translate("MainWindow", "SVG"))
        self.actionLoadSession.setText(_translate("MainWindow", "Load Session..."))
        self.action_save_image.setText(_translate("MainWindow", "Save Image..."))
        self.action_canvas_to_script.setText(_translate("MainWindow", "Generate Script..."))
        self.actionEnable_Auto_Refresh.setText(_translate("MainWindow", "Enable Auto Refresh..."))
        self.actionEvery_10_s.setText(_translate("MainWindow", "Every 2 s..."))
        self.actionEvery_5_s.setText(_translate("MainWindow", "Every 5 s..."))
        self.action_refresh_5m.setText(_translate("MainWindow", "Every 5 minutes..."))
        self.actionEvery_5_minutes.setText(_translate("MainWindow", "Every 5 minutes..."))
        self.action_refresh_stop.setText(_translate("MainWindow", "Stop..."))
        self.action_refresh_30s.setText(_translate("MainWindow", "Every 30 seconds..."))
        self.action_refresh_30m.setText(_translate("MainWindow", "Every 30 minutes..."))
        self.action_current_file.setText(_translate("MainWindow", "Current File..."))
        self.action_all_files.setText(_translate("MainWindow", "All Files..."))
        self.action_filters.setText(_translate("MainWindow", "Filters..."))
        self.action_merge_raw.setText(_translate("MainWindow", "Raw Data..."))
        self.action_merge_processed.setText(_translate("MainWindow", "Processed Data..."))
        self.action_save_canvas.setText(_translate("MainWindow", "Save Canvas..."))
        self.action_save_raster.setText(_translate("MainWindow", "Raster..."))

