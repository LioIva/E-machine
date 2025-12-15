import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QCheckBox, QGroupBox, QMessageBox, QScrollArea,
    QFrame, QDialog, QSizePolicy, QRadioButton, QListWidget, QAbstractItemView, QMenu, QInputDialog,
    QListWidgetItem, QColorDialog, QToolButton, QDialogButtonBox, QComboBox, QCompleter, QTextEdit,
    QStackedWidget, QSpacerItem, QTableView, QHeaderView, QProgressBar,
    QButtonGroup,
    QFormLayout
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, QRegularExpression, QAbstractTableModel,
    QThread, pyqtSignal, QObject
)
from PyQt6.QtGui import QFont, QIntValidator, QDoubleValidator, QColor, QAction, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from PIL import Image, ImageDraw


# --- GLOBAL EXCEPTION HANDLER ---
def custom_exception_hook(exc_type, exc_value, exc_traceback):
    """Global handler for uncaught exceptions."""
    error_message = f"Critical Uncaught Error:\nType: {exc_type.__name__}\nValue: {exc_value}"
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    error_box = QMessageBox(QMessageBox.Icon.Critical,
                            "Application Error",
                            error_message)
    error_box.exec()


sys.excepthook = custom_exception_hook

# --- Global NVH Constants and Utilities ---
P_REF = 2e-5 # 2 * 10^-5 Pa
MPA_TO_PA = 1e6
NUMERICAL_FLOOR_PA = 1e-12
SQRT_2 = np.sqrt(2)
CONSOLIDATED_DATA = None


def calculate_a_weighting(frequencies):
    """Calculates the A-weighting gain (in dB)."""
    f = np.maximum(frequencies, 1e-3)
    f_sq = f ** 2
    f1_sq = 20.6 ** 2
    f2_sq = 107.7 ** 2
    f3_sq = 737.9 ** 2
    f4_sq = 12194 ** 2

    num_sq = f4_sq * f_sq ** 2
    den_sq = (f_sq + f1_sq) * np.sqrt(f_sq + f2_sq) * np.sqrt(f_sq + f3_sq) * (f_sq + f4_sq)

    R_A_ratio = np.divide(num_sq, den_sq, out=np.zeros_like(num_sq, dtype=float), where=den_sq != 0)
    R_A = 20 * np.log10(R_A_ratio, out=np.full_like(R_A_ratio, np.nan), where=R_A_ratio > 0)
    return R_A + 2.000


# Standalone utility function for calculation (used in Tab 4)
def calculate_derived_data(peak_mpa_series, freq_series, y_axis_metric, is_rms_selected):
    """Performs dynamic calculation chain: MPa -> (RMS) -> dB -> dBA."""
    if is_rms_selected:
        mpa_calc = peak_mpa_series / SQRT_2
    else:
        mpa_calc = peak_mpa_series

    if y_axis_metric == 'Magnitude (MPa)':
        return mpa_calc

    magnitude_pa_calc = mpa_calc * MPA_TO_PA
    magnitude_pa_safe = np.maximum(magnitude_pa_calc, NUMERICAL_FLOOR_PA)
    final_spl_db = 20 * np.log10(magnitude_pa_safe / P_REF)

    if y_axis_metric == 'dB':
        return final_spl_db

    a_weighting_gain = calculate_a_weighting(freq_series.values)
    final_spl_dba = final_spl_db + a_weighting_gain

    return final_spl_dba # Returns dBA


# --- A-Weighting Function (for Tab 3 TF Summation) ---
def a_weighting_correction_new(freq):
    """Calculates the A-weighting correction factor (in dB) for a given frequency (Hz)."""
    f = np.abs(freq)
    safe_f = np.where(f == 0, 1e-6, f)
    f_c2 = safe_f ** 2
    f_c4 = safe_f ** 4

    num_A = (12194 ** 2) * f_c4
    den_A = ((f_c2 + 20.6 ** 2) * np.sqrt((f_c2 + 107.7 ** 2) * (f_c2 + 737.9 ** 2)) * (f_c2 + 12194 ** 2))

    R_A_transfer = np.divide(num_A, den_A, out=np.zeros_like(num_A, dtype=float), where=den_A != 0)

    A_dB = 20 * np.log10(R_A_transfer, out=np.full_like(R_A_transfer, np.nan), where=R_A_transfer > 0)

    return A_dB + 2.0


# ----------------------------------------------------------------------
# --- TAB 4: PLOTTING UTILITIES & WIDGET (PostProcessingWidget) ---
# ----------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    """Matplotlib canvas widget embedded in PyQt6."""
    plot_settings = {}
    sensor_axes_map = {}
    custom_titles = {}
    custom_x_labels = {}
    custom_y_labels = {}
    line_styles = ['-', '--', ':', '-.']

    def __init__(self, parent=None, width=5, height=4, dpi=100, is_dark_mode=True):
        plt.style.use('default')
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.clf()
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'Arial', 'sans-serif']})
        self.is_dark_mode = is_dark_mode
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.plotting_widget = parent

    def set_theme(self, is_dark_mode):
        """Updates the theme state and forces a plot redraw."""
        self.is_dark_mode = is_dark_mode
        if self.plotting_widget:
            self.plotting_widget.draw_plots()
        else:
            self.fig.clf()
            self.draw()

    def show_context_menu(self, point):
        """Displays the context menu when the canvas is right-clicked."""
        if self.is_dark_mode:
            MENU_BG = '#2D2D30';
            MENU_BORDER = '#555555';
            MENU_TEXT = 'white';
            MENU_ACCENT_BG = '#004D8A'
        else:
            MENU_BG = 'white';
            MENU_BORDER = '#DDDDDD';
            MENU_TEXT = 'black';
            MENU_ACCENT_BG = '#E8E8E8'

        try:
            x_display, y_display = point.x(), point.y()
            context_menu = QMenu(self)
            context_menu.setStyleSheet(f"""
                QMenu {{ border: 1px solid {MENU_BORDER}; padding: 5px; background-color: {MENU_BG}; color: {MENU_TEXT}; border-radius: 4px; }}
                QMenu::item {{ padding: 4px 15px 4px 10px; background-color: transparent; color: {MENU_TEXT}; border-radius: 2px; }}
                QMenu::item:selected {{ background-color: {MENU_ACCENT_BG}; color: {'white' if self.is_dark_mode else 'black'}; }}
                QMenu::separator {{ height: 1px; background: {MENU_BORDER}; margin: 5px 0px; }}
            """)

            ax_clicked = None
            sensor_base_clicked = None
            for ax_key, ax_obj in self.sensor_axes_map.items():
                bbox = ax_obj.get_window_extent()
                if bbox.contains(x_display, y_display).all():
                    ax_clicked = ax_obj
                    sensor_base_clicked = ax_key
                    break

            if not ax_clicked:
                if self.plotting_widget.shared_sensors:
                    context_menu.addAction(self.plotting_widget.create_action("Toggle Legend Visibility", lambda
                        checked: self.toggle_all_legends()))
                    context_menu.addAction(self.plotting_widget.create_action("Adjust Legend Position...",
                                                                              lambda _: self.change_legend_position()))
                    context_menu.addSeparator()
                    context_menu.addAction(self.plotting_widget.create_action("Edit Global X-Axis Label...",
                                                                              lambda _: self.change_global_x_label()))
                    context_menu.addAction(self.plotting_widget.create_action("Edit Global Y-Axis Label...",
                                                                              lambda _: self.change_global_y_label()))
                    context_menu.addAction(self.plotting_widget.create_action("Edit All Subplot Titles...", lambda
                        _: self.change_all_subplot_titles()))
                else:
                    context_menu.addAction(QAction("No plots loaded or selected.", self, enabled=False))
            else:
                curves_menu = QMenu(f"Curve Options for {sensor_base_clicked}", self)
                curves_menu.setStyleSheet(context_menu.styleSheet())
                context_menu.addMenu(curves_menu)

                lines = [l for l in ax_clicked.lines if l.get_label() and l.get_label() != '_nolegend_']
                if lines:
                    for line in lines:
                        curve_name = line.get_label()
                        curve_key = f"{sensor_base_clicked}_{curve_name}"
                        settings = self.plot_settings.get(curve_key, {})

                        sub_menu = QMenu(curve_name, self);
                        sub_menu.setStyleSheet(context_menu.styleSheet())

                        action_hide = QAction("Toggle Visibility", self);
                        action_hide.setCheckable(True)
                        action_hide.setChecked(settings.get('visible', True))
                        action_hide.triggered.connect(
                            lambda checked, key=curve_key: self.toggle_curve_visibility(key, checked))
                        sub_menu.addAction(action_hide)

                        action_color = self.plotting_widget.create_action("Change Color...",
                                                                          lambda _: self.change_curve_color(curve_key))
                        sub_menu.addAction(action_color)
                        action_width = self.plotting_widget.create_action("Change Line Width...",
                                                                          lambda _: self.change_line_width(curve_key))
                        sub_menu.addAction(action_width)

                        style_menu = QMenu("Change Line Style", self);
                        style_menu.setStyleSheet(context_menu.styleSheet())
                        for style in ['Solid (-)', 'Dashed (--)', 'Dotted (:)', 'Dash-dot (-.)']:
                            s_char = style.split(' ')[1].strip('()')
                            action = self.plotting_widget.create_action(style, lambda _, key=curve_key,
                                                                        s=s_char: self.change_line_style(
                                    key, s))
                            style_menu.addAction(action)
                        sub_menu.addMenu(style_menu)
                        curves_menu.addMenu(sub_menu)

                    context_menu.addSeparator()

                context_menu.addAction(self.plotting_widget.create_action(f"Edit Title ({sensor_base_clicked})...",
                                                                          lambda _: self.change_single_subplot_title(
                                                                              sensor_base_clicked)))
                context_menu.addAction(self.plotting_widget.create_action(f"Edit X-Axis Label...",
                                                                          lambda _: self.change_single_axis_label(
                                                                              sensor_base_clicked, 'x')))
                context_menu.addAction(self.plotting_widget.create_action(f"Edit Y-Axis Label...",
                                                                          lambda _: self.change_single_axis_label(
                                                                              sensor_base_clicked, 'y')))
                context_menu.addSeparator()
                context_menu.addAction(self.plotting_widget.create_action("Toggle Legend Visibility",
                                                                          lambda checked: self.toggle_all_legends()))
                context_menu.addAction(self.plotting_widget.create_action("Adjust Legend Position...",
                                                                          lambda _: self.change_legend_position()))

            if context_menu.actions():
                context_menu.exec(self.mapToGlobal(point))

        except Exception as e:
            QMessageBox.critical(self.plotting_widget, "Context Menu Error",
                                 f"An unexpected error occurred while processing the right-click menu. Details: {e}")

    def change_curve_color(self, curve_key):
        settings = self.plot_settings.get(curve_key, {})
        initial_color = settings.get('color', 'blue')
        color = QColorDialog.getColor(QColor(initial_color), self)
        if color.isValid():
            if curve_key not in self.plot_settings: self.plot_settings[curve_key] = {}
            self.plot_settings[curve_key]['color'] = color.name()
            self.plotting_widget.draw_plots()

    def toggle_curve_visibility(self, curve_key, checked):
        if curve_key not in self.plot_settings: self.plot_settings[curve_key] = {}
        self.plot_settings[curve_key]['visible'] = checked
        self.plotting_widget.draw_plots()

    def change_line_width(self, curve_key):
        settings = self.plot_settings.get(curve_key, {})
        current_width = settings.get('linewidth', 1.8)
        new_width, ok = QInputDialog.getDouble(self, "Line Width", "Enter new line width:", current_width, 0.1, 10.0, 1)
        if ok:
            if curve_key not in self.plot_settings: self.plot_settings[curve_key] = {}
            self.plot_settings[curve_key]['linewidth'] = new_width
            self.plotting_widget.draw_plots()

    def change_line_style(self, curve_key, style_char):
        if curve_key not in self.plot_settings: self.plot_settings[curve_key] = {}
        self.plot_settings[curve_key]['linestyle'] = style_char
        self.plotting_widget.draw_plots()

    def toggle_all_legends(self):
        is_visible = self.plot_settings.get('global_legend_visible', True)
        new_state = not is_visible
        self.plot_settings['global_legend_visible'] = new_state
        self.plotting_widget.draw_plots()

    def change_legend_position(self):
        positions = ["best", "upper right", "upper left", "lower left", "lower right", "right", "center left",
                     "center right", "lower center", "upper center", "center"]
        position, ok = QInputDialog.getItem(self, "Legend Position", "Select legend position:", positions, 0, False)
        if ok and position:
            self.plot_settings['global_legend_loc'] = position
            self.plotting_widget.draw_plots()

    def change_single_axis_label(self, sensor_base, axis):
        if axis == 'x':
            current_label = self.custom_x_labels.get(sensor_base, "Frequency (Hz)")
            new_label, ok = QInputDialog.getText(self, f"Edit X-Axis Label for {sensor_base}",
                                                 "Enter new X-Axis Label:", QLineEdit.EchoMode.Normal, current_label)
            if ok: self.custom_x_labels[sensor_base] = new_label
        elif axis == 'y':
            current_unit = self.plotting_widget.y_axis_metric.replace('Magnitude (', '').replace(')', '')
            current_label = self.custom_y_labels.get(sensor_base, current_unit)
            new_label, ok = QInputDialog.getText(self, f"Edit Y-Axis Label for {sensor_base}",
                                                 "Enter new Y-Axis Label:", QLineEdit.EchoMode.Normal, current_label)
            if ok: self.custom_y_labels[sensor_base] = new_label
        self.plotting_widget.draw_plots()

    def change_global_x_label(self):
        current_label = self.custom_x_labels.get(next(iter(self.sensor_axes_map.keys()), "Frequency (Hz)"),
                                                 "Frequency (Hz)")
        new_label, ok = QInputDialog.getText(self, "Edit Global X-Axis Label", "Enter new Global X-Axis Label:",
                                             QLineEdit.EchoMode.Normal, current_label)
        if ok:
            for ax_key in self.sensor_axes_map.keys(): self.custom_x_labels[ax_key] = new_label
            self.plotting_widget.draw_plots()

    def change_global_y_label(self):
        current_unit = self.plotting_widget.y_axis_metric.replace('Magnitude (', '').replace(')', '')
        current_label = self.custom_y_labels.get(next(iter(self.sensor_axes_map.keys()), current_unit), current_unit)
        new_label, ok = QInputDialog.getText(self, "Edit Global Y-Axis Label", "Enter new Global Y-Axis Label:",
                                             QLineEdit.EchoMode.Normal, current_unit)
        if ok:
            for ax_key in self.sensor_axes_map.keys(): self.custom_y_labels[ax_key] = new_label
            self.plotting_widget.draw_plots()

    def change_single_subplot_title(self, sensor_base):
        current_title = self.custom_titles.get(sensor_base, sensor_base)
        new_title, ok = QInputDialog.getText(self, f"Edit Subplot Title for {sensor_base}", "Enter new Title:",
                                             QLineEdit.EchoMode.Normal, current_title)
        if ok:
            self.custom_titles[sensor_base] = new_title
            self.plotting_widget.draw_plots()

    def change_all_subplot_titles(self):
        default_title = next(iter(self.custom_titles.values()), next(iter(self.sensor_axes_map.keys()), "Response"))
        new_title, ok = QInputDialog.getText(self, "Edit All Subplot Titles",
                                             "Enter new base Title (e.g., 'Front Mic'):", QLineEdit.EchoMode.Normal,
                                             default_title)
        if ok:
            for ax_key in self.sensor_axes_map.keys(): self.custom_titles[ax_key] = new_title
            self.plotting_widget.draw_plots()

    def update_plot(self, df_list, legend_names, unique_sensor_bases, y_axis_metric, y_log_scale, limits):
        if self.is_dark_mode:
            FIG_BG = '#1E1E1E';
            AXIS_BG = '#2D2D30';
            TEXT_COLOR = 'white';
            GRID_COLOR = '#666666';
            WARNING_COLOR = '#FF9800'
        else:
            FIG_BG = '#F0F0F0';
            AXIS_BG = 'white';
            TEXT_COLOR = 'black';
            GRID_COLOR = '#CCCCCC';
            WARNING_COLOR = '#CC0000'

        TITLE_FONTSIZE = 10;
        AXIS_LABEL_FONTSIZE = 8;
        TICK_LABEL_FONTSIZE = 8;
        LEGEND_FONTSIZE = 8

        df_plot_list = [df.copy() for df in df_list];
        contains_non_positive = False;
        data_col_suffix = '_MPa'

        if y_axis_metric in ['dB', 'dBA']:
            for df_plot in df_plot_list:
                for sensor_base in unique_sensor_bases:
                    col_name = f"{sensor_base}{data_col_suffix}"
                    if col_name in df_plot.columns and (df_plot[col_name] <= 0).any():
                        contains_non_positive = True;
                        break
                if contains_non_positive: break

        y_unit = y_axis_metric.replace('Magnitude (', '').replace(')', '')

        N = len(unique_sensor_bases)
        if N == 0:
            self.fig.clf();
            self.sensor_axes_map = {};
            self.draw();
            return

        nrows, ncols = 1, N
        if N > 2: nrows, ncols = 2, 2
        if N > 4: nrows, ncols = 2, 3
        if N > 6: N, nrows, ncols = 6, 2, 3

        self.fig.clf();
        self.sensor_axes_map = {}
        self.fig.patch.set_facecolor(FIG_BG)

        for i, sensor_base in enumerate(unique_sensor_bases[:N]):
            ax_i = self.fig.add_subplot(nrows, ncols, i + 1)
            self.sensor_axes_map[sensor_base] = ax_i
            ax_i.set_facecolor(AXIS_BG)
            data_col_name = f"{sensor_base}{data_col_suffix}"

            title_base = self.custom_titles.get(sensor_base, sensor_base)
            x_label = self.custom_x_labels.get(sensor_base, "Frequency (Hz)")
            y_label = self.custom_y_labels.get(sensor_base, y_unit)

            all_lines = []
            for df_index, df_plot in enumerate(df_plot_list):
                if df_plot is not None and data_col_name in df_plot.columns and 'Frequency (Hz)' in df_plot.columns:
                    curve_name = legend_names[df_index]
                    curve_key = f"{sensor_base}_{curve_name}"
                    settings = self.plot_settings.get(curve_key, {})

                    color = settings.get('color', f'C{df_index}')
                    linewidth = settings.get('linewidth', 1.8)
                    linestyle = settings.get('linestyle', self.line_styles[df_index % len(self.line_styles)])
                    visible = settings.get('visible', True)

                    line = ax_i.plot(df_plot['Frequency (Hz)'], df_plot[data_col_name],
                                     label=curve_name, linewidth=linewidth, color=color,
                                     linestyle=linestyle, visible=visible)
                    all_lines.extend(line)

            log_applied = y_log_scale and not contains_non_positive
            ax_i.set_yscale('log' if log_applied else 'linear')

            title_color = TEXT_COLOR
            if y_log_scale and contains_non_positive: title_color = WARNING_COLOR

            title_text = f"{title_base} ({y_unit})"
            ax_i.set_title(title_text, fontsize=TITLE_FONTSIZE, fontweight='semibold', color=title_color)

            try:
                if limits['Xmin'] is not None and limits['Xmax'] is not None and limits['Xmin'] < limits['Xmax']:
                    ax_i.set_xlim(limits['Xmin'], limits['Xmax'])
                if limits['Ymin'] is not None and limits['Ymax'] is not None and limits['Ymin'] < limits['Ymax']:
                    if log_applied and limits['Ymin'] <= 0:
                        pass
                    else:
                        ax_i.set_ylim(limits['Ymin'], limits['Ymax'])
                elif log_applied:
                    ax_i.set_ylim(bottom=1e-12)
            except Exception:
                if log_applied: ax_i.set_ylim(bottom=1e-12)
                pass

            ax_i.set_xlabel(x_label, fontsize=AXIS_LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_i.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE, color=TEXT_COLOR)
            ax_i.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE, colors=TEXT_COLOR)
            ax_i.grid(True, linestyle=':', alpha=0.5, color=GRID_COLOR)
            ax_i.spines['right'].set_visible(False);
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['left'].set_color(GRID_COLOR);
            ax_i.spines['bottom'].set_color(GRID_COLOR)
            ax_i.spines['left'].set_linewidth(0.8);
            ax_i.spines['bottom'].set_linewidth(0.8)

            if all_lines:
                legend_loc = self.plot_settings.get('global_legend_loc', 'best')
                legend_visible = self.plot_settings.get('global_legend_visible', True)
                legend = ax_i.legend(loc=legend_loc, fontsize=LEGEND_FONTSIZE, frameon=True, facecolor=AXIS_BG,
                                     edgecolor=GRID_COLOR)
                if legend:
                    for text in legend.get_texts(): text.set_color(TEXT_COLOR)
                    legend.set_visible(legend_visible)

        self.fig.tight_layout(pad=1.5)
        self.draw()


class LoadedFileManager(QWidget):
    """Manages the list of loaded files for comparison in the PostProcessingWidget."""

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget
        self.files = [] # List of tuples: [(df, custom_name, original_path), ...]
        self.layout = QVBoxLayout(self);
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemSelectionChanged.connect(self._selection_changed)
        self.list_widget.setObjectName("PlotListWidget");
        self.layout.addWidget(self.list_widget)

        self.btn_browse = QPushButton("➕ Load Raw Files")
        self.btn_browse.setObjectName("SecondaryButton_Add");
        self.btn_browse.clicked.connect(self._prompt_load_file)
        self.layout.addWidget(self.btn_browse)

    def _selection_changed(self):
        self.parent_widget.update_shared_sensors()
        self.parent_widget.draw_plots()

    def _show_context_menu(self, point):
        context_menu = QMenu(self);
        item = self.list_widget.itemAt(point)
        if item:
            rename_action = context_menu.addAction("✏️ Rename Curve...")
            remove_action = context_menu.addAction("🗑️ Remove File")
            action = context_menu.exec(self.list_widget.mapToGlobal(point))
            if action == rename_action:
                self._rename_item(item)
            elif action == remove_action:
                self._remove_item(item)

    def _rename_item(self, item):
        file_index = self.list_widget.row(item)
        if 0 <= file_index < len(self.files):
            new_name, ok = QInputDialog.getText(self, "Rename Curve",
                                                f"Enter new name for '{self.files[file_index][1]}':",
                                                QLineEdit.EchoMode.Normal, self.files[file_index][1])
            if ok and new_name:
                self.files[file_index] = (self.files[file_index][0], new_name, self.files[file_index][2])
                item.setText(new_name);
                self.parent_widget.draw_plots()

    def _remove_item(self, item):
        file_index = self.list_widget.row(item)
        if 0 <= file_index < len(self.files):
            del self.files[file_index];
            self.list_widget.takeItem(file_index)
            self.parent_widget.update_shared_sensors();
            self.parent_widget.draw_plots()

    def _add_file_to_list(self, df, filename):
        default_name = os.path.basename(filename).replace('CONSOLIDATED_RAW_MPA', 'Run')
        default_name = os.path.splitext(default_name)[0]

        if any(item[2] == filename for item in self.files):
            QMessageBox.warning(self, "Duplicate File", f"File '{os.path.basename(filename)}' is already loaded.")
            return False

        self.files.append((df, default_name, filename))
        item = QListWidgetItem(default_name);
        item.setSelected(True)
        self.list_widget.addItem(item);
        self.parent_widget.update_shared_sensors();
        self.parent_widget.draw_plots()
        return True

    def load_summary_file(self, filename):
        if not filename: return
        try:
            df_temp = pd.read_excel(filename, header=None)
            header_row_index = -1
            for i, row in df_temp.iterrows():
                if any('frequency (hz)' in str(cell).lower() for cell in row):
                    header_row_index = i;
                    break

            df = pd.read_excel(filename, header=header_row_index) if header_row_index != -1 else pd.read_excel(filename)
            df.dropna(axis=0, how='all', inplace=True);
            df.dropna(axis=1, how='all', inplace=True)
            freq_col_match = [col for col in df.columns if 'frequency' in str(col).lower()]

            if not freq_col_match or df.shape[1] < 2:
                QMessageBox.critical(self, "File Load Error",
                                     "File must contain a 'Frequency' column and at least one sensor data column.")
                return

            df.rename(columns={freq_col_match[0]: 'Frequency (Hz)'}, inplace=True)

            if not any(col.endswith('_MPa') for col in df.columns):
                QMessageBox.warning(self, "File Format Warning",
                                    "The loaded file does not appear to be a Raw MPa report. Plotting might be inconsistent.")

            self._add_file_to_list(df, filename)

        except Exception as e:
            QMessageBox.critical(self, "File Load Error", f"Failed to load file '{os.path.basename(filename)}': {e}")

    def _prompt_load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Consolidated Raw MPa Summary File",
                                                  os.path.expanduser("~"), "Excel files (*.xlsx *.xls)")
        if not filename: return
        self.load_summary_file(filename)

    def get_selected_files_data(self):
        selected_data = []
        for index in self.list_widget.selectedIndexes():
            file_index = index.row()
            if 0 <= file_index < len(self.files):
                df = self.files[file_index][0];
                name = self.files[file_index][1]
                selected_data.append((df, name))
        return selected_data

    def get_all_sensor_bases(self):
        all_sensor_bases = set()
        for df, _, _ in self.files:
            all_sensor_bases.update({col.rsplit('_', 1)[0].strip() for col in df.columns if col != 'Frequency (Hz)'})
        return all_sensor_bases


class PostProcessingWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)

        # Initialize data attributes first
        self.y_axis_metric = 'Magnitude (MPa)'
        self.y_log_scale = True
        self.is_rms_selected = False
        self.shared_sensors = []
        self.axis_edits = {}

        # Initialize UI components in order
        self.file_manager = LoadedFileManager(self)
        self._setup_plotting_area()
        self._setup_right_sidebar()  # This now initializes checklist_layout

        self.layout.addWidget(self.plot_area, 1)
        self.layout.addWidget(self.sidebar_right)

        self.set_metric_button_state('Magnitude (MPa)', initial=True)
        self.update_ui_elements(enable_controls=False)  # Now safe to call

    def _setup_plotting_area(self):
        self.plot_area = QWidget()
        layout_left = QVBoxLayout(self.plot_area)
        layout_left.setSpacing(15)
        layout_left.setContentsMargins(0, 0, 0, 0)

        plot_frame = QFrame()
        plot_frame.setObjectName("PlotFrame")
        plot_frame_layout = QVBoxLayout(plot_frame)
        plot_frame_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas = PlotCanvas(plot_frame, is_dark_mode=self.main_window.is_dark_mode)
        self.canvas.plotting_widget = self
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        plot_frame_layout.addWidget(self.canvas)
        layout_left.addWidget(plot_frame, 1)

    def _setup_right_sidebar(self):
        self.sidebar_right = QWidget()
        self.sidebar_right.setFixedWidth(220)  # Narrow Gmail style
        self.sidebar_right.setObjectName("PlotOptionsSidebar")

        sidebar_vbox = QVBoxLayout(self.sidebar_right)
        sidebar_vbox.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("ControlScrollArea")

        scroll_content = QWidget()
        self.layout_right = QVBoxLayout(scroll_content)
        self.layout_right.setSpacing(10)
        self.layout_right.setContentsMargins(8, 8, 8, 8)
        self.layout_right.setAlignment(Qt.AlignmentFlag.AlignTop)

        sidebar_title = QLabel("Plot Options")
        sidebar_title.setStyleSheet("font-weight: 600; color: #1F1F1F; margin-bottom: 5px;")
        self.layout_right.addWidget(sidebar_title)

        # 1. Metric Selection
        group_metric = QGroupBox("Metric")
        group_metric.setObjectName("CompactGroupBox")
        layout_metric = QVBoxLayout(group_metric)
        layout_metric.setContentsMargins(6, 12, 6, 6)

        self.btn_mpa = QPushButton("MPa")
        self.btn_db = QPushButton("dB")
        self.btn_dba = QPushButton("dBA")
        self.metric_group = QButtonGroup(self)
        self.metric_group.setExclusive(True)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        for i, (btn, lbl) in enumerate([(self.btn_mpa, "MPa"), (self.btn_db, "dB"), (self.btn_dba, "dBA")]):
            btn.setCheckable(True)
            btn.setFixedWidth(58)
            btn.setObjectName(f"MetricButton_{lbl}")
            self.metric_group.addButton(btn, i)
            btn_layout.addWidget(btn)

        self.btn_mpa.clicked.connect(lambda: self.update_metric('Magnitude (MPa)'))
        self.btn_db.clicked.connect(lambda: self.update_metric('dB'))
        self.btn_dba.clicked.connect(lambda: self.update_metric('dBA'))
        layout_metric.addLayout(btn_layout)

        self.cb_rms = QCheckBox("RMS Scale")
        self.cb_rms.stateChanged.connect(self.toggle_rms)
        self.cb_log_y = QCheckBox("Log Y")
        self.cb_log_y.setChecked(True)
        self.cb_log_y.stateChanged.connect(self.toggle_log_y)
        layout_metric.addWidget(self.cb_rms)
        layout_metric.addWidget(self.cb_log_y)
        self.layout_right.addWidget(group_metric)

        # 2. Axis Limits
        group_limits = QGroupBox("Limits")
        group_limits.setObjectName("CompactGroupBox")
        layout_limits = QGridLayout(group_limits)
        layout_limits.setContentsMargins(6, 12, 6, 6)
        layout_limits.setSpacing(4)

        validator = QDoubleValidator()
        limit_keys = [("Xmin", "X Min"), ("Xmax", "X Max"), ("Ymin", "Y Min"), ("Ymax", "Y Max")]
        for row, (key, lbl) in enumerate(limit_keys):
            edit = QLineEdit()
            edit.setValidator(validator)
            edit.setPlaceholderText("Auto")
            edit.setFixedWidth(65)
            edit.textChanged.connect(self.draw_plots)
            self.axis_edits[key] = edit
            layout_limits.addWidget(QLabel(key[0]), row // 2, (row % 2) * 2)
            layout_limits.addWidget(edit, row // 2, (row % 2) * 2 + 1)
        self.layout_right.addWidget(group_limits)

        # 3. Sensor Checklist (Crucial Fix: Initialize checklist_layout here)
        group_checklist = QGroupBox("Sensors")
        group_checklist.setObjectName("CompactGroupBox")
        self.checklist_layout = QVBoxLayout()  # Initialized here
        self.checklist_layout.setContentsMargins(6, 12, 6, 6)
        group_checklist.setLayout(self.checklist_layout)
        self.layout_right.addWidget(group_checklist)

        # 4. File Comparison
        group_file = QGroupBox("Runs")
        group_file.setObjectName("CompactGroupBox")
        layout_file = QVBoxLayout(group_file)
        layout_file.addWidget(self.file_manager)
        self.layout_right.addWidget(group_file)

        self.scroll_area.setWidget(scroll_content)
        sidebar_vbox.addWidget(self.scroll_area)

    def create_action(self, text, callback, shortcut=None):
        action = QAction(text, self)

        def safe_callback(*args):
            try:
                callback(*args)
            except Exception as e:
                QMessageBox.critical(self, "Plot Interaction Error",
                                     f"An internal error occurred during plot modification. Details: {e}")

        action.triggered.connect(safe_callback)
        if shortcut: action.setShortcut(shortcut)
        return action

    def set_metric_button_state(self, metric, initial=False):
        self.y_axis_metric = metric
        for btn, name in [(self.btn_mpa, 'Magnitude (MPa)'), (self.btn_db, 'dB'), (self.btn_dba, 'dBA')]:
            is_selected = (name == metric)
            btn.setProperty("selected", "true" if is_selected else "false")
            btn.style().polish(btn)

        if not initial:
            self.draw_plots()

    def update_ui_elements(self, enable_controls=True):
        has_data = len(self.file_manager.get_selected_files_data()) > 0

        self.btn_mpa.setEnabled(has_data);
        self.btn_db.setEnabled(has_data);
        self.btn_dba.setEnabled(has_data)
        self.cb_log_y.setEnabled(has_data);
        self.cb_rms.setEnabled(has_data)
        for edit in self.axis_edits.values(): edit.setEnabled(has_data)
        for i in range(self.checklist_layout.count()):
            widget = self.checklist_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox): widget.setEnabled(has_data)

    def update_shared_sensors(self):
        selected_files = self.file_manager.get_selected_files_data()
        if not selected_files:
            shared_sensor_bases = set();
            self.update_ui_elements(enable_controls=False)
        else:
            first_df = selected_files[0][0]
            shared_sensor_bases = {col.rsplit('_', 1)[0].strip() for col in first_df.columns if col != 'Frequency (Hz)'}
            for df, _ in selected_files[1:]:
                current_sensors = {col.rsplit('_', 1)[0].strip() for col in df.columns if col != 'Frequency (Hz)'}
                shared_sensor_bases = shared_sensor_bases.intersection(current_sensors)
            self.update_ui_elements(enable_controls=True)
        self.shared_sensors = sorted(list(shared_sensor_bases));
        self.populate_checklist()

    def populate_checklist(self):
        for i in reversed(range(self.checklist_layout.count())):
            widget = self.checklist_layout.itemAt(i).widget()
            if widget is not None: widget.deleteLater()
        if not self.shared_sensors: return

        for sensor_name in self.shared_sensors:
            full_col_name = f"{sensor_name}_MPa"
            cb = QCheckBox(sensor_name);
            cb.setProperty('full_name', full_col_name);
            cb.setChecked(True)
            cb.stateChanged.connect(self.draw_plots);
            self.checklist_layout.addWidget(cb)
        self.checklist_layout.addStretch(1)

    def update_metric(self, new_metric):
        self.set_metric_button_state(new_metric)

    def toggle_rms(self, state):
        self.is_rms_selected = state == Qt.CheckState.Checked.value;
        self.draw_plots()

    def toggle_log_y(self, state):
        self.y_log_scale = state == Qt.CheckState.Checked.value;
        self.draw_plots()

    def _get_axis_limits(self):
        limits = {};
        for key, edit in self.axis_edits.items():
            try:
                limits[key] = float(edit.text())
            except ValueError:
                limits[key] = None
        return limits

    def draw_plots(self):
        selected_file_data = self.file_manager.get_selected_files_data()

        if not selected_file_data:
            if self.main_window.is_dark_mode:
                FIG_BG, TEXT_COLOR = '#1E1E1E', 'white'
            else:
                FIG_BG, TEXT_COLOR = '#F0F0F0', 'black'
            self.canvas.fig.clf();
            self.canvas.fig.patch.set_facecolor(FIG_BG);
            self.canvas.ax = self.canvas.fig.add_subplot(111)
            self.canvas.ax.set_facecolor(FIG_BG);
            self.canvas.ax.axis('off')
            self.canvas.ax.text(0.5, 0.5, "Load and Select Files for Plotting", ha='center', va='center', fontsize=12,
                                color=TEXT_COLOR)
            self.canvas.draw();
            self.canvas.plot_settings = {};
            self.canvas.custom_titles = {};
            self.canvas.custom_x_labels = {};
            self.canvas.custom_y_labels = {};
            return

        checked_mpa_cols = [];
        unique_sensor_bases_to_plot = []
        for i in range(self.checklist_layout.count()):
            widget = self.checklist_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                checked_mpa_cols.append(widget.property('full_name'))
                unique_sensor_bases_to_plot.append(widget.text())

        if not checked_mpa_cols:
            if self.main_window.is_dark_mode:
                FIG_BG, TEXT_COLOR = '#1E1E1E', 'white'
            else:
                FIG_BG, TEXT_COLOR = '#F0F0F0', 'black'
            self.canvas.fig.clf();
            self.canvas.fig.patch.set_facecolor(FIG_BG);
            self.canvas.ax = self.canvas.fig.add_subplot(111)
            self.canvas.ax.set_facecolor(FIG_BG);
            self.canvas.ax.axis('off')
            self.canvas.ax.text(0.5, 0.5, f"Select a shared sensor to plot '{self.y_axis_metric}'.", ha='center',
                                va='center', fontsize=10, color=TEXT_COLOR)
            self.canvas.draw();
            return

        df_processed_list = [];
        legend_names = []
        for df_raw, custom_name in selected_file_data:
            df_processed = pd.DataFrame({'Frequency (Hz)': df_raw['Frequency (Hz)']});
            legend_names.append(custom_name)
            for full_col_name in checked_mpa_cols:
                if full_col_name in df_raw.columns and 'Frequency (Hz)' in df_raw.columns:
                    peak_mpa_series = df_raw[full_col_name]
                    freq_series = df_raw['Frequency (Hz)']
                    plot_series = calculate_derived_data(peak_mpa_series, freq_series, self.y_axis_metric,
                                                         self.is_rms_selected)
                    df_processed[full_col_name] = plot_series
            df_processed_list.append(df_processed)

        limits = self._get_axis_limits()
        self.canvas.update_plot(df_processed_list, legend_names, unique_sensor_bases_to_plot, self.y_axis_metric,
                                self.y_log_scale, limits)


# ----------------------------------------------------------------------
# --- TAB 3: Transfer Function Complex Sum & Scaling Tool ---
# ----------------------------------------------------------------------

class MappingDialog(QDialog):
    FORCE_TYPES = [
        "Select Component...",
        "Tan Magnitude (N)",
        "Tan Phase (Deg)",
        "Rad Magnitude (N)",
        "Rad Phase (Deg)"
    ]

    SPATIAL_ORDER_PLACEHOLDER = "Select Spatial Order (m)..."

    def __init__(self, subcase_ids, unique_spatial_orders, force_headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map TF Subcase IDs to Force Components & Spatial Order");
        self.setMinimumWidth(750)
        self.subcase_ids = subcase_ids;
        self.unique_spatial_orders = [str(int(o)) for o in sorted(unique_spatial_orders)]
        self.force_headers = force_headers
        self.mappings = {};
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout();
        main_layout.setSpacing(15);
        main_layout.setContentsMargins(20, 20, 20, 20)
        header_label = QLabel(
            "Mapping required: For each TF Subcase ID, select the corresponding **Spatial Order (m)** in the Force File, and the **Magnitude/Phase** columns to be used.")
        header_label.setObjectName("DialogInstructionLabel");
        main_layout.addWidget(header_label)

        scroll_area = QScrollArea();
        scroll_area.setWidgetResizable(True);
        scroll_area.setObjectName("DialogScrollArea")
        scroll_content = QWidget();
        scroll_content.setObjectName("DialogScrollContent")
        form_layout = QVBoxLayout(scroll_content);
        form_layout.setAlignment(Qt.AlignmentFlag.AlignTop);
        form_layout.setContentsMargins(10, 10, 10, 10);
        form_layout.setSpacing(10)

        valid_mag_headers = [h for h in self.force_headers if 'Magnitude' in h]
        valid_phase_headers = [h for h in self.force_headers if 'Phase' in h]

        final_mag_options = [self.FORCE_TYPES[0]] + valid_mag_headers
        final_phase_options = [self.FORCE_TYPES[0]] + valid_phase_headers

        for subcase_id in sorted(list(self.subcase_ids)):
            group_box = QGroupBox(f"TF Subcase **{subcase_id}** Mapping");
            group_box.setObjectName("MappingGroupBox")
            grid_layout = QGridLayout();
            grid_layout.setSpacing(10)

            m_combo = QComboBox();
            m_combo.addItems([self.SPATIAL_ORDER_PLACEHOLDER] + self.unique_spatial_orders)
            grid_layout.addWidget(QLabel("1. Spatial Order (m):"), 0, 0);
            grid_layout.addWidget(m_combo, 0, 1)

            mag_combo = QComboBox();
            mag_combo.addItems(final_mag_options)
            grid_layout.addWidget(QLabel("2. Magnitude Source:"), 1, 0);
            grid_layout.addWidget(mag_combo, 1, 1)

            phase_combo = QComboBox();
            phase_combo.addItems(final_phase_options)
            grid_layout.addWidget(QLabel("3. Phase Source:"), 2, 0);
            grid_layout.addWidget(phase_combo, 2, 1)

            group_box.setLayout(grid_layout);
            form_layout.addWidget(group_box)

            self.mappings[subcase_id] = {'m_combo': m_combo, 'mag_combo': mag_combo, 'phase_combo': phase_combo}

        scroll_area.setWidget(scroll_content);
        main_layout.addWidget(scroll_area)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.validate_and_accept);
        button_box.rejected.connect(self.reject)
        button_ok = button_box.button(QDialogButtonBox.StandardButton.Ok);
        button_ok.setText("Confirm & Continue");
        button_ok.setObjectName("PrimaryButton")
        button_cancel = button_box.button(QDialogButtonBox.StandardButton.Cancel);
        button_cancel.setObjectName("SecondaryButton")
        main_layout.addWidget(button_box);
        self.setLayout(main_layout)

    def validate_and_accept(self):
        try:
            for subcase_id, combo_data in self.mappings.items():
                m_val = combo_data['m_combo'].currentText().strip()
                mag_val = combo_data['mag_combo'].currentText().strip()
                phase_val = combo_data['phase_combo'].currentText().strip()

                if m_val == self.SPATIAL_ORDER_PLACEHOLDER or not m_val:
                    raise ValueError(f"Spatial Order (m) is missing for Subcase {subcase_id}.")
                if mag_val == self.FORCE_TYPES[0] or not mag_val:
                    raise ValueError(f"Magnitude Source is missing for Subcase {subcase_id}.")
                if phase_val == self.FORCE_TYPES[0] or not phase_val:
                    raise ValueError(f"Phase Source is missing for Subcase {subcase_id}.")
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Mapping Error", str(e))

    def get_mappings(self):
        final_mappings = {}
        for subcase_id, combo_data in self.mappings.items():
            final_mappings[subcase_id] = {
                'm': int(combo_data['m_combo'].currentText().strip()),
                'mag': combo_data['mag_combo'].currentText().strip(),
                'phase': combo_data['phase_combo'].currentText().strip()
            }
        return final_mappings


class TransferFunctionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.tf_folder_path = "";
        self.force_file_path = ""
        self.aggregate_peak_mpa = {};
        self.aggregate_acoustic_level = {}
        self.frequency_data = None;
        self.force_df_raw = None
        self.unique_spatial_orders = []
        self.main_window_ref = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self);
        main_layout.setContentsMargins(30, 30, 30, 30);
        main_layout.setSpacing(30)

        self.left_panel = QWidget();
        left_vbox = QVBoxLayout(self.left_panel);
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(20)

        frame_file = QGroupBox("1. Input Data Configuration");
        frame_file.setObjectName("FlatGroupBox")
        layout_file = QGridLayout(frame_file);
        layout_file.setContentsMargins(15, 25, 15, 15);
        layout_file.setSpacing(15)

        def add_input_row(label_text, line_edit, button_text, command, row, tooltip):
            label = QLabel(label_text);
            label.setMinimumWidth(200);
            layout_file.addWidget(label, row, 0)
            layout_file.addWidget(line_edit, row, 1);
            line_edit.setToolTip(tooltip);
            btn = QPushButton(button_text)
            btn.clicked.connect(command);
            btn.setObjectName("SecondaryButton_Browse");
            btn.setMaximumWidth(120)
            btn.setToolTip(tooltip);
            layout_file.addWidget(btn, row, 2);
            line_edit.setPlaceholderText("Select a folder or file path...");
            line_edit.setReadOnly(True)

        tf_tooltip = "Select the folder containing all Transfer Function (TF) files (CSV/Excel).";
        force_tooltip = "Select the single Excel/CSV file containing the Forcing data (long format: k, m, F_mag, F_phase)."

        self.tf_path_edit = QLineEdit();
        self.force_path_edit = QLineEdit()
        add_input_row("Transfer Function Folder (TF):", self.tf_path_edit, "Browse", lambda: self.browse_folder('tf'),
                      0, tf_tooltip)
        add_input_row("Forcing Order File (Long Format):", self.force_path_edit, "Browse",
                      lambda: self.browse_file('force'), 1, force_tooltip)

        left_vbox.addWidget(frame_file)

        options_group = QGroupBox("2. Processing Options (Output Conversions)");
        options_group.setObjectName("FlatGroupBox")
        options_layout = QGridLayout(options_group);
        options_layout.setContentsMargins(15, 25, 15, 15);
        options_layout.setSpacing(15)

        self.rms_checkbox = QCheckBox("1. Convert final Magnitude to **RMS** (Mag / 1.414)");
        self.rms_checkbox.setMinimumWidth(250)
        self.db_checkbox = QCheckBox("2. Calculate **dB Level** in Summary Report");
        self.db_checkbox.setMinimumWidth(250)
        self.dba_checkbox = QCheckBox("3. Calculate **dBA Level** in Summary Report");
        self.dba_checkbox.setMinimumWidth(250)

        self.rms_checkbox.setChecked(True);
        self.db_checkbox.setChecked(True);
        self.dba_checkbox.setChecked(True)
        self.db_checkbox.stateChanged.connect(self.handle_db_toggle)

        options_layout.addWidget(self.rms_checkbox, 0, 0);
        options_layout.addWidget(self.db_checkbox, 0, 1)
        options_layout.addWidget(self.dba_checkbox, 0, 2)

        info_label = QLabel(
            "Mandatory Output: **Total Scaled Sum Magnitude (Peak MPa)** is saved in individual files and the first summary file.");
        info_label.setObjectName("InfoLabel")
        options_layout.addWidget(info_label, 1, 0, 1, 3)

        left_vbox.addWidget(options_group)

        process_button = QPushButton("⚡ Execute Batch Scaling, Summation & Save Reports")
        process_button.setObjectName("PrimaryButton");
        process_button.setMinimumHeight(45)
        process_button.setFont(QFont('Segoe UI', 11, QFont.Weight.Bold));
        process_button.clicked.connect(self.start_processing_workflow)
        left_vbox.addWidget(process_button);
        left_vbox.addStretch(1)

        log_group = QGroupBox("3. Status Log");
        log_group.setObjectName("FlatGroupBox")
        log_layout = QVBoxLayout(log_group);
        log_layout.setContentsMargins(10, 20, 10, 10)
        self.log_text = QTextEdit();
        self.log_text.setReadOnly(True)
        self.log_text.setText("Ready to select Transfer Function folder and Force file (XLS, XLSX, CSV).");
        self.log_text.setObjectName("LogTextEdit")
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(self.left_panel, 1);
        main_layout.addWidget(log_group, 1)

    def set_main_window_ref(self, main_window):
        self.main_window_ref = main_window

    def handle_db_toggle(self):
        db_checked = self.db_checkbox.isChecked()
        if not db_checked:
            self.dba_checkbox.setChecked(False);
            self.dba_checkbox.setEnabled(False)
        else:
            self.dba_checkbox.setEnabled(True)

    def log(self, message):
        self.log_text.append(message);
        self.log_text.ensureCursorVisible()

    def browse_folder(self, file_type):
        folder_name = QFileDialog.getExistingDirectory(self, "Open Transfer Function Data Folder")
        if folder_name:
            self.tf_folder_path = folder_name
            self.tf_path_edit.setText(folder_name)
            self.log(f"TF Folder loaded: {os.path.basename(folder_name)}")

    def browse_file(self, file_type):
        file_filter = "Data Files (*.xls *.xlsx *.csv);;Excel Files (*.xls *.xlsx);;CSV Files (*.csv)"
        title = "Open Force Data File (Long Format)"
        file_name, _ = QFileDialog.getOpenFileName(self, title, os.path.expanduser("~"), file_filter)
        if file_name:
            self.force_file_path = file_name
            self.force_path_edit.setText(file_name)
            self.log(f"Force File loaded: {os.path.basename(file_name)}")

    def load_data(self, file_path):
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df_temp = pd.read_excel(file_path, header=None)
                header_row_index = -1
                for i, row in df_temp.iterrows():
                    if any('frequency' in str(cell).lower() for cell in row) or any(
                            'temporal order' in str(cell).lower() for cell in row):
                        header_row_index = i;
                        break
                df = pd.read_excel(file_path, header=header_row_index) if header_row_index != -1 else pd.read_excel(
                    file_path)

            df.columns = [str(col).strip() for col in df.columns]
            df.dropna(axis=0, how='all', inplace=True);
            df.dropna(axis=1, how='all', inplace=True)
            return df
        except Exception as e:
            self.log(f"ERROR: Could not load data from {os.path.basename(file_path)}: {e}")
            raise

    def get_file_list(self, folder_path):
        if not folder_path: return []
        compatible_extensions = ('.xls', '.xlsx', '.csv')
        file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     f.lower().endswith(compatible_extensions)]
        return file_list

    def process_force_data(self, df_force_raw):
        df_force = df_force_raw.copy()

        col_m = [c for c in df_force.columns if 'Spatial Order' in c]
        if not col_m:
            raise ValueError("Force file must contain 'Spatial Order (m)' column in the header.")
        else:
            col_m = col_m[0]
            df_force.rename(columns={col_m: 'm'}, inplace=True)
            df_force['m'] = pd.to_numeric(df_force['m'], errors='coerce').fillna(0).astype(int)
            unique_spatial_orders = df_force['m'].unique()
            unique_spatial_orders = unique_spatial_orders[unique_spatial_orders != 0]

        self.force_df_raw = df_force

        return df_force, unique_spatial_orders

    def start_processing_workflow(self):
        if not self.tf_folder_path or not self.force_file_path:
            QMessageBox.warning(self, "Error",
                                "Please select both the **Transfer Function Folder** and **Force File**.");
            return

        tf_file_list = self.get_file_list(self.tf_folder_path)
        if not tf_file_list:
            QMessageBox.warning(self, "Error", "No compatible data files found in the selected TF folder.");
            return

        self.log("\n--- Starting Batch Data Processing Workflow ---");
        self.aggregate_peak_mpa = {};
        self.aggregate_acoustic_level = {}
        self.frequency_data = None
        self.force_df_raw = None
        self.unique_spatial_orders = []

        try:
            df_force, self.unique_spatial_orders = self.process_force_data(self.load_data(self.force_file_path))
            self.log(
                f"Force file loaded and preprocessed. Found {len(self.unique_spatial_orders)} unique spatial orders (m).")

            first_tf_file_path = tf_file_list[0];
            df_tf_first = self.load_data(first_tf_file_path)
            subcase_ids = self.identify_subcases(df_tf_first.columns)
            if not subcase_ids:
                QMessageBox.critical(self, "Error",
                                     f"Could not identify any Subcase IDs (e.g., 'Subcase 1234') in the headers of the first TF file: {os.path.basename(first_tf_file_path)}");
                return

            dialog = MappingDialog(subcase_ids, self.unique_spatial_orders, df_force.columns.tolist(), self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.log("Mapping cancelled by user. Processing stopped.");
                return
            mappings = dialog.get_mappings()
            options = {'rms': self.rms_checkbox.isChecked(), 'db': self.db_checkbox.isChecked(),
                       'dba': self.dba_checkbox.isChecked()}
            self.log(f"Processing options: RMS={options['rms']}, dB={options['db']}, dBA={options['dba']}")

            scalar_force_map = self._pre_calculate_scalar_forces(mappings)
            self.log("Scalar forces successfully calculated based on mapping.")

        except Exception as e:
            self.log(f"CRITICAL SETUP ERROR: {e}")
            QMessageBox.critical(self, "Setup Error", f"A critical error occurred during setup or mapping: {e}");
            return

        successful_files = 0
        for tf_file_path in tf_file_list:
            try:
                self.log(f"\nProcessing file: **{os.path.basename(tf_file_path)}**")
                df_tf = self.load_data(tf_file_path)

                results_df = self.process_single_file(df_tf, scalar_force_map, mappings, options, tf_file_path)

                if results_df is not None:
                    self.save_individual_file(results_df, tf_file_path)
                    self.aggregate_results(results_df, tf_file_path, options)
                    successful_files += 1

            except Exception as e:
                import traceback
                self.log(f"SKIPPING FILE (Fatal error during processing): {os.path.basename(tf_file_path)}: {e}")
                QMessageBox.warning(self, "File Error", f"Skipping {os.path.basename(tf_file_path)} due to error: {e}")

        if successful_files > 0:
            try:
                self.save_summary_files(options)
                QMessageBox.information(self, "Batch Success",
                                        f"Batch processing complete. **{successful_files}** files processed, and summary files saved to '{os.path.join(self.tf_folder_path, 'Processed_TF_Outputs')}'.")

                if self.main_window_ref and hasattr(self.main_window_ref, 'plotting_tab'):
                    self.main_window_ref.plotting_tab.file_manager.load_summary_file(
                        os.path.join(self.tf_folder_path, "Processed_TF_Outputs", "CONSOLIDATED_RAW_MPA.xlsx")
                    )
                    self.main_window_ref.switch_tab(3)

            except Exception as e:
                self.log(f"ERROR: Failed to save consolidated summary files (Check if files are open). Details: {e}")
                QMessageBox.critical(self, "Save Error",
                                     "Failed to save consolidated summary file. Please ensure the file is closed.")
        else:
            QMessageBox.warning(self, "Batch Failed", "No files were processed successfully.")

    def _pre_calculate_scalar_forces(self, mappings):
        scalar_force_map = {}

        for subcase_id, mapping_data in mappings.items():
            spatial_order_m = mapping_data['m']
            mag_col_name = mapping_data['mag']
            phase_col_name = mapping_data['phase']

            force_row = self.force_df_raw[self.force_df_raw['m'] == spatial_order_m]

            if force_row.empty:
                self.log(f" WARNING: Scalar force for m={spatial_order_m} not found in Force File. Using 0 N.")
                scalar_force_map[subcase_id] = 0.0 + 0j
                continue

            try:
                force_mag = pd.to_numeric(force_row.iloc[0].get(mag_col_name, 0.0), errors='coerce')
                force_phase = pd.to_numeric(force_row.iloc[0].get(phase_col_name, 0.0), errors='coerce')
            except Exception as e:
                self.log(
                    f" CRITICAL: Failed to read data for m={spatial_order_m} from columns {mag_col_name}/{phase_col_name}. Error: {e}")
                force_mag, force_phase = 0.0, 0.0

            if pd.isna(force_mag) or pd.isna(force_phase) or force_mag == 0.0:
                self.log(
                    f" WARNING: Force (Mag/Phase) for m={spatial_order_m} is invalid/zero (Mag={force_mag:.2e}). Using 0 N.")
                complex_force_scalar = 0.0 + 0j
            else:
                complex_force_scalar = self.convert_polar_to_complex(force_mag, force_phase)
                self.log(f" Scalar Force for Subcase {subcase_id} (m={spatial_order_m}) calculated:")
                self.log(f" -> Mag: {force_mag:.4e} N, Phase: {force_phase:.2f} Deg")

            scalar_force_map[subcase_id] = complex_force_scalar

        return scalar_force_map

    def identify_subcases(self, columns):
        unique_cases = set()
        regex = re.compile(r'Subcase\s*(\S+)\s*\(', re.IGNORECASE)
        for col in columns:
            match = regex.search(str(col))
            if match: unique_cases.add(match.group(1).strip())
        return sorted(list(unique_cases))

    def convert_polar_to_complex(self, magnitude, phase_degrees):
        phase_radians = np.deg2rad(phase_degrees)
        return magnitude * np.exp(1j * phase_radians)

    def process_single_file(self, df_tf, scalar_force_map, subcase_mappings, options, tf_file_path):
        frequency_column_name = [col for col in df_tf.columns if 'Frequency' in str(col)]
        if not frequency_column_name:
            raise ValueError("Frequency column not found.")
        frequency_column_name = frequency_column_name[0]
        frequencies = df_tf[frequency_column_name].values

        results_data = {'Frequency': df_tf[frequency_column_name].copy()}
        total_complex_scaled_sum = np.zeros(len(df_tf), dtype=complex)

        if self.frequency_data is None:
            self.frequency_data = frequencies
        elif not np.array_equal(self.frequency_data, frequencies):
            raise ValueError("Frequency vector does not match the first file. Cannot aggregate.")

        for subcase_id in subcase_mappings.keys():
            subcase_id_pattern = re.escape(subcase_id)

            tf_mag_cols = [col for col in df_tf.columns if
                           re.search(fr'Subcase\s*{subcase_id_pattern}\s*\(.*?\)\s*Mag', col, re.IGNORECASE)]
            tf_phase_cols = [col for col in df_tf.columns if
                             re.search(fr'Subcase\s*{subcase_id_pattern}\s*\(.*?\)\s*Phase', col, re.IGNORECASE)]

            if not tf_mag_cols or not tf_phase_cols:
                tf_mag_cols = [col for col in df_tf.columns if subcase_id in col and 'MAG' in col.upper()]
                tf_phase_cols = [col for col in df_tf.columns if subcase_id in col and 'PHASE' in col.upper()]
                if not tf_mag_cols or not tf_phase_cols:
                    continue

            tf_mag = pd.to_numeric(df_tf[tf_mag_cols[0]], errors='coerce').fillna(0).values
            tf_phase = pd.to_numeric(df_tf[tf_phase_cols[0]], errors='coerce').fillna(0).values

            force_scalar = scalar_force_map.get(subcase_id, 0.0 + 0j)

            tf_complex = self.convert_polar_to_complex(tf_mag, tf_phase)

            scaled_complex_response = tf_complex * force_scalar

            total_complex_scaled_sum += scaled_complex_response

            results_data[f'{subcase_id}_Scaled_Real'] = np.real(scaled_complex_response)
            results_data[f'{subcase_id}_Scaled_Imag'] = np.imag(scaled_complex_response)

        final_mag = np.abs(total_complex_scaled_sum)
        final_phase = np.rad2deg(np.angle(total_complex_scaled_sum))

        results_data['Total_Scaled_Sum_Magnitude_MPa_Peak'] = final_mag
        results_data['Total_Scaled_Sum_Phase_deg'] = final_phase

        if options['db'] or options['dba']:
            pressure_peak_pa = final_mag * MPA_TO_PA

            if options['rms']:
                pressure_base_pa = pressure_peak_pa / SQRT_2
                results_data['Total_Scaled_Sum_Magnitude_MPa_RMS'] = final_mag / SQRT_2
                level_suffix = "RMS"
            else:
                pressure_base_pa = pressure_peak_pa
                level_suffix = "Peak"

            pressure_base_pa_safe = np.maximum(pressure_base_pa, NUMERICAL_FLOOR_PA)
            final_spl_db = 20 * np.log10(pressure_base_pa_safe / P_REF)

            results_data[f'Total_SPL_dB_{level_suffix}'] = final_spl_db

            if options['dba']:
                R_A = a_weighting_correction_new(frequencies)
                final_spl_dba = final_spl_db + R_A
                results_data[f'Total_SPL_dBA_{level_suffix}'] = final_spl_dba

        self.log(
            f"Successfully processed {os.path.basename(tf_file_path)}. Max Final Magnitude: {final_mag.max():.4e} MPa.")
        return pd.DataFrame(results_data)

    def save_individual_file(self, results_df, tf_file_path):
        base_name = os.path.basename(tf_file_path);
        name, ext = os.path.splitext(base_name)
        output_dir = os.path.join(self.tf_folder_path, "Processed_TF_Outputs");
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = os.path.join(output_dir, f"{name}_PROCESSED_SUM.xlsx")
        try:
            results_df.to_excel(output_file_name, index=False, engine='openpyxl')
            self.log(f"-> Individual results saved: {os.path.basename(output_file_name)}")
        except Exception as e:
            self.log(f"ERROR: Could not save individual file {os.path.basename(output_file_name)}: {e}")

    def extract_column_name_suffix(self, filename):
        name_no_ext = os.path.splitext(os.path.basename(filename))[0]
        match = re.search(r'(?:ORDER_\d+[\s_-]+)(.+?)(?:[\s_-]+-|[\s_-]+SPL|[\s_-]+Pressure|$)', name_no_ext,
                          flags=re.IGNORECASE)
        if match:
            base_name_raw = match.group(1).strip(' _-')
        else:
            base_name_raw = re.sub(r'[\s_-]+(SPL|Pressure|SPLPressure|Mic|Sensor|ORDER_\d+)$', '', name_no_ext,
                                   flags=re.IGNORECASE).strip(' _-')
            if not base_name_raw: base_name_raw = name_no_ext

        base_name = re.sub(r'[_\-\s]+', ' ', base_name_raw).strip()
        parts = [p.capitalize() for p in base_name.split()];
        base_name = " ".join(parts)
        if not base_name: base_name = "Response"
        return base_name.replace(' ', '_')

    def aggregate_results(self, results_df, tf_file_path, options):
        filename = os.path.basename(tf_file_path);
        file_suffix = self.extract_column_name_suffix(filename)
        peak_mpa_col = 'Total_Scaled_Sum_Magnitude_MPa_Peak'

        if peak_mpa_col in results_df.columns:
            header_name = f'{file_suffix}_MPa'
            self.aggregate_peak_mpa[header_name] = results_df[peak_mpa_col].values

        if options['db'] or options['dba']:
            level_suffix = "RMS" if options['rms'] else "Peak"
            if options['dba']:
                acoustic_col_name = f'Total_SPL_dBA_{level_suffix}';
                acoustic_header_suffix = 'dBA'
            elif options['db']:
                acoustic_col_name = f'Total_SPL_dB_{level_suffix}';
                acoustic_header_suffix = 'dB'
            else:
                return

            if acoustic_col_name in results_df.columns:
                header_name = f'{file_suffix}_{acoustic_header_suffix}'
                self.aggregate_acoustic_level[header_name] = results_df[acoustic_col_name].values

    def save_summary_files(self, options):
        if self.frequency_data is None: self.log("ERROR: Frequency data is missing. Cannot save summary files."); return
        output_dir = os.path.join(self.tf_folder_path, "Processed_TF_Outputs");
        os.makedirs(output_dir, exist_ok=True)

        if self.aggregate_peak_mpa:
            peak_mpa_df = pd.DataFrame({'Frequency (Hz)': self.frequency_data, **self.aggregate_peak_mpa})
            peak_mpa_output_file = os.path.join(output_dir, "CONSOLIDATED_RAW_MPA.xlsx")

            global CONSOLIDATED_DATA;
            CONSOLIDATED_DATA = peak_mpa_df
            try:
                peak_mpa_df.to_excel(peak_mpa_output_file, index=False, engine='openpyxl')
                self.log(f"-> Summary file 1 (Raw MPa) saved: {os.path.basename(peak_mpa_output_file)}")
            except Exception as e:
                self.log(f"ERROR: Could not save Raw MPa summary file: {e}")

        if self.aggregate_acoustic_level:
            acoustic_df = pd.DataFrame({'Frequency (Hz)': self.frequency_data, **self.aggregate_acoustic_level})
            acoustic_output_file = os.path.join(output_dir, f"CONSOLIDATED_SPL_REPORT.xlsx")
            try:
                acoustic_df.to_excel(acoustic_output_file, index=False, engine='openpyxl')
                self.log(f"-> Summary file 2 (SPL Report) saved: {os.path.basename(acoustic_output_file)}")
            except Exception as e:
                self.log(f"ERROR: Could not save SPL Report summary file: {e}")


# ----------------------------------------------------------------------
# --- TAB 2: Forcing Analysis (CORRECTED FFT & TABLES FOR 54 TEETH) ---
# ----------------------------------------------------------------------

class AnalysisSignals(QObject):
    warning_signal = pyqtSignal(str, str)


class NVHAnalyzer:
    def __init__(self):
        self.radial_magnitude = None;
        self.radial_angle = None
        self.tangential_magnitude = None;
        self.tangential_angle = None
        self.temporal_order = None;
        self.spatial_order = None
        self.total_teeth = 48;
        self.rpm = None
        self.rotor_position = None
        self.dense_data_cache = {};
        self.TEMPORAL_ORDERS_COUNT = 198
        self.signals = AnalysisSignals()

    def load_excel(self, filepath):
        if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
        xls = pd.ExcelFile(filepath);
        sheet_name = 'Torque_Characteristic'
        target_sheet = next((s for s in xls.sheet_names if s.lower() == sheet_name.lower()), None)
        if not target_sheet: raise ValueError(
            f"Required sheet '{sheet_name}' not found. Found sheets: {', '.join(xls.sheet_names)}")

        df = pd.read_excel(filepath, sheet_name=target_sheet, header=9)
        if df.empty or len(df.index) < 2: raise ValueError(
            "The Excel sheet is empty or contains insufficient data after reading the header.")

        df = df.iloc[1:].reset_index(drop=True)
        df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9_]+', '_', regex=True)
        rotor_cols = [c for c in df.columns if 'rotor' in c and 'position' in c]
        if rotor_cols:
            self.rotor_position = pd.to_numeric(df[rotor_cols[0]], errors='coerce').dropna().values
        else:
            self.rotor_position = None
        return df

    def process_fft(self, df, force_type="radial"):
        pattern = fr'^force_{force_type}_\d+$'
        force_cols = [col for col in df.columns if re.match(pattern, col)]
        force_cols.sort(key=lambda x: int(re.search(r'(\d+)\( ', x).group(1)) if re.search(r'(\d+) \)', x) else 0)

        if not force_cols: raise ValueError(f"No force data columns found matching pattern '{force_type}'.")

        force_df = df[force_cols].apply(pd.to_numeric, errors='coerce')
        force_data = force_df.dropna(how='all').values
        if force_data.size == 0: raise ValueError(
            f"All {force_type} data rows were non-numeric or empty after cleaning.")

        L_time, num_cols = force_data.shape
        if num_cols == 54:
            self.total_teeth = 54
        elif num_cols == 48:
            self.total_teeth = 48
        else:
            self.total_teeth = 48

        if num_cols < self.total_teeth:
            replication = int(self.total_teeth / num_cols)
            rad_force = np.tile(force_data, (1, replication))
        else:
            rad_force = force_data

        L_time, L_space = rad_force.shape

        fft_result = np.fft.fft2(rad_force)
        fft_shifted = np.fft.fftshift(fft_result)

        self.spatial_order = np.fft.fftshift(np.fft.fftfreq(L_space, d=1 / L_space))

        mid_time = L_time // 2
        magnitude_full = np.abs(fft_shifted) / (L_time * L_space)
        angle_dat_full = np.angle(fft_shifted)

        magnitude = magnitude_full[mid_time:, :]
        angle_dat = angle_dat_full[mid_time:, :]
        magnitude[1:, :] = 2 * magnitude[1:, :]

        if force_type == "radial":
            self.radial_magnitude = magnitude
            self.radial_angle = angle_dat
        elif force_type == "tangential":
            self.tangential_magnitude = magnitude
            self.tangential_angle = angle_dat

        if self.rotor_position is not None and len(self.rotor_position) > 1:
            angle_range = self.rotor_position.max() - self.rotor_position.min()
            if angle_range < 1: angle_range = 360.0
            scaling_factor = 360.0 / angle_range
        else:
            scaling_factor = 1.0

        raw_orders = np.arange(magnitude.shape[0]) * scaling_factor
        self.temporal_order = np.round(raw_orders, decimals=2)

        self.dense_data_cache = {}

    def get_dense_data(self, force_type="radial", exclude_zero_order=True):
        key = (force_type, exclude_zero_order);
        if key in self.dense_data_cache: return self.dense_data_cache[key]

        magnitude = self.radial_magnitude if force_type == "radial" else self.tangential_magnitude
        angle_dat = self.radial_angle if force_type == "radial" else self.tangential_angle

        if magnitude is None or self.temporal_order is None or self.spatial_order is None: return None

        df_mag = pd.DataFrame(magnitude.T, index=self.spatial_order, columns=self.temporal_order)
        df_ph = pd.DataFrame(angle_dat.T, index=self.spatial_order, columns=self.temporal_order)

        max_time_ord = int(np.ceil(self.temporal_order.max()))
        time_range = np.arange(0, max_time_ord + 1, 1)

        min_space_ord = int(np.floor(self.spatial_order.min()))
        max_space_ord = int(np.ceil(self.spatial_order.max()))
        space_range = np.arange(min_space_ord, max_space_ord + 1, 1)

        col_map = {c: int(round(c)) for c in df_mag.columns}
        df_mag_T = df_mag.T.groupby(col_map).mean().T
        df_ph_T = df_ph.T.groupby(col_map).mean().T

        df_mag_T.index = np.round(df_mag_T.index).astype(int)
        df_ph_T.index = np.round(df_ph_T.index).astype(int)

        df_mag_final = df_mag_T.groupby(df_mag_T.index).mean()
        df_ph_final = df_ph_T.groupby(df_ph_T.index).mean()

        df_mag_dense = df_mag_final.reindex(index=space_range, columns=time_range, fill_value=0.0)
        df_ph_dense = df_ph_final.reindex(index=space_range, columns=time_range, fill_value=0.0)

        if exclude_zero_order:
            time_range = time_range[1:]
            df_mag_dense = df_mag_dense.iloc[:, 1:]
            df_ph_dense = df_ph_dense.iloc[:, 1:]

        result = (time_range, space_range, df_mag_dense.values, df_ph_dense.values)
        self.dense_data_cache[key] = result
        return result

    def export_order_data_to_excel(self, output_folder, temporal_orders, spatial_orders, rpm=1000):
        """Exports selected orders in the exact Excel format as per your template."""
        try:
            os.makedirs(output_folder, exist_ok=True)

            all_rows = []

            # Get dense data (excluding order 0)
            dense_rad = self.get_dense_data("radial", exclude_zero_order=True)
            dense_tan = self.get_dense_data("tangential", exclude_zero_order=True)
            if dense_rad is None or dense_tan is None:
                return False, "FFT data not available."

            t_range_rad, s_range_rad, mag_rad, ph_rad = dense_rad
            t_range_tan, s_range_tan, mag_tan, ph_tan = dense_tan

            for t_ord in temporal_orders:
                t_ord_int = int(t_ord)
                freq_hz = round(t_ord_int * rpm / 60.0, 2) # Frequency = order * RPM / 60

                # Find index for temporal order (allow small tolerance)
                t_idx_rad = np.argmin(np.abs(t_range_rad - t_ord_int))
                t_idx_tan = np.argmin(np.abs(t_range_tan - t_ord_int))

                for s_ord in spatial_orders:
                    s_ord_int = int(s_ord)

                    # Radial data
                    if s_ord_int in s_range_rad:
                        s_idx_rad = np.where(s_range_rad == s_ord_int)[0][0]
                        rad_mag = mag_rad[s_idx_rad, t_idx_rad]
                        rad_phase = np.rad2deg(ph_rad[s_idx_rad, t_idx_rad])
                    else:
                        rad_mag = 0.0
                        rad_phase = 0.0

                    # Tangential data
                    if s_ord_int in s_range_tan:
                        s_idx_tan = np.where(s_range_tan == s_ord_int)[0][0]
                        tan_mag = mag_tan[s_idx_tan, t_idx_tan]
                        tan_phase = np.rad2deg(ph_tan[s_idx_tan, t_idx_tan])
                    else:
                        tan_mag = 0.0
                        tan_phase = 0.0

                    all_rows.append({
                        'Temporal Order (k)': t_ord_int,
                        'Spatial Order (m)': s_ord_int,
                        'Frequency Bin (Hz)': freq_hz,
                        'RPM': rpm,
                        'Tan Magnitude (N)': tan_mag,
                        'Tan Phase (Deg)': tan_phase,
                        'Rad Magnitude (N)': rad_mag,
                        'Rad Phase (Deg)': rad_phase
                    })

            if not all_rows:
                return False, "No matching orders found."

            df_export = pd.DataFrame(all_rows)

            # Sort exactly like your example: first by Temporal, then by Spatial (negative to positive)
            df_export = df_export.sort_values(['Temporal Order (k)', 'Spatial Order (m)']).reset_index(drop=True)

            filename = f"Forcing_Orders_k{'_'.join(map(str, map(int, temporal_orders)))}_m{'_'.join(map(str, map(int, spatial_orders)))}.xlsx"
            output_path = os.path.join(output_folder, filename)

            df_export.to_excel(output_path, index=False, engine='openpyxl')

            return True, f"Excel exported successfully!\nSaved to: {os.path.basename(output_path)}"

        except Exception as e:
            return False, f"Export failed: {str(e)}"


class DualFFTWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, analyzer, df, rpm):
        super().__init__()
        self.analyzer = analyzer;
        self.df = df;
        self.rpm = rpm

    def run(self):
        try:
            self.analyzer.rpm = self.rpm
            self.analyzer.process_fft(self.df, "radial")
            self.analyzer.process_fft(self.df, "tangential")
            self.finished.emit(True, "Success")
        except Exception as e:
            self.finished.emit(False, str(e))


class ScientificTableModel(QAbstractTableModel):
    def __init__(self, data, row_headers, col_headers, highlight_threshold=None):
        super().__init__()
        self._data = data
        self._row_headers = row_headers
        self._col_headers = col_headers
        self._highlight_threshold = highlight_threshold
        self.high_color = QColor(255, 204, 204)
        self.norm_bg = QColor(255, 255, 255)
        self.alt_bg = QColor(249, 249, 249)

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1] + 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        row, col = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return str(int(self._row_headers[row]))
            return f"{self._data[row, col - 1]:.4f}"
        if role == Qt.ItemDataRole.BackgroundRole:
            if col > 0 and self._highlight_threshold and self._data[row, col - 1] > self._highlight_threshold:
                return self.high_color
            return self.alt_bg if row % 2 else self.norm_bg
        if role == Qt.ItemDataRole.TextAlignmentRole: return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return "S-Ord" if section == 0 else str(int(self._col_headers[section - 1]))
            if orientation == Qt.Orientation.Vertical: return str(section + 1)
        return None


class NVH_Forcing_App(QWidget):
    def __init__(self):
        super().__init__()
        self.analyzer = NVHAnalyzer();
        self.df = None;
        self.worker = None
        self.output_folder = os.getcwd();
        self.init_ui()
        self.analyzer.signals.warning_signal.connect(self.show_warning_dialog)

    def show_warning_dialog(self, title, message):
        QMessageBox.information(self, title, message)

    def init_ui(self):
        main_layout = QHBoxLayout(self);
        main_layout.setContentsMargins(15, 15, 15, 15);
        main_layout.setSpacing(15)

        sidebar = QWidget();
        sidebar.setFixedWidth(300);
        side_layout = QVBoxLayout(sidebar);
        side_layout.setContentsMargins(0, 0, 0, 0)

        grp_file = QGroupBox("Data Source");
        grp_file.setObjectName("FlatGroupBox")
        l_file = QVBoxLayout()
        self.btn_load = QPushButton("Load Excel File (Torque_Characteristic)");
        self.btn_load.setObjectName("btn_load")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setMinimumHeight(35)
        self.lbl_file_info = QLabel("No file loaded");
        self.lbl_file_info.setStyleSheet("color: #777; font-size: 9pt;")
        self.txt_rpm = QLineEdit("0");
        self.txt_rpm.setReadOnly(True);
        self.txt_rpm.setPlaceholderText("RPM (Extracted from filename)")
        l_file.addWidget(self.btn_load);
        l_file.addWidget(self.lbl_file_info);
        l_file.addWidget(QLabel("Extracted RPM:"));
        l_file.addWidget(self.txt_rpm)
        grp_file.setLayout(l_file);
        side_layout.addWidget(grp_file)

        grp_params = QGroupBox("Configuration");
        grp_params.setObjectName("FlatGroupBox")
        l_params = QVBoxLayout()
        self.radio_radial = QRadioButton("Radial Force");
        self.radio_radial.setChecked(True)
        self.radio_tan = QRadioButton("Tangential Force");
        self.bg = QButtonGroup(self);
        self.bg.addButton(self.radio_radial);
        self.bg.addButton(self.radio_tan)
        self.bg.buttonToggled.connect(self._handle_visual_update);
        l_params.addWidget(self.radio_radial);
        l_params.addWidget(self.radio_tan);
        l_params.addSpacing(15)

        self.chk_exclude_zero = QCheckBox("Exclude Temporal Order 0 (DC)");
        self.chk_exclude_zero.setChecked(True)
        self.chk_exclude_zero.stateChanged.connect(self._handle_visual_update);
        l_params.addWidget(self.chk_exclude_zero);
        l_params.addSpacing(15)

        l_params.addWidget(QLabel("Plot Scale:"));
        self.combo_scale = QComboBox();
        self.combo_scale.addItems(["Linear", "Logarithmic (dB)"])
        self.combo_scale.currentIndexChanged.connect(self._handle_visual_update);
        l_params.addWidget(self.combo_scale)
        l_params.addWidget(QLabel("Min Amp / dB:"));
        self.txt_min = QLineEdit("0");
        l_params.addWidget(self.txt_min)
        l_params.addWidget(QLabel("Max Amp / dB:"));
        self.txt_max = QLineEdit("0");
        l_params.addWidget(self.txt_max)
        self.btn_apply = QPushButton("Refresh Visuals");
        self.btn_apply.clicked.connect(self._handle_visual_update);
        l_params.addWidget(self.btn_apply)
        grp_params.setLayout(l_params);
        side_layout.addWidget(grp_params)

        grp_order_forcing = QGroupBox("Order Forcing Excel Export")
        grp_order_forcing.setObjectName("FlatGroupBox")
        form_layout_export = QFormLayout(grp_order_forcing)

        self.txt_temp_orders = QLineEdit("8");
        self.txt_spatial_orders = QLineEdit("-8, 16")
        self.btn_process_forcing = QPushButton("Process Order Export");
        self.btn_process_forcing.setObjectName("OrderForcingProcessButton")
        self.btn_process_forcing.clicked.connect(self.process_order_forcing)

        form_layout_export.addRow("Temporal Orders (k):", self.txt_temp_orders)
        form_layout_export.addRow("Spatial Orders (m):", self.txt_spatial_orders)
        form_layout_export.addRow(self.btn_process_forcing)

        side_layout.addWidget(grp_order_forcing)

        self.btn_optistruct = QPushButton("Generate Optistruct Files")
        self.btn_optistruct.setObjectName("OptistructButton")
        self.btn_optistruct.clicked.connect(self.open_optistruct_dialog);
        self.btn_optistruct.setMinimumHeight(45)
        side_layout.addWidget(self.btn_optistruct)

        self.progress = QProgressBar();
        self.progress.setVisible(False)
        side_layout.addStretch();
        side_layout.addWidget(self.progress)
        main_layout.addWidget(sidebar)

        self.tabs = QTabWidget();
        main_layout.addWidget(self.tabs, stretch=1)

        self.tab_plot = QWidget();
        plot_layout = QVBoxLayout(self.tab_plot)
        self.fig = Figure(figsize=(10, 8), dpi=100);
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig);
        self.toolbar = NavigationToolbar2QT(self.canvas, self.tab_plot)
        plot_layout.addWidget(self.toolbar);
        plot_layout.addWidget(self.canvas)
        self.tabs.addTab(self.tab_plot, "Spectrogram Analysis")

        self.tab_data = QWidget();
        data_layout = QVBoxLayout(self.tab_data)
        h_ctrl = QHBoxLayout();
        h_ctrl.addWidget(QLabel("Temporal Step:"))
        self.combo_step = QComboBox();
        self.combo_step.addItems([str(i) for i in range(1, 11)])
        self.combo_step.currentTextChanged.connect(self._handle_visual_update);
        h_ctrl.addWidget(self.combo_step)
        h_ctrl.addStretch();
        data_layout.addLayout(h_ctrl)

        self.sub_tabs = QTabWidget()
        self.cont_mag = QWidget();
        l_mag = QVBoxLayout(self.cont_mag);
        self.table_mag = QTableView();
        l_mag.addWidget(self.table_mag)
        self.sub_tabs.addTab(self.cont_mag, "Magnitude Matrix (N)")
        self.cont_ph = QWidget();
        l_ph = QVBoxLayout(self.cont_ph);
        self.table_ph = QTableView();
        l_ph.addWidget(self.table_ph)
        self.sub_tabs.addTab(self.cont_ph, "Phase Matrix (Rad)")
        data_layout.addWidget(self.sub_tabs);
        self.tabs.addTab(self.tab_data, "Harmonic Order Data")

    def set_theme(self, is_dark_mode):
        theme_color = '#2D2D30' if is_dark_mode else 'white'
        self.fig.patch.set_facecolor(theme_color)
        self.fig.tight_layout()
        self.canvas.draw()

    def extract_rpm_from_filename(self, filename):
        match = re.search(r'(\d+)rpm', os.path.basename(filename), re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def load_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open Excel", "", "Excel (*.xlsx *.xls)")
        if not f: return
        self.lbl_file_info.setText("Loading...");
        self.progress.setVisible(True);
        self.btn_load.setEnabled(False)
        self.df = None
        try:
            self.df = self.analyzer.load_excel(f)
            rpm_val = self.extract_rpm_from_filename(f)
            self.txt_rpm.setText(str(rpm_val))
            self.lbl_file_info.setText(f"File: {f.split('/')[-1]}")
            self.request_calc_dual(rpm_val)
        except Exception as e:
            self.progress.setVisible(False);
            self.btn_load.setEnabled(True)
            QMessageBox.critical(self, "File Load Error",
                                 f"An error occurred while loading or parsing the file:\n{str(e)}")
            self.lbl_file_info.setText("Load failed.")

    def request_calc_dual(self, rpm):
        if self.df is None or rpm == 0: return
        if self.worker and self.worker.isRunning(): self.worker.terminate(); self.worker.wait()

        self.progress.setVisible(True);
        self.btn_load.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.worker = DualFFTWorker(self.analyzer, self.df, rpm)
        self.worker.finished.connect(self.on_calc_done)
        self.worker.start()

    def on_calc_done(self, success, msg):
        self.progress.setVisible(False);
        self.btn_load.setEnabled(True);
        self.btn_apply.setEnabled(True)
        if not success:
            QMessageBox.critical(self, "Calculation Error",
                                 f"FFT calculation failed:\n{msg}\n\nCheck if the correct force columns exist in the file.")
            return
        self._handle_visual_update()

    def process_order_forcing(self):
        if self.analyzer.radial_magnitude is None or self.analyzer.tangential_magnitude is None:
            QMessageBox.warning(self, "Warning",
                                "Please load and process a file first before exporting order data.")
            return

        try:
            t_str = self.txt_temp_orders.text().split(',')
            temporal_orders = [float(x.strip()) for x in t_str if x.strip()]
            s_str = self.txt_spatial_orders.text().split(',')
            spatial_orders = [float(x.strip()) for x in s_str if x.strip()]
            if not temporal_orders or not spatial_orders:
                raise ValueError("Temporal and Spatial order lists cannot be empty.")
            temporal_orders = [int(o) for o in temporal_orders]
            spatial_orders = [int(o) for o in spatial_orders]
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid list format:\n{e}")
            return

        rpm_text = self.txt_rpm.text()
        try:
            rpm = int(float(rpm_text)) if rpm_text else 1000
        except:
            rpm = 1000

        output_folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save Order Export Files",
                                                         self.output_folder)
        if not output_folder: return
        self.output_folder = output_folder

        self.progress.setVisible(True)
        try:
            success, msg = self.analyzer.export_order_data_to_excel(output_folder, temporal_orders, spatial_orders, rpm)
            QMessageBox.information(self, "Export Success" if success else "Export Error", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error: {e}")
        finally:
            self.progress.setVisible(False)

    def open_optistruct_dialog(self):
        if self.analyzer.radial_magnitude is None:
            QMessageBox.warning(self, "Warning", "Please load and analyze a file first.");
            return

        default_orders = [o for o in self.analyzer.temporal_order if o > 0]
        dlg = OptiStructDialog(self, total_teeth=self.analyzer.total_teeth, default_orders=default_orders)
        dlg.exec()

    def _handle_visual_update(self):
        force_type = "radial" if self.radio_radial.isChecked() else "tangential"
        if getattr(self.analyzer, f"{force_type}_magnitude") is None: return

        try:
            exclude_zero = self.chk_exclude_zero.isChecked()
            dense = self.analyzer.get_dense_data(force_type, exclude_zero)
            if not dense: return
            t_range, s_range, d_mag, d_ph = dense
            self.update_plots(t_range, s_range, d_mag, d_ph)
            self.update_tables(t_range, s_range, d_mag, d_ph)
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to refresh plots/tables:\n{e}")

    def update_plots(self, t_range, s_range, d_mag, d_ph):
        if d_mag is None or t_range.size == 0 or s_range.size == 0: self.fig.clear(); self.canvas.draw(); return

        matlab_colors_list = [(1, 1, 1), (0.95477, 0.95477, 1), (0.90954, 0.90954, 1), (0.86432, 0.86432, 1),
                              (0.81909, 0.81909, 1), (0.77386, 0.77386, 1), (0.72864, 0.72864, 1),
                              (0.68341, 0.68341, 1), (0.63819, 0.63819, 1), (0.59296, 0.59296, 1),
                              (0.54773, 0.54773, 1), (0.50251, 0.50251, 1), (0.45728, 0.45728, 1),
                              (0.41206, 0.41206, 1), (0.36683, 0.36683, 1), (0.32160, 0.32160, 1),
                              (0.27638, 0.27638, 1), (0.23115, 0.23115, 1), (0.18592, 0.18592, 1),
                              (0.14070, 0.14070, 1), (0.09547, 0.09547, 1), (0.05025, 0.05025, 1),
                              (0.01822, 0.02261, 1), (0, 0.01340, 1), (0, 0.02847, 1), (0, 0.04355, 1), (0, 0.05862, 1),
                              (0, 0.07370, 1), (0, 0.08877, 1), (0, 0.10385, 1), (0, 0.11892, 1), (0, 0.13400, 1),
                              (0, 0.14907, 1), (0, 0.16415, 1), (0, 0.17922, 1), (0, 0.19430, 1), (0, 0.20938, 1),
                              (0, 0.22445, 1), (0, 0.23953, 1), (0, 0.25460, 1), (0, 0.26968, 1), (0, 0.28475, 1),
                              (0, 0.29983, 1), (0, 0.31490, 1), (0, 0.32998, 1), (0, 0.34505, 1), (0, 0.36013, 1),
                              (0, 0.37520, 1), (0, 0.39028, 1), (0, 0.40536, 1), (0, 0.42043, 1), (0, 0.43551, 1),
                              (0, 0.45058, 1), (0, 0.46566, 1), (0, 0.48073, 1), (0, 0.49581, 1), (0, 0.51088, 1),
                              (0, 0.52596, 1), (0, 0.54103, 1), (0, 0.55611, 1), (0, 0.57118, 1), (0, 0.58626, 1),
                              (0, 0.60134, 1), (0, 0.61641, 1), (0, 0.63149, 1), (0, 0.64656, 1), (0, 0.66164, 1),
                              (0, 0.67671, 1), (0, 0.69179, 1), (0, 0.70686, 1), (0, 0.72194, 1), (0, 0.73701, 1),
                              (0, 0.75209, 1), (0, 0.76716, 1), (0, 0.78224, 1), (0, 0.79731, 1), (0, 0.81239, 1),
                              (0, 0.82747, 1), (0, 0.84254, 1), (0, 0.85762, 1), (0, 0.87269, 1), (0, 0.88777, 1),
                              (0, 0.90284, 1), (0, 0.91792, 1), (0, 0.93299, 1), (0, 0.94807, 1), (0, 0.96314, 1),
                              (0, 0.97822, 1), (0.00224, 0.99105, 0.99775), (0.00837, 1, 0.99162),
                              (0.02345, 1, 0.97654), (0.03852, 1, 0.96147), (0.05360, 1, 0.94639),
                              (0.06867, 1, 0.93132), (0.08375, 1, 0.91624), (0.09882, 1, 0.90117),
                              (0.11390, 1, 0.88609), (0.12897, 1, 0.87102), (0.14405, 1, 0.85594),
                              (0.15912, 1, 0.84087), (0.17420, 1, 0.82579), (0.18927, 1, 0.81072),
                              (0.20435, 1, 0.79564), (0.21943, 1, 0.78056), (0.23450, 1, 0.76549),
                              (0.24958, 1, 0.75041), (0.26465, 1, 0.73534), (0.27973, 1, 0.72026),
                              (0.29480, 1, 0.70519), (0.30988, 1, 0.69011), (0.32495, 1, 0.67504),
                              (0.34003, 1, 0.65996), (0.35510, 1, 0.64489), (0.37018, 1, 0.62981),
                              (0.38525, 1, 0.61474), (0.40033, 1, 0.59966), (0.41541, 1, 0.58458),
                              (0.43048, 1, 0.56951), (0.44556, 1, 0.55443), (0.46063, 1, 0.53936),
                              (0.47571, 1, 0.52428), (0.49078, 1, 0.50921), (0.50586, 1, 0.49413),
                              (0.52093, 1, 0.47906), (0.53601, 1, 0.46398), (0.55108, 1, 0.44891),
                              (0.56616, 1, 0.43383), (0.58123, 1, 0.41876), (0.59631, 1, 0.40368),
                              (0.61139, 1, 0.38860), (0.62646, 1, 0.37353), (0.64154, 1, 0.35845),
                              (0.65661, 1, 0.34338), (0.67169, 1, 0.32830), (0.68676, 1, 0.31323),
                              (0.70184, 1, 0.29815), (0.71691, 1, 0.28308), (0.73199, 1, 0.26800),
                              (0.74706, 1, 0.25293), (0.76214, 1, 0.23785), (0.77721, 1, 0.22278),
                              (0.79229, 1, 0.20770), (0.80737, 1, 0.19262), (0.82244, 1, 0.17755),
                              (0.83752, 1, 0.16247), (0.85259, 1, 0.14740), (0.86767, 1, 0.13232),
                              (0.88274, 1, 0.11725), (0.89782, 1, 0.10217), (0.91289, 1, 0.08710),
                              (0.92797, 1, 0.07202), (0.94304, 1, 0.05695), (0.95812, 1, 0.04187),
                              (0.97319, 1, 0.02680), (0.98827, 1, 0.01172), (0.99824, 0.99233, 0.00175),
                              (0.99991, 0.97223, 8.4e-05), (1, 0.94974, 0), (1, 0.92713, 0), (1, 0.90452, 0),
                              (1, 0.88190, 0), (1, 0.85929, 0), (1, 0.83668, 0), (1, 0.81407, 0), (1, 0.79145, 0),
                              (1, 0.76884, 0), (1, 0.74623, 0), (1, 0.72361, 0), (1, 0.70100, 0), (1, 0.67839, 0),
                              (1, 0.65577, 0), (1, 0.63316, 0), (1, 0.61055, 0), (1, 0.58793, 0), (1, 0.56532, 0),
                              (1, 0.54271, 0), (1, 0.52010, 0), (1, 0.49748, 0), (1, 0.47487, 0), (1, 0.45226, 0),
                              (1, 0.42964, 0), (1, 0.40703, 0), (1, 0.38442, 0), (1, 0.36180, 0), (1, 0.33919, 0),
                              (1, 0.31658, 0), (1, 0.29396, 0), (1, 0.27135, 0), (1, 0.24874, 0), (1, 0.22613, 0),
                              (1, 0.20351, 0), (1, 0.18090, 0), (1, 0.15829, 0), (1, 0.13567, 0), (1, 0.11306, 0),
                              (1, 0.09045, 0), (1, 0.06783, 0), (1, 0.04522, 0), (1, 0.02261, 0), (1, 0, 0)]
        matlab_cmap = ListedColormap(matlab_colors_list)

        is_log = "Log" in self.combo_scale.currentText()
        if is_log:
            pm = d_mag.copy();
            pm[pm <= 0] = 1e-9;
            plot_data = 20 * np.log10(pm)
            cbar_lbl = "Force (dB)"
        else:
            plot_data = d_mag;
            cbar_lbl = "Force (N)"

        teeth_limit = self.analyzer.total_teeth

        try:
            max_t_idx = np.where(t_range <= teeth_limit)[0][-1] + 1
        except IndexError:
            max_t_idx = len(t_range)

        bar_x_positions = t_range[:max_t_idx]
        bar_btm = np.max(plot_data[:, :max_t_idx], axis=0)
        bar_left = np.max(plot_data, axis=1)
        bar_y_positions = s_range

        self.fig.clear()
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 4, 0.15], height_ratios=[4, 1], wspace=0.06, hspace=0.06)
        ax_left = self.fig.add_subplot(gs[0, 0])
        ax_map = self.fig.add_subplot(gs[0, 1], sharey=ax_left)
        cax = self.fig.add_subplot(gs[0, 2])
        ax_btm = self.fig.add_subplot(gs[1, 1], sharex=ax_map)

        def clean_ax(ax):
            ax.spines['top'].set_visible(False);
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.5, color='#aaa');
            ax.tick_params(colors='#444')

        clean_ax(ax_left);
        clean_ax(ax_btm)
        bar_width = 1.0

        x_min_data = t_range[0] - bar_width / 2
        x_max_data = t_range[-1] + bar_width / 2
        y_min = s_range.min() - bar_width / 2
        y_max = s_range.max() + bar_width / 2
        extent = [x_min_data, x_max_data, y_min, y_max]

        try:
            vmin, vmax = float(self.txt_min.text()), float(self.txt_max.text())
        except:
            vmin, vmax = 0, 0

        im = ax_map.imshow(plot_data, aspect='auto', origin='lower', extent=extent, cmap=matlab_cmap,
                           interpolation='nearest')
        if vmax > vmin: im.set_clim(vmin, vmax)
        cbar = self.fig.colorbar(im, cax=cax, fraction=1.0)
        cbar.set_label(cbar_lbl, rotation=270, labelpad=15);
        cbar.outline.set_visible(False)

        ax_left.barh(bar_y_positions, bar_left, color='#1f77b4', height=bar_width * 0.8)
        ax_left.invert_xaxis();
        ax_left.set_ylabel("Spatial Order", fontweight='bold');
        ax_left.set_xlabel(cbar_lbl)

        ax_btm.bar(bar_x_positions, bar_btm, color='#ff7f0e', width=bar_width * 0.8)
        ax_btm.set_xlabel("Temporal Order", fontweight='bold');
        ax_btm.set_ylabel(cbar_lbl)

        x_limit_max = teeth_limit + 0.5
        x_limit_min = t_range[0] - bar_width / 2

        if x_limit_max > x_limit_min: ax_map.set_xlim(x_limit_min, x_limit_max); ax_btm.set_xlim(x_limit_min,
                                                                                                 x_limit_max)
        ax_map.set_ylim(y_min, y_max);
        ax_left.set_ylim(y_min, y_max)

        interval_temp = 4 if teeth_limit == 48 else (3 if teeth_limit == 54 else 5)
        ax_btm.xaxis.set_major_locator(ticker.MultipleLocator(interval_temp));
        ax_map.xaxis.set_major_locator(ticker.MultipleLocator(interval_temp))

        interval_spatial = 4 if teeth_limit == 48 else (3 if teeth_limit == 54 else 5)
        ax_left.yaxis.set_major_locator(ticker.MultipleLocator(interval_spatial));
        ax_map.yaxis.set_major_locator(ticker.MultipleLocator(interval_spatial))

        ax_map.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_tables(self, t_range, s_range, d_mag, d_ph):
        if d_mag is None: return
        try:
            step = int(self.combo_step.currentText())
        except:
            step = 1

        if t_range.size == 0:
            self.table_mag.setModel(ScientificTableModel(np.array([[]]), np.array([]), np.array([])))
            self.table_ph.setModel(ScientificTableModel(np.array([[]]), np.array([]), np.array([])))
            return

        idx = np.arange(0, len(t_range), step)
        t_cols = t_range[idx];
        sub_mag = d_mag[:, idx];
        sub_ph = d_ph[:, idx]

        self.table_mag.setModel(ScientificTableModel(sub_mag, s_range, t_cols, 10.0))
        self.table_ph.setModel(ScientificTableModel(sub_ph, s_range, t_cols))

        for tv in [self.table_mag, self.table_ph]:
            h = tv.horizontalHeader();
            h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            h.setMinimumSectionSize(50)

class OptiStructDialog(QDialog):
    TEMPLATE_CONTENT = """$
$ NASTRAN INPUT DECK GENERATED BY PYQT NVH TOOL
$
SOL 108
CEND
TITLE = ELECTRIC MOTOR NVH
ECHO = NONE
$
$ [ SUBCASE HEAD ADDED THROUGH SCRIPT START
$ ]SUBCASE HEAD ADDED THROUGH SCRIPT END
$
BEGIN BULK
$
$ [ DAREA HEAD ADDED THROUGH SCRIPT START
$ ]DAREA HEAD ADDED THROUGH SCRIPT END
$
ENDDATA
"""

    @staticmethod
    def generate(output_folder, order_lst, waveform_lst, position, total_teeth):
        try:
            os.makedirs(output_folder, exist_ok=True)

            # Clean header comment - every line starts with $
            chr_str = "$ Temporal Order (k) : " + ' '.join(map(str, order_lst))
            chr_str += "\n$ Spatial Order (m) : " + ' '.join(map(str, waveform_lst))
            chr_str += "\n$"

            dchar_str = ""

            force_lst = ["RAD", "TAN"]
            component_lst = ["STATOR", "ROTOR"]
            Nteeth = total_teeth

            for order_no in order_lst:
                order_int = int(round(float(order_no)))

                for wf_val in waveform_lst:
                    wf_val_int = int(round(float(wf_val)))
                    waveform_abs = abs(wf_val_int)
                    wf_id = "2" if wf_val_int < 0 else "1"

                    # Spatial order sequence
                    if wf_val_int < 0:
                        space_order = np.concatenate(([1], np.arange(Nteeth, 1, -1)))
                    else:
                        space_order = np.arange(1, Nteeth + 1)

                    # Phase calculation
                    if waveform_abs == 24:
                        form = 360.0 * (waveform_abs / Nteeth) * np.arange(1, Nteeth + 1) + 90.0
                        form[0] = 90.0
                    else:
                        form = 360.0 * (waveform_abs / Nteeth) * np.arange(1, Nteeth + 1)
                        form[0] = 0.0

                    # Separate DLOAD IDs for stator and rotor
                    subcase_base = f"1{wf_id}{waveform_abs:02d}{order_int:02d}"
                    dload_stator = f"1102{subcase_base}"
                    dload_rotor = f"1103{subcase_base}"

                    label_text = f"Order {order_int} Waveform {wf_val_int}"

                    # SUBCASE for STATOR
                    chr_str += f"\nSUBCASE {subcase_base}"
                    chr_str += f"\nLABEL={label_text} STATOR"
                    chr_str += "\nMETHOD=1"
                    chr_str += "\nFREQ=400"
                    chr_str += f"\nDLOAD={dload_stator}"
                    chr_str += "\n$"

                    # SUBCASE for ROTOR
                    chr_str += f"\nSUBCASE {subcase_base}R"
                    chr_str += f"\nLABEL={label_text} ROTOR"
                    chr_str += "\nMETHOD=1"
                    chr_str += "\nFREQ=400"
                    chr_str += f"\nDLOAD={dload_rotor}"
                    chr_str += "\n$"

                    grid_prefix = "7" if position == "Rear" else "1"

                    # Separate RLOAD lists for stator and rotor
                    rload_stator = []
                    rload_rotor = []

                    for force_name in force_lst:
                        force_id = "1" if force_name == "RAD" else "2"

                        for component in component_lst:
                            stat_rot_id1 = "1" if component == "STATOR" else "2"
                            stat_rot_id_suffix = "0001" if component == "STATOR" else "0005"
                            darea_insert = "1.0" if component == "STATOR" else "-1.0"

                            current_rload_list = rload_stator if component == "STATOR" else rload_rotor

                            dchar_str += "\n$"
                            dchar_str += "\n$ ----------------------------"
                            dchar_str += f"\n$ DAREA {force_name} WF{wf_val_int} {component}"
                            dchar_str += "\n$ ----------------------------"

                            for k, space_k in enumerate(space_order):
                                darea_id = f"{stat_rot_id1}{force_id}{wf_id}{waveform_abs:02d}1{space_k:04d}"
                                grid_final = f"{grid_prefix}{stat_rot_id_suffix}{space_k:02d}"

                                dchar_str += f"\n$ T{space_k}_{force_name}_{component}"
                                dchar_str += f"\nDAREA, {darea_id}, {grid_final}, {force_id}, {darea_insert}"

                                rload_id = f"{stat_rot_id1}{force_id}{wf_id}{waveform_abs:02d}2{space_k:04d}"
                                phase_val = form[k]

                                if abs(phase_val) < 1e-6:
                                    dchar_str += f"\nRLOAD2, {rload_id}, {darea_id},,, 110021"
                                else:
                                    dchar_str += f"\nRLOAD2, {rload_id}, {darea_id},,{phase_val:8.1f}, 110021"

                                current_rload_list.append(rload_id)

                            dchar_str += "\n$"

                    # DLOAD for STATOR (RAD + TAN)
                    if rload_stator:
                        dchar_str += "\n$"
                        dchar_str += "\n$ Combined STATOR loads (RAD + TAN)"

                        line = f"DLOAD,{dload_stator},1.0"
                        count = 0
                        for i in range(min(3, len(rload_stator))):
                            line += f",1.0,{rload_stator[i]}"
                            count += 1
                        dchar_str += f"\n{line}"

                        remaining = rload_stator[count:]
                        for i in range(0, len(remaining), 4):
                            chunk = remaining[i:i+4]
                            cont = "+"
                            for r in chunk:
                                cont += f",1.0,{r}"
                            dchar_str += f"\n{cont}"

                        dchar_str += "\n$"

                    # DLOAD for ROTOR (RAD + TAN)
                    if rload_rotor:
                        dchar_str += "\n$"
                        dchar_str += "\n$ Combined ROTOR loads (RAD + TAN)"

                        line = f"DLOAD,{dload_rotor},1.0"
                        count = 0
                        for i in range(min(3, len(rload_rotor))):
                            line += f",1.0,{rload_rotor[i]}"
                            count += 1
                        dchar_str += f"\n{line}"

                        remaining = rload_rotor[count:]
                        for i in range(0, len(remaining), 4):
                            chunk = remaining[i:i+4]
                            cont = "+"
                            for r in chunk:
                                cont += f",1.0,{r}"
                            dchar_str += f"\n{cont}"

                        dchar_str += "\n$"

            # Insert blocks - every marker line starts with $
            final_content = OptiStructDialog.TEMPLATE_CONTENT
            final_content = re.sub(
                r'\$ \[ SUBCASE HEAD ADDED THROUGH SCRIPT START.*?\$ \]SUBCASE HEAD ADDED THROUGH SCRIPT END',
                f'$ [ SUBCASE HEAD ADDED THROUGH SCRIPT START\n{chr_str}\n$ ]SUBCASE HEAD ADDED THROUGH SCRIPT END',
                final_content, flags=re.DOTALL)
            final_content = re.sub(
                r'\$ \[ DAREA HEAD ADDED THROUGH SCRIPT START.*?\$ \]DAREA HEAD ADDED THROUGH SCRIPT END',
                f'$ [ DAREA HEAD ADDED THROUGH SCRIPT START\n{dchar_str}\n$ ]DAREA HEAD ADDED THROUGH SCRIPT END',
                final_content, flags=re.DOTALL)

            # Save one deck per temporal order
            last_path = ""
            for order_no in order_lst:
                order_int = int(round(float(order_no)))
                filename = f"EDU_Autosim_createdDeck_Order_{order_int}.fem"
                full_path = os.path.join(output_folder, filename)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                last_path = full_path

            return True, last_path

        except Exception as e:
            return False, str(e)

    def __init__(self, parent=None, total_teeth=48, default_orders=None):
        super().__init__(parent)
        self.setWindowTitle("OptiStruct Deck Generation")
        self.resize(500, 400)
        self.total_teeth = total_teeth
        self.default_orders = default_orders if default_orders is not None else []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.tab_basic = QWidget()
        self.tab_adv = QWidget()
        tabs.addTab(self.tab_basic, "Basic")
        tabs.addTab(self.tab_adv, "Advanced")

        layout_basic = QFormLayout(self.tab_basic)
        info_lbl = QLabel(f"Total Teeth Detected: {self.total_teeth}")
        info_lbl.setStyleSheet("font-weight: bold; color: #004D8A;")
        layout_basic.addRow(info_lbl)

        self.radio_rear = QRadioButton("Rear")
        self.radio_rear.setChecked(True)
        self.radio_front = QRadioButton("Front")
        h_pos = QHBoxLayout()
        h_pos.addWidget(self.radio_rear)
        h_pos.addWidget(self.radio_front)
        self.grp_pos = QButtonGroup(self)
        self.grp_pos.addButton(self.radio_rear)
        self.grp_pos.addButton(self.radio_front)
        layout_basic.addRow("Motor Position:", h_pos)

        layout_adv = QFormLayout(self.tab_adv)
        self.txt_orders = QLineEdit()
        default_order = next((o for o in self.default_orders if o > 0), 8)
        self.txt_orders.setText(str(int(default_order)))
        self.txt_waveforms = QLineEdit("-1")
        layout_adv.addRow("Temporal Order List (comma sep):", self.txt_orders)
        layout_adv.addRow("Spatial Order List (comma sep):", self.txt_waveforms)

        layout.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.generate_files)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def generate_files(self):
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Save OptiStruct Deck(s)",
            os.path.expanduser("~")
        )
        if not output_folder:
            return

        position = "Rear" if self.radio_rear.isChecked() else "Front"

        try:
            ord_str = self.txt_orders.text().split(',')
            order_list = [float(x.strip()) for x in ord_str if x.strip()]
            wf_str = self.txt_waveforms.text().split(',')
            wf_list = [float(x.strip()) for x in wf_str if x.strip()]
            if not order_list or not wf_list:
                raise ValueError("Both lists must be non-empty.")
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Invalid list format:\n{e}")
            return

        success, msg = OptiStructDialog.generate(output_folder, order_list, wf_list, position, self.total_teeth)
        if success:
            QMessageBox.information(self, "Success", f"OptiStruct deck(s) generated successfully!\nSaved to:\n{msg}")
            self.accept()
        else:
            QMessageBox.critical(self, "Generation Error", f"Failed to create deck:\n{msg}")






# ----------------------------------------------------------------------
# --- TAB 1: TF Generator (PCH Converter Logic) ---
# ----------------------------------------------------------------------

class PchParser:
    MIC_MAP = {
        7008110: "Top Mic",
        7008114: "Bottom Mic",
        7008118: "Left Mic",
        7008122: "Right Mic",
        7008126: "Front Mic",
        7008130: "Rear Mic",
    }

    def __init__(self, pch_filepath):
        self.pch_filepath = pch_filepath
        self.all_data_list = []

    def _parse_frequency_block(self, block_lines):
        metadata = {}
        data_rows = []
        data_started = False

        for line in block_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('$'):
                if '$SUBCASE ID' in line:
                    metadata['SUBCASE_ID'] = line.split('=')[-1].strip()
                elif '$LABEL' in line:
                    label_raw = line.split('=')[-1].strip()
                    label_parts = label_raw.split()
                    label_clean = label_parts[0] if label_parts else 'UNKNOWN'
                    metadata['LABEL'] = label_clean.upper()
                elif '$FREQUENCY' in line:
                    freq_str = line.split('=')[-1].strip()
                    try:
                        metadata['FREQUENCY'] = float(freq_str)
                    except ValueError:
                        metadata['FREQUENCY'] = 0.0
                    data_started = True

            elif data_started and not line.startswith('+'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        mic_id = int(parts[0])
                        real = float(parts[2])
                        imag = float(parts[3])
                        data_rows.append({
                            'MIC_ID': mic_id,
                            'REAL': real,
                            'IMAGINARY': imag
                        })
                    except ValueError:
                        continue

        return metadata, data_rows

    def parse_file(self, log_signal):
        log_signal.emit(f"Parsing file: {self.pch_filepath}...")

        try:
            with open(self.pch_filepath, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            log_signal.emit(f"ERROR: Could not read file. {e}")
            return False

        block_lines = []
        parsed_blocks = 0

        for line in lines:
            if line.startswith('$TITLE'):
                if block_lines:
                    metadata, data_rows = self._parse_frequency_block(block_lines)
                    if metadata and 'FREQUENCY' in metadata and data_rows:
                        self._process_data(metadata, data_rows, log_signal)
                        parsed_blocks += 1
                block_lines = [line]
            else:
                block_lines.append(line)

        if block_lines:
            metadata, data_rows = self._parse_frequency_block(block_lines)
            if metadata and 'FREQUENCY' in metadata and data_rows:
                self._process_data(metadata, data_rows, log_signal)
                parsed_blocks += 1

        log_signal.emit(f"File parsed. Total frequency blocks processed: {parsed_blocks}")
        return True

    def _process_data(self, metadata, data_rows, log_signal):
        freq = metadata['FREQUENCY']
        subcase = metadata.get('SUBCASE_ID', 'N/A').strip()
        label = metadata.get('LABEL', 'N/A').strip()
        column_id = f"Subcase {subcase} ({label})"

        for row in data_rows:
            mic_id = row['MIC_ID']
            real = row['REAL']
            imag = row['IMAGINARY']

            magnitude = math.sqrt(real ** 2 + imag ** 2)
            phase_deg = np.degrees(np.arctan2(imag, real))

            mic_name = self.MIC_MAP.get(mic_id)
            if mic_name:
                self.all_data_list.append({
                    'MicName': mic_name,
                    'Frequency': freq,
                    'ColumnID': column_id,
                    'Magnitude': magnitude,
                    'Phase': phase_deg
                })
            else:
                log_signal.emit(f"WARNING: Unknown Mic ID {mic_id} found at Freq {freq}. Skipping data.")

    def export_to_excel(self, output_dir, log_signal):
        if not self.all_data_list:
            log_signal.emit("No data was parsed to export.")
            return

        log_signal.emit("Starting Excel export and data restructuring...")

        df_master = pd.DataFrame(self.all_data_list)
        grouped_by_mic = df_master.groupby('MicName')

        for mic_name, df_mic in grouped_by_mic:
            log_signal.emit(f"Processing data for {mic_name}...")

            df_mag = df_mic.pivot_table(
                index='Frequency',
                columns='ColumnID',
                values='Magnitude'
            ).rename(columns=lambda x: f"{x} Mag")

            df_phase = df_mic.pivot_table(
                index='Frequency',
                columns='ColumnID',
                values='Phase'
            ).rename(columns=lambda x: f"{x} Phase")

            df_combined = pd.concat([df_mag, df_phase], axis=1)
            df_combined = df_combined.sort_index()

            unique_subcase_cols = sorted(list(set(col.replace(' Mag', '').replace(' Phase', '').strip()
                                                  for col in df_combined.columns)))

            final_columns = {'Frequency (Hz)': df_combined.index.to_series()}
            final_order = ['Frequency (Hz)']

            for subcase_base in unique_subcase_cols:
                mag_col = f"{subcase_base} Mag"
                phase_col = f"{subcase_base} Phase"
                if mag_col in df_combined.columns and phase_col in df_combined.columns:
                    final_columns[mag_col] = df_combined[mag_col]
                    final_columns[f'Frequency (Hz) {subcase_base}_Phase'] = df_combined.index.to_series()
                    final_columns[phase_col] = df_combined[phase_col]
                    final_order.extend([mag_col, f'Frequency (Hz) {subcase_base}_Phase', phase_col])

            final_df = pd.DataFrame(final_columns).reindex(columns=final_order)

            output_filename = os.path.join(output_dir, f"TF__{mic_name.replace(' ', '_')}_Data.xlsx")

            try:
                writer = pd.ExcelWriter(output_filename, engine='xlsxwriter')
                final_df.to_excel(writer, sheet_name=mic_name, index=False, header=False, startrow=1)
                worksheet = writer.sheets[mic_name]

                excel_headers = []
                for col in final_df.columns:
                    if 'Frequency' in col:
                        excel_headers.append('Frequency (Hz)')
                    else:
                        excel_headers.append(col)

                for col_num, value in enumerate(excel_headers):
                    worksheet.write(0, col_num, value)

                writer.close()
                log_signal.emit(f"SUCCESS: Exported data for {mic_name} to {os.path.basename(output_filename)}")

            except Exception as e:
                log_signal.emit(f"ERROR: Failed to export {mic_name} data to Excel. {e}")


class ConverterWorker(QThread):
    finished = pyqtSignal(bool)
    log_update = pyqtSignal(str)

    def __init__(self, pch_file, output_dir):
        super().__init__()
        self.pch_file = pch_file
        self.output_dir = output_dir

    def run(self):
        if not self.pch_file.lower().endswith('.pch'):
            self.log_update.emit("ERROR: Input file is not a PCH file. Conversion halted.")
            self.finished.emit(False)
            return

        parser = PchParser(self.pch_file)

        if not parser.parse_file(self.log_update):
            self.finished.emit(False)
            return

        parser.export_to_excel(self.output_dir, self.log_update)

        self.finished.emit(True)


class TFGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.pch_file_path = ""
        self.output_folder_path = os.getcwd()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)

        self.left_panel = QWidget()
        left_vbox = QVBoxLayout(self.left_panel)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(20)

        grp_io = QGroupBox("1. Input PCH File & Output Folder")
        grp_io.setObjectName("FlatGroupBox")
        l_io = QFormLayout(grp_io)
        l_io.setContentsMargins(15, 25, 15, 15)
        l_io.setSpacing(15)

        self.pch_path_edit = QLineEdit()
        self.pch_path_edit.setReadOnly(True)
        self.pch_path_edit.setPlaceholderText("Select the input .pch file...")
        btn_input = QPushButton("Select PCH")
        btn_input.setObjectName("SecondaryButton_Browse")
        btn_input.clicked.connect(self.select_pch_file)
        h_input = QHBoxLayout()
        h_input.addWidget(self.pch_path_edit)
        h_input.addWidget(btn_input)
        l_io.addRow("Input PCH File:", h_input)

        self.output_folder_edit = QLineEdit(self.output_folder_path)
        self.output_folder_edit.setReadOnly(True)
        btn_output = QPushButton("Browse")
        btn_output.setObjectName("SecondaryButton_Browse")
        btn_output.clicked.connect(self.select_output_folder)
        h_output = QHBoxLayout()
        h_output.addWidget(self.output_folder_edit)
        h_output.addWidget(btn_output)
        l_io.addRow("Output Folder:", h_output)

        left_vbox.addWidget(grp_io)

        grp_conversion = QGroupBox("2. Execute Conversion")
        grp_conversion.setObjectName("FlatGroupBox")
        l_conv = QVBoxLayout(grp_conversion)
        l_conv.setContentsMargins(15, 25, 15, 15)
        l_conv.setSpacing(15)

        self.btn_convert = QPushButton("▶️ Convert PCH to TF Excel Files")
        self.btn_convert.setObjectName("PrimaryButton")
        self.btn_convert.setMinimumHeight(45)
        self.btn_convert.clicked.connect(self.start_conversion)
        l_conv.addWidget(self.btn_convert)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        l_conv.addWidget(self.progress)

        left_vbox.addWidget(grp_conversion)
        left_vbox.addStretch(1)
        main_layout.addWidget(self.left_panel, 1)

        log_group = QGroupBox("3. Conversion Status Log")
        log_group.setObjectName("FlatGroupBox")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 20, 10, 10)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setText("Ready to select PCH file for Transfer Function generation.")
        self.log_text.setObjectName("LogTextEdit")
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group, 1)

    def log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def select_pch_file(self):
        file_filter = "PCH Files (*.pch);;All Files (*)"
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Input PCH File", os.path.expanduser("~"), file_filter)
        if file_name:
            self.pch_file_path = file_name
            self.pch_path_edit.setText(file_name)
            self.log(f"Input PCH file selected: {os.path.basename(file_name)}")

    def select_output_folder(self):
        folder_name = QFileDialog.getExistingDirectory(self, "Select Output Folder for TF Files")
        if folder_name:
            self.output_folder_path = folder_name
            self.output_folder_edit.setText(folder_name)
            self.log(f"Output folder set: {os.path.basename(folder_name)}")

    def start_conversion(self):
        input_file = self.pch_file_path
        output_folder = self.output_folder_path

        if not os.path.exists(input_file):
            QMessageBox.critical(self, "Error", "Input PCH file not found.")
            return
        if not os.path.isdir(output_folder):
            QMessageBox.critical(self, "Error", "Output folder is not valid or accessible.")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Warning", "Conversion is already in progress.")
            return

        self.log("\n--- Starting PCH to TF Conversion ---")
        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("Conversion In Progress...")
        self.progress.setVisible(True)

        self.worker = ConverterWorker(input_file, output_folder)
        self.worker.log_update.connect(self.log)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.start()

    def conversion_finished(self, success):
        self.btn_convert.setEnabled(True)
        self.btn_convert.setText("▶️ Convert PCH to TF Excel Files")
        self.progress.setVisible(False)

        if success:
            self.log("--- Conversion Complete! ---")
            QMessageBox.information(self, "Success", f"TF files generated successfully in:\n{self.output_folder_path}")
        else:
            self.log("--- Conversion Failed! ---")
            QMessageBox.critical(self, "Conversion Error", "Conversion failed. Check the log for details.")


# ----------------------------------------------------------------------
# --- MAIN WINDOW ---
# ----------------------------------------------------------------------

class ThemeToggleButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Dark Mode");
        self.setIconSize(QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon);
        self.setObjectName("ThemeToggleButton")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed);
        self.setFixedHeight(30);
        self.setFixedWidth(130)

    def set_dark_mode(self, is_dark_mode):
        icon_path = os.path.join(os.path.dirname(__file__), 'moon.png') if is_dark_mode else os.path.join(
            os.path.dirname(__file__), 'sun.png')
        self.setIcon(QIcon(icon_path))
        self.setText("Light Mode" if is_dark_mode else "Dark Mode")
        self.setToolTip("Switch to Light Mode" if is_dark_mode else "Switch to Dark Mode")


# Replace these methods and the MainWindow setup in your existing script

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_dark_mode = False
        self.setWindowTitle("E-Machine NVH Tool")
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Main Background (Gmail Gray-Blue)
        self.central_widget = QWidget()
        self.central_widget.setObjectName("GmailMainBackground")
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(self.central_widget)

        self._setup_top_navigation()
        self.apply_theme()
        self.switch_tab(3)  # Defaulting to Plotting as requested

    def _setup_top_navigation(self):
        # 1. Top Header Row
        self.nav_bar = QFrame()
        self.nav_bar.setObjectName("GmailTopNav")
        self.nav_bar.setFixedHeight(64)
        self.nav_layout = QHBoxLayout(self.nav_bar)
        self.nav_layout.setContentsMargins(24, 0, 24, 0)
        self.nav_layout.setSpacing(8)

        # App Brand
        brand_lbl = QLabel("E-Machine NVH")
        brand_lbl.setStyleSheet("font-size: 20px; color: #1F1F1F; font-weight: 400; margin-right: 30px;")
        self.nav_layout.addWidget(brand_lbl)

        # Tabs in a Row
        self.btn_tf_gen = QPushButton("TF Generator")
        self.btn_force_gen = QPushButton("Forcing Analysis")
        self.btn_scale_sum = QPushButton("Scaling & Summation")
        self.btn_plotting = QPushButton("Plotting & Analysis")

        self.tabs_list = [self.btn_tf_gen, self.btn_force_gen, self.btn_scale_sum, self.btn_plotting]
        for btn in self.tabs_list:
            btn.setObjectName("GmailTopTab")
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            self.nav_layout.addWidget(btn)

        self.btn_tf_gen.clicked.connect(lambda: self.switch_tab(0))
        self.btn_force_gen.clicked.connect(lambda: self.switch_tab(1))
        self.btn_scale_sum.clicked.connect(lambda: self.switch_tab(2))
        self.btn_plotting.clicked.connect(lambda: self.switch_tab(3))

        self.nav_layout.addStretch(1)
        self.main_layout.addWidget(self.nav_bar)

        # 2. Main Content Area (Rounded White Container)
        self.content_container = QFrame()
        self.content_container.setObjectName("GmailContentContainer")
        self.container_layout = QVBoxLayout(self.content_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)

        self.content_stack = QStackedWidget()
        self.container_layout.addWidget(self.content_stack)

        self.tf_gen_tab = TFGeneratorApp()
        self.force_gen_tab = NVH_Forcing_App()
        self.scale_sum_tab = TransferFunctionApp()
        self.plotting_tab = PostProcessingWidget(self)

        self.content_stack.addWidget(self.tf_gen_tab)
        self.content_stack.addWidget(self.force_gen_tab)
        self.content_stack.addWidget(self.scale_sum_tab)
        self.content_stack.addWidget(self.plotting_tab)

        # Add some margin around the white container so the gray background shows
        wrapper_layout = QVBoxLayout()
        wrapper_layout.setContentsMargins(12, 0, 12, 12)
        wrapper_layout.addWidget(self.content_container)
        self.main_layout.addLayout(wrapper_layout)

    def switch_tab(self, index):
        self.content_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tabs_list):
            btn.setChecked(i == index)
            btn.setProperty("active", "true" if i == index else "false")
            btn.style().polish(btn)

    def apply_theme(self):
        self.setStyleSheet(self._get_gmail_light_style())

    def _get_gmail_light_style(self):
        """
        Returns the CSS style sheet for the Gmail Light (Material 3) theme.
        Features top-navigation tabs and a streamlined, compact right sidebar.
        """
        # Gmail Color Palette
        G_BG = "#F6F8FC"  # Light blue-gray main background
        G_SURFACE = "#FFFFFF"  # Pure white for content card
        G_HOVER = "#E1E3E1"  # Grey hover state
        G_ACTIVE_BG = "#C2E7FF"  # Light blue for active tabs
        G_ACTIVE_TEXT = "#001D35"  # Dark blue for active text
        G_TEXT = "#444746"  # Standard Gmail text grey
        G_PRIMARY = "#0B57D0"  # Google Primary Blue
        G_BORDER = "#E0E0E0"  # Subtle border grey

        return f"""
        /* Global Defaults */
        * {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-size: 12px;
            color: {G_TEXT};
        }}

        QWidget#GmailMainBackground {{
            background-color: {G_BG};
        }}

        /* Top Navigation Styling */
        QFrame#GmailTopNav {{
            background-color: {G_BG};
            border: none;
        }}

        QPushButton#GmailTopTab {{
            border: none;
            background-color: transparent;
            border-radius: 20px;
            padding: 0 18px;
            height: 40px;
            color: {G_TEXT};
            font-weight: 500;
        }}

        QPushButton#GmailTopTab:hover {{
            background-color: {G_HOVER};
        }}

        QPushButton#GmailTopTab[active="true"] {{
            background-color: {G_ACTIVE_BG};
            color: {G_ACTIVE_TEXT};
            font-weight: 600;
        }}

        /* Main Content Card */
        QFrame#GmailContentContainer {{
            background-color: {G_SURFACE};
            border-radius: 28px;
            border: 1px solid {G_BORDER};
        }}

        /* Streamlined Right Sidebar */
        QWidget#PlotOptionsSidebar {{
            background-color: {G_SURFACE};
            border-left: 1px solid {G_BORDER};
            border-top-right-radius: 28px;
            border-bottom-right-radius: 28px;
        }}

        QScrollArea#ControlScrollArea {{
            background-color: transparent;
        }}

        /* Compact GroupBoxes for Narrow Sidebar */
        QGroupBox#CompactGroupBox {{
            border: 1px solid #F0F0F0;
            border-radius: 12px;
            margin-top: 10px;
            font-size: 11px;
            font-weight: bold;
            color: #717171;
            background-color: {G_SURFACE};
        }}

        QGroupBox#CompactGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 3px;
        }}

        /* Gmail Pill-Style Buttons */
        #PrimaryButton {{
            background-color: {G_PRIMARY};
            color: white;
            border-radius: 20px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
        }}

        #PrimaryButton:hover {{
            background-color: #0842A0;
        }}

        #SecondaryButton_Browse, #SecondaryButton_Add {{
            background-color: {G_SURFACE};
            border: 1px solid #DADCE0;
            border-radius: 18px;
            padding: 5px 14px;
            color: {G_PRIMARY};
            font-weight: 500;
        }}

        /* Input Controls */
        QLineEdit {{
            background-color: {G_SURFACE};
            border: 1px solid #DADCE0;
            border-radius: 4px;
            padding: 6px;
            font-size: 11px;
        }}

        QLineEdit:focus {{
            border: 2px solid {G_PRIMARY};
        }}

        QCheckBox {{
            font-size: 11px;
            spacing: 8px;
        }}

        /* Compact List/Table Elements */
        QListWidget#PlotListWidget {{
            border: 1px solid #F0F0F0;
            border-radius: 8px;
            background-color: #FAFAFA;
            font-size: 11px;
        }}

        QListWidget::item:selected {{
            background-color: {G_ACTIVE_BG};
            color: {G_ACTIVE_TEXT};
            border-radius: 4px;
        }}

        QHeaderView::section {{
            background-color: #F8F9FA;
            border: none;
            border-bottom: 1px solid {G_BORDER};
            padding: 6px;
            font-weight: 600;
        }}
        """


# --- Main Application Execution ---
if __name__ == "__main__":
    try:
        import pandas;
        import numpy;
        import openpyxl;
        import matplotlib
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from PIL import Image, ImageDraw
    except ImportError as e:
        print(
            f"Missing required library: {e}. Please install them using 'pip install pandas numpy openpyxl matplotlib PyQt6 Pillow'")
        sys.exit(1)

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    try:
        if not os.path.exists('sun.png'):
            img = Image.new('RGBA', (32, 32), (0, 0, 0, 0));
            draw = ImageDraw.Draw(img)
            draw.ellipse((4, 4, 28, 28), fill=(255, 140, 0, 255), outline=(255, 69, 0, 255));
            img.save('sun.png')
        if not os.path.exists('moon.png'):
            img = Image.new('RGBA', (32, 32), (0, 0, 0, 0));
            draw = ImageDraw.Draw(img)
            draw.ellipse((2, 2, 28, 28), fill=(150, 150, 150, 255));
            draw.ellipse((8, 8, 34, 34), fill=(0, 0, 0, 0), outline=(0, 0, 0, 0));
            img.save('moon.png')
    except ImportError:
        pass

    app = QApplication(sys.argv)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'Arial']
    plt.rcParams['font.size'] = 9

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
