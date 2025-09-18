#!/usr/bin/env python3
import importlib.resources
import logging
from pathlib import Path
import re
import signal
import sys
import multiprocessing
import time

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6 import QtCore, QtWidgets, QtGui, uic

from .stream_processor import start_stream_processor
from .ring_buffer import RingBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def parse_splot_dtype_string(text):
    """Find pattern 'n[ABC]' and replace with n comma-separated repeats of 'ABC'.

    This function exists to allow users to easily specify complex packet structures,
    e.g., a packet containing a 3 floats and 5 int16s could be written "3[f4],5[i2]"
    instead of "f4,f4,f4,i2,i2,i2,i2,i2".
    """
    pattern = r"([1-9]\d*)\[(.*?)\]"

    def replacer(match):
        multiplier = int(match.group(1))
        content = match.group(2)
        return ",".join([content] * multiplier)

    return re.sub(pattern, replacer, text)


def vector_to_bit_raster(dat: np.ndarray[int], max_bit_index: int = None):
    """Take a vector of integers and return plotting vectors x and y.
    :returns: vectors x and y such that plotting x vs y yields vertical line segments
        at each index when a given bit is 1. E.g., if dat[3] == 5, then x and y will
        include two line segments at x=3 around y=0 and y=2. Note that this data must
        be plotted with pyqtgraph's `connect='pairs'` argument.
    """
    x = []
    y = []

    if max_bit_index is None:
        if not all(np.isnan(dat)) and np.nanmax(dat) > 0 and np.isfinite(np.nanmax(dat)):
            max_bit_index = int(np.ceil(np.log2(np.nanmax(dat))))
        else:
            max_bit_index = 7

    # analyze only non-zero indices for speed (sparsity assumption)
    non_zero_indices = np.where(dat > 0)[0]

    # go through each bit up to max_bit_value and add vertical lines of height 0.8 at each x position
    for bit_index in range(max_bit_index):
        ind = np.where(dat[non_zero_indices].astype(int) & (1 << bit_index))[0]
        x.append(np.repeat(non_zero_indices[ind], 2))
        y.append(np.tile([bit_index - 0.4, bit_index + 0.4], len(ind)))

    # add horizontal lines for each bit (so that all less-significant bits will still get plotted)
    x.append(np.tile([0, len(dat)], max_bit_index + 1))
    y.append(np.concatenate([[i, i] for i in range(max_bit_index + 1)]))

    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = importlib.resources.files("splot") / "splot.ui"
        uic.loadUi(ui_file_path, self)

        # stream processor runs in a separate process
        # we communicate with that process via a pipe that enables RPCs
        self.stream_processor_conn, child_conn = multiprocessing.Pipe()
        self.stream_processor_process = multiprocessing.Process(target=start_stream_processor, args=(child_conn,))
        self.stream_processor_process.start()

        self.settings = QtCore.QSettings("utilities", "splot")
        logger.info(f"Settings obtained from: {self.settings.fileName()}")

        # continually check serial port availability 3x/second
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_serial_ports)
        self.timer.start(300)

        # timer to update plots
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_stream_plots)

        # set plot background and color based on system theme
        palette = QtWidgets.QApplication.palette()
        bgcolor = palette.color(QtGui.QPalette.ColorRole.Window)
        fgcolor = palette.color(QtGui.QPalette.ColorRole.WindowText)
        self.plot_series_color = fgcolor
        pg.setConfigOption("background", bgcolor.name())
        pg.setConfigOption("foreground", fgcolor.name())

        self.plots = []
        self.plot_cursor_lines = []
        self.plot_types = []

        self.plot_layout = pg.GraphicsLayoutWidget()
        self.plot_layout.ci.layout.setSpacing(0.0)
        self.plot_layout.ci.setContentsMargins(0.0, 0.0, 0.0, 0.0)
        self.plotVBoxLayout.addWidget(self.plot_layout)
        # suppress constant debug messages on mac associated with trackpad
        self.plot_layout.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        self.populate_serial_options()

        self.load_stored_settings()

        self.enable_ui_elements_on_connection(connected=False)

        self.saveLocationLabel.setText(str(Path.home()))

        self.zmq_listener_thread = None
        self.zmq_listener_loop_running = False

        self.plot_buffer = RingBuffer(self.plotLengthSpinBox.value(), adaptive_dtype=True)

        self.start_time = time.time_ns() // 1000

    def populate_serial_options(self):
        option_map = {
            self.serialBaudRateComboBox: serial.serialutil.SerialBase.BAUDRATES,
            self.serialStopBitsComboBox: serial.serialutil.SerialBase.STOPBITS,
            self.serialParityComboBox: serial.serialutil.SerialBase.PARITIES,
        }
        for combo_box, values in option_map.items():
            combo_box.blockSignals(True)
            combo_box.addItems([str(x) for x in values])
            combo_box.blockSignals(False)

    def load_stored_settings(self):
        settings_map = {
            "ui/serialBaudRate": self.serialBaudRateComboBox.setCurrentText,
            "ui/serialParityIndex": lambda x: self.serialParityComboBox.setCurrentIndex(int(x)),
            "ui/serialStopBitsIndex": lambda x: self.serialStopBitsComboBox.setCurrentIndex(int(x)),
            "ui/serialReadChunkSize": lambda x: self.serialReadChunkSizeSpinBox.setValue(int(x)),
            "ui/serialBufferSize": lambda x: self.serialBufferSizeSpinBox.setValue(int(x)),
            "ui/numberOfStreams": lambda x: self.numberOfStreamsSpinBox.setValue(int(x)),
            "ui/dataFormatIndex": lambda x: self.dataFormatComboBox.setCurrentIndex(int(x)),
            "ui/asciiMessageDelimiter": self.asciiMessageDelimiterLineEdit.setText,
            "ui/binaryMessageDelimiter": lambda x: self.binaryMessageDelimiterSpinBox.setValue(int(x)),
            "ui/binaryDtypeString": self.binaryDtypeStringLineEdit.setText,
            "ui/plotLength": lambda x: self.plotLengthSpinBox.setValue(int(x)),
            "ui/zmqReceiveDataPort": lambda x: self.receiveDataPortSpinBox.setValue(int(x)),
            "ui/zmqEmitDataPort": lambda x: self.emitDataPortSpinBox.setValue(int(x)),
            "ui/saveTimestamps": lambda x: self.saveTimestampsCheckBox.setChecked(int(x)),
        }
        for key, set_function in settings_map.items():
            value = self.settings.value(key)
            if value is not None and set_function is not None:
                set_function(value)

        # update UI widget visibility as needed:
        self.on_dataFormatComboBox_currentIndexChanged(self.dataFormatComboBox.currentIndex())

    def connect_to_serial(self):
        port = self.serialPortComboBox.currentData()
        text = self.serialPortComboBox.currentText()
        if port is None:  # its either empty or a user-entered string
            if text == "(not connected)":
                self.stream_processor_rpc("disconnect_from_serial")
                return
            self.stream_processor_rpc("connect_to_serial", is_socket=True, port=text)
        else:
            self.stream_processor_rpc(
                "connect_to_serial",
                is_socket=False,
                port=port,
                baudrate=int(self.serialBaudRateComboBox.currentText()),
                parity=self.serialParityComboBox.currentText(),
                stopbits=float(self.serialStopBitsComboBox.currentText()),
            )

        self.enable_ui_elements_on_connection(connected=True)
        self.plot_timer.start(33)
        self.statusBar().showMessage("Connected.")

    def data_format_updated(self):
        """Update StreamProcessor's data format based on the current UI settings.

        Also triggers creating a new plot series.
        """

        binary = self.dataFormatComboBox.currentText() == "binary"
        if binary:
            delimiter = self.binaryMessageDelimiterSpinBox.value()
            delimiter = int(delimiter) % 256
        else:
            delimiter = self.asciiMessageDelimiterLineEdit.text()
            # interpret escape characters correctly
            delimiter = bytes(delimiter, "utf-8").decode("unicode_escape")

        parsed_dtype_string = parse_splot_dtype_string(self.binaryDtypeStringLineEdit.text())

        self.stream_processor_rpc(
            "configure_message_format",
            message_delimiter=delimiter,
            binary=binary,
            binary_dtype_string=parsed_dtype_string,
            ascii_num_streams=self.numberOfStreamsSpinBox.value(),
        )

        num_streams = self.stream_processor_rpc("get_num_streams")
        self.create_plot_series(num_streams=num_streams)

    def disconnect_from_serial(self):
        self.plot_timer.stop()
        self.stream_processor_rpc("disconnect_from_serial")

        self.savePushButton.setChecked(False)
        self.enable_ui_elements_on_connection(connected=False)
        self.statusBar().showMessage("Disconnected.")

    def enable_ui_elements_on_connection(self, connected: bool):
        self.serialParametersGroupBox.setEnabled(not connected)
        self.seriesPropertyGroupBox.setEnabled(connected)
        self.savePushButton.setEnabled(connected)

    def update_serial_ports(self):
        new_ports = serial.tools.list_ports.comports()
        prev_ports = [self.serialPortComboBox.itemData(i) for i in range(self.serialPortComboBox.count())]
        current_port = self.serialPortComboBox.currentData()

        # handle new entries
        for port, desc, hwid in sorted(new_ports):
            text = f"{port}: {desc} [{hwid}]"
            if port not in prev_ports:
                logger.info(f"Detected {port}")
                self.serialPortComboBox.addItem(text, port)

        # handle deleted entries.
        # Note: any user-entered strings will have itemData=None so won't be deleted
        for port in prev_ports:
            if port not in [p[0] for p in new_ports] + [None]:
                logger.info(f"Lost {port}")
                # if we're removing the one we're connected to, reset index to 0
                if port == current_port:
                    self.disconnect_from_serial()
                    self.serialPortComboBox.setCurrentIndex(0)
                self.serialPortComboBox.removeItem(self.serialPortComboBox.findData(port))

    def create_plot_series(self, num_streams):
        self.plots = []
        self.plot_cursor_lines = []
        self.plot_layout.clear()

        visible = [self.settings.value(f"ui/seriesVisible[{i}]") for i in range(num_streams)]
        visible = [True if x is None else bool(x) for x in visible]
        visible_plots = np.where(visible)[0]

        for i in range(num_streams):
            plot = self.plot_layout.addPlot(x=[], y=[], row=i, col=0, pen=self.plot_series_color)
            line = pg.InfiniteLine(pos=0, angle=90, pen="red")
            plot.addItem(line)
            plot.setLabel("left", self.settings.value(f"ui/seriesName[{i}]"))
            plot.setVisible(visible[i])
            if i > 0:
                plot.setXLink(self.plots[0])
            if i != visible_plots[-1]:
                plot.hideAxis("bottom")
            self.plots.append(plot)
            self.plot_cursor_lines.append(line)

        # default to plot_type = 0 if no QSettings entry exists
        settings_plot_types = [self.settings.value(f"ui/seriesPlotType[{i}]") for i in range(num_streams)]
        self.plot_types = [0 if x is None else int(x) for x in settings_plot_types]

        self.seriesSelectorSpinBox.setValue(0)
        self.seriesSelectorSpinBox.setMaximum(num_streams - 1)

        # set initial UI elements to correct values
        self.on_seriesSelectorSpinBox_valueChanged(0)

    @QtCore.pyqtSlot(str)
    def on_xAxisChoiceComboBox_currentTextChanged(self, x_axis_choice):
        for line in self.plot_cursor_lines:
            line.setVisible(x_axis_choice != "time (s)")

    def update_stream_plots(self):
        new_data = self.stream_processor_rpc("get_new_messages")
        self.plot_buffer.add(new_data)
        cursor_x = self.plot_buffer._write_ptr

        data = self.plot_buffer.get_valid_buffer()

        use_timestamp = self.xAxisChoiceComboBox.currentText() == "time (s)"
        if use_timestamp:
            data = np.roll(data, -cursor_x)

        for i, plot in enumerate(self.plots):
            series = plot.listDataItems()[0]
            stream_data = data[f"f{i}"]

            if not use_timestamp:
                self.plot_cursor_lines[i].setValue(cursor_x)

            if self.plot_types[i] == 1:  # raster
                # convert data to raster (series of line segments)
                x, y = vector_to_bit_raster(stream_data)
                if use_timestamp:
                    x = (data["timestamp_usec"][x] - self.start_time) / 1e6
                series.setData(x, y, connect="pairs")
            else:
                if use_timestamp:
                    x = (data["timestamp_usec"] - self.start_time) / 1e6
                    step_mode = "right"
                else:
                    x = np.arange(len(stream_data))
                    step_mode = None
                series.setData(x, stream_data, connect="finite", stepMode=step_mode)

    def closeEvent(self, event):
        """This function is called when the main window is closed"""
        self.stream_processor_rpc("close")
        self.stop_zmq_listener()

    @QtCore.pyqtSlot(int)
    def on_serialPortComboBox_currentIndexChanged(self, index):
        self.disconnect_from_serial()
        self.connect_to_serial()

    @QtCore.pyqtSlot(str)
    def on_serialBaudRateComboBox_currentTextChanged(self, value):
        self.settings.setValue("ui/serialBaudRate", value)

    @QtCore.pyqtSlot(int)
    def on_serialParityComboBox_currentIndexChanged(self, index):
        self.settings.setValue("ui/serialParityIndex", index)

    @QtCore.pyqtSlot(int)
    def on_serialStopBitsComboBox_currentIndexChanged(self, index):
        self.settings.setValue("ui/serialStopBitsIndex", index)

    @QtCore.pyqtSlot(int)
    def on_serialReadChunkSizeSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/serialReadChunkSize", value)

    @QtCore.pyqtSlot(int)
    def on_serialBufferSizeSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/serialBufferSize", value)

    @QtCore.pyqtSlot(int)
    def on_dataFormatComboBox_currentIndexChanged(self, index):
        self.settings.setValue("ui/dataFormatIndex", index)
        binary = index == 0
        self.binaryDtypeStringLineEdit.setEnabled(binary)
        self.binaryDtypeStringLabel.setEnabled(binary)
        self.numberOfStreamsLabel.setEnabled(not binary)
        self.numberOfStreamsSpinBox.setEnabled(not binary)
        self.binaryMessageDelimiterSpinBox.setEnabled(binary)
        self.binaryMessageDelimiterLabel.setEnabled(binary)
        self.asciiMessageDelimiterLineEdit.setEnabled(not binary)
        self.asciiMessageDelimiterLabel.setEnabled(not binary)
        self.data_format_updated()

    @QtCore.pyqtSlot()
    def on_asciiMessageDelimiterLineEdit_editingFinished(self):
        value = self.asciiMessageDelimiterLineEdit.text()
        self.settings.setValue("ui/asciiMessageDelimiter", value)
        self.data_format_updated()

    @QtCore.pyqtSlot(int)
    def on_binaryMessageDelimiterSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/binaryMessageDelimiter", value)
        self.data_format_updated()

    @QtCore.pyqtSlot()
    def on_binaryDtypeStringLineEdit_editingFinished(self):
        value = self.binaryDtypeStringLineEdit.text()
        self.settings.setValue("ui/binaryDtypeString", value)
        self.data_format_updated()

    @QtCore.pyqtSlot(int)
    def on_numberOfStreamsSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/numberOfStreams", value)
        self.data_format_updated()

    @QtCore.pyqtSlot(int)
    def on_seriesSelectorSpinBox_valueChanged(self, series_index):
        # update the series property boxes appropriately
        self.seriesVisibleCheckBox.setChecked(self.plots[series_index].isVisible())
        self.seriesPlotTypeComboBox.setCurrentIndex(self.plot_types[series_index])
        self.seriesNameLineEdit.setText(self.plots[series_index].getAxis("left").labelText)

    @QtCore.pyqtSlot(str)
    def on_seriesNameLineEdit_textChanged(self, text):
        series_index = self.seriesSelectorSpinBox.value()
        self.plots[series_index].setLabel("left", text)
        self.settings.setValue(f"ui/seriesName[{series_index}]", text)

    @QtCore.pyqtSlot(bool)
    def on_seriesVisibleCheckBox_clicked(self, checked):
        series_index = self.seriesSelectorSpinBox.value()
        self.plots[series_index].setVisible(checked)
        self.settings.setValue(f"ui/seriesVisible[{series_index}]", checked)

        visible_plot_indices = [i for i in range(len(self.plots)) if self.plots[i].isVisible()]
        # make sure the last plot has an x-axis visible and the 2nd-to-last doesn't
        self.plots[visible_plot_indices[-2]].hideAxis("bottom")
        self.plots[visible_plot_indices[-1]].showAxis("bottom")

    @QtCore.pyqtSlot(int)
    def on_seriesPlotTypeComboBox_currentIndexChanged(self, index):
        series_index = self.seriesSelectorSpinBox.value()
        self.plot_types[series_index] = index
        self.settings.setValue(f"ui/seriesPlotType[{series_index}]", index)

    @QtCore.pyqtSlot()
    def on_setSaveLocationPushButton_clicked(self):
        self.saveLocationLabel.setText(QtWidgets.QFileDialog.getExistingDirectory())

    @QtCore.pyqtSlot(bool)
    def on_savePushButton_toggled(self, checked):
        # note: toggled signal will be called regardless of whether user clicks or if
        #   we programmatically change the state of the button.
        if checked:
            series_names = [plot.getAxis("left").labelText for plot in self.plots]
            # give info to stream processor to open a file and start saving
            self.stream_processor_rpc(
                "start_saving",
                save_location=self.saveLocationLabel.text(),
                save_timestamps=self.saveTimestampsCheckBox.isChecked(),
                series_names=series_names,
            )

            # let the user know we're saving, change the button's function to "stop saving"
            self.savePushButton.setText("Stop saving")
            self.saveTimestampsCheckBox.setEnabled(False)

        elif not checked:
            # stop recording data
            self.savePushButton.setText("Save data")
            self.saveTimestampsCheckBox.setEnabled(True)
            self.stream_processor_rpc("stop_saving")

    @QtCore.pyqtSlot(bool)
    def on_saveTimestampsCheckBox_clicked(self, checked):
        self.settings.setValue("ui/saveTimestamps", checked)

    @QtCore.pyqtSlot(int)
    def on_receiveDataPortSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/zmqReceiveDataPort", value)

    @QtCore.pyqtSlot(int)
    def on_emitDataPortSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/zmqEmitDataPort", value)

    @QtCore.pyqtSlot(bool)
    def on_emitDataCheckBox_clicked(self, checked):
        if checked:
            port = self.emitDataPortSpinBox.value()
            try:
                self.stream_processor_rpc("start_zmq_forwarding", port=port)
                self.emitDataPortSpinBox.setEnabled(False)
            except Exception:
                self.emitDataPortSpinBox.setChecked(False)
        elif not checked:
            self.stream_processor_rpc("stop_zmq_forwarding")

    def stream_processor_rpc(self, method: str, *args, **kwargs):
        command = {"method": method, "args": args, "kwargs": kwargs}
        self.stream_processor_conn.send(command)
        value = self.stream_processor_conn.recv()
        if isinstance(value, Exception):
            raise value
        return value

    @QtCore.pyqtSlot(bool)
    def on_receiveDataCheckBox_clicked(self, checked):
        if checked:
            try:
                port = self.receiveDataPortSpinBox.value()
                self.stream_processor_rpc("start_zmq_forwarding", port=port)
                self.receiveDataPortSpinBox.setEnabled(False)
            except Exception:
                logger.error(f"Unable to bind port {port} for receiving data to forward to serial.")
                self.receiveDataCheckBox.setChecked(False)
        elif not checked:
            self.stream_processor_rpc("stop_zmq_forwarding")

    @QtCore.pyqtSlot(int)
    def on_plotLengthSpinBox_valueChanged(self, plot_buffer_length):
        self.settings.setValue("ui/plotLength", plot_buffer_length)
        self.plot_buffer = RingBuffer(plot_buffer_length, adaptive_dtype=True)

    @QtCore.pyqtSlot(bool)
    def on_pausePushButton_clicked(self, checked):
        if checked:
            self.plot_timer.stop()
        else:
            # flush streamprocessor buffer, then start getting new data
            self.stream_processor_rpc("get_new_messages")
            self.plot_timer.start(33)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
