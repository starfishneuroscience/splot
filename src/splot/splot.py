#!/usr/bin/env python3
import csv
import datetime
import importlib.resources
import json
import logging
from pathlib import Path
import signal
import socket
import sys
import threading

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6 import QtCore, QtWidgets, QtGui, uic
import zmq

from .serial_receiver import SerialReceiver
from .stream_processor import StreamProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = importlib.resources.files("splot") / "splot.ui"
        uic.loadUi(ui_file_path, self)

        self.settings = QtCore.QSettings("utilities", "splot")
        logger.info(f"Settings obtained from: {self.settings.fileName()}")

        # continually check serial port availability 3x/second
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_serial_ports)
        self.timer.start(300)

        self.serial_receiver = None
        self.stream_processor = None

        self.socket = None  # keep these around to properly close them
        self.serial_connection = None

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
        self.save_file = None

        self.zmq_listener_thread = None
        self.zmq_listener_loop_running = False
        self.zmq_emitter_conn = None

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
                self.serial_connection = None
                return
            else:
                logger.info(f"Trying to connect to socket: {text}")
                try:
                    socket_type = socket.SOCK_DGRAM if text.startswith("udp://") else socket.SOCK_STREAM
                    self.socket = socket.socket(socket.AF_INET, socket_type)
                    host, port = text.rsplit(":")
                    self.socket.connect((host, int(port)))
                    read_function = self.socket.recv
                except Exception as e:
                    logger.error(f"Failed to connect to {text}. Error: {e}")
                    return
        else:
            logger.info(f"Trying to connect to serial port: {port}")
            self.serial_connection = serial.Serial(
                port,
                baudrate=int(self.serialBaudRateComboBox.currentText()),
                parity=self.serialParityComboBox.currentText(),
                stopbits=float(self.serialStopBitsComboBox.currentText()),
                timeout=0.010,
            )
            read_function = self.serial_connection.read

        self.serial_receiver = SerialReceiver(
            read_function=read_function,
            buffer_length=self.serialBufferSizeSpinBox.value(),
            read_chunk_size=self.serialReadChunkSizeSpinBox.value(),
            forward_conn=self.zmq_emitter_conn,
        )
        binary = self.dataFormatComboBox.currentText() == "binary"
        if binary:
            delimiter = self.binaryMessageDelimiterSpinBox.value()
        else:
            delimiter = self.asciiMessageDelimiterLineEdit.text()
        self.stream_processor = StreamProcessor(
            serial_receiver=self.serial_receiver,
            plot_buffer_length=self.plotLengthSpinBox.value(),
            message_delimiter=delimiter,
            binary=binary,
            binary_dtype_string=self.binaryDtypeStringLineEdit.text(),
            ascii_num_streams=self.numberOfStreamsSpinBox.value(),
            paused=self.pausePushButton.isChecked(),
        )
        self.serial_receiver.data_received.connect(self.stream_processor.process_new_data)
        self.serial_receiver.data_rate.connect(lambda x: self.statusBar().showMessage(f"Data rate: {x} bytes/sec"))
        self.create_plot_series(num_streams=self.stream_processor.get_output_dimensions()[1])
        self.serial_receiver.start()
        self.enable_ui_elements_on_connection(connected=True)
        self.plot_timer.start(33)

    def disconnect_from_serial(self):
        self.plot_timer.stop()
        if self.stream_processor is not None:
            self.stream_processor = None

        if self.serial_receiver is not None:
            self.serial_receiver.stop()
            self.serial_receiver.wait()
            self.serial_receiver.disconnect()  # disconnect all slots
            self.serial_receiver = None

        if self.serial_connection is not None:
            self.serial_connection.close()
        if self.socket is not None:
            self.socket.close()

        self.savePushButton.setChecked(False)
        self.enable_ui_elements_on_connection(connected=False)
        self.statusBar().showMessage("Disconnected.")

    def enable_ui_elements_on_connection(self, connected: bool):
        self.serialParametersFrame.setEnabled(not connected)
        self.seriesPropertyFrame.setEnabled(connected)
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

    def vector_to_bit_raster(self, dat):
        x = []
        y = []

        if not all(np.isnan(dat)) and np.nanmax(dat) > 0 and np.isfinite(np.nanmax(dat)):
            max_bit_index = int(np.log2(np.nanmax(dat)) + 1)

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

    def update_stream_plots(self):
        for i, plot in enumerate(self.plots):
            series = plot.listDataItems()[0]
            j = self.stream_processor.write_ptr
            self.plot_cursor_lines[i].setValue(j)
            dat = self.stream_processor.plot_buffer[:, i].copy()

            if self.plot_types[i] == 1:  # raster
                # convert data to raster (series of line segments)
                x, y = self.vector_to_bit_raster(dat)
                series.setData(x, y, connect="pairs")
            else:
                dat[j] = np.nan  # force plotting break
                series.setData(dat)

    def closeEvent(self, event):
        """This function is called when the main window is closed"""
        self.disconnect_from_serial()

    @QtCore.pyqtSlot(int)
    def on_serialPortComboBox_currentIndexChanged(self, index):
        logger.info(f"you changed the serial port to index {index}!")
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

        if self.stream_processor is not None:
            if binary:
                self.stream_processor.set_binary_mode(
                    binary_dtype_string=self.binaryDtypeStringLineEdit.text(),
                    message_delimiter=self.binaryMessageDelimiterSpinBox.value(),
                )
            else:
                self.stream_processor.set_ascii_mode(
                    ascii_num_streams=self.numberOfStreamsSpinBox.value(),
                    message_delimiter=self.asciiMessageDelimiterLineEdit.text(),
                )
            self.create_plot_series(num_streams=self.stream_processor.get_output_dimensions()[1])

    @QtCore.pyqtSlot()
    def on_asciiMessageDelimiterLineEdit_editingFinished(self):
        value = self.asciiMessageDelimiterLineEdit.text()
        self.settings.setValue("ui/asciiMessageDelimiter", value)
        if self.stream_processor is not None:
            self.stream_processor.set_message_delimiter(value)

    @QtCore.pyqtSlot(int)
    def on_binaryMessageDelimiterSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/binaryMessageDelimiter", value)
        if self.stream_processor is not None:
            self.stream_processor.set_message_delimiter(value)

    @QtCore.pyqtSlot()
    def on_binaryDtypeStringLineEdit_editingFinished(self):
        value = self.binaryDtypeStringLineEdit.text()
        self.settings.setValue("ui/binaryDtypeString", value)
        if self.stream_processor is not None:
            self.stream_processor.set_binary_dtype(value)
            # we may have changed the number of plots, so re-create the plot series
            self.create_plot_series(num_streams=self.stream_processor.get_output_dimensions()[1])

    @QtCore.pyqtSlot(int)
    def on_numberOfStreamsSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/numberOfStreams", value)
        if self.stream_processor is not None:
            self.stream_processor.set_ascii_mode(
                ascii_num_streams=value,
                message_delimiter=self.asciiMessageDelimiterLineEdit.text(),
            )
            # we may have changed the number of plots, so re-create the plot series
            self.create_plot_series(num_streams=self.stream_processor.get_output_dimensions()[1])

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
            # start recording data
            filename = datetime.datetime.now().strftime("serialcapture_%Y-%m-%d_%H-%M-%S")
            full_path = self.saveLocationLabel.text() + "/" + filename
            full_path += ".bin" if self.stream_processor.binary else ".csv"
            logger.info(f"Creating file for saving data: {full_path}")

            series_names = [plot.getAxis("left").labelText for plot in self.plots]

            if self.stream_processor.binary:
                self.save_file = open(full_path, "wb")
                # write json header

                header = {"dtype_string": self.binaryDtypeStringLineEdit.text(), "series_names": series_names}
                byte_str = bytes(json.dumps(header), "utf-8")
                self.save_file.write(byte_str)

            else:  # ascii data, write csv header
                self.save_file = open(full_path, "w")
                writer = csv.writer(self.save_file)
                writer.writerow(series_names)

            # give handle to stream_processor to dump data into
            self.stream_processor.save_file = self.save_file

            # let the user know we're saving, change the button's function to "stop saving"
            self.savePushButton.setText("Stop saving")
        elif not checked:
            # stop recording data
            self.savePushButton.setText("Save data")
            if self.save_file is not None:
                logger.info("Closing save file.")
                self.stream_processor.save_file = None
                self.save_file.close()
                self.save_file = None

    @QtCore.pyqtSlot(int)
    def on_receiveDataPortSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/zmqReceiveDataPort", value)

    @QtCore.pyqtSlot(int)
    def on_emitDataPortSpinBox_valueChanged(self, value):
        self.settings.setValue("ui/zmqEmitDataPort", value)

    @QtCore.pyqtSlot(bool)
    def on_emitDataCheckBox_clicked(self, checked):
        self.emitDataPortSpinBox.setEnabled(not checked)
        if checked:
            port = self.emitDataPortSpinBox.value()
            self.zmq_emitter_conn = zmq.Context().socket(zmq.PUB)
            self.zmq_emitter_conn.bind(f"tcp://*:{port}")
            # pass conn to serial receiver
            if self.serial_receiver:
                self.serial_receiver.forward_conn = self.zmq_emitter_conn
        elif not checked:
            if self.serial_receiver:
                self.serial_receiver.forward_conn = None

    @QtCore.pyqtSlot(bool)
    def on_receiveDataCheckBox_clicked(self, checked):
        if checked:
            try:
                port = self.receiveDataPortSpinBox.value()
                conn = zmq.Context().socket(zmq.SUB)
                conn.bind(f"tcp://*:{port}")
                conn.setsockopt(zmq.RCVTIMEO, 100)  # set 100ms timeout
                conn.subscribe(b"")
                self.receiveDataPortSpinBox.setEnabled(False)
                logger.info(f"Now receiving data on port {port}, will be forwarded to serial connection")
            except Exception:
                logger.error(f"Unable to bind port {port} to for receiving outgoing serial data.")
                self.receiveDataCheckBox.setChecked(False)
                return

            # start listener loop
            self.zmq_listener_thread = threading.Thread(target=self.zmq_listener_loop, args=(conn,))
            self.zmq_listener_thread.start()

        elif not checked:
            # tear down listener loop thread if it exists
            self.zmq_listener_loop_running = False
            if self.zmq_listener_thread:
                self.zmq_listener_thread.join()
            self.receiveDataPortSpinBox.setEnabled(True)

    def zmq_listener_loop(self, conn):
        self.zmq_listener_loop_running = True
        while self.zmq_listener_loop_running:
            try:
                data = conn.recv()
            except zmq.ZMQError:
                continue
            if self.serial_connection:
                self.serial_connection.write(data)

    @QtCore.pyqtSlot(int)
    def on_plotLengthSpinBox_valueChanged(self, plot_buffer_length):
        self.settings.setValue("ui/plotLength", plot_buffer_length)
        if self.stream_processor is not None:
            self.stream_processor.change_plot_buffer_length(plot_buffer_length)

    @QtCore.pyqtSlot(bool)
    def on_pausePushButton_clicked(self, checked):
        if self.stream_processor is not None:
            self.stream_processor.paused = checked
        if checked:
            self.plot_timer.stop()
        else:
            self.plot_timer.start(33)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
