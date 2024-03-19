#!/usr/bin/env python3
import importlib.resources
import logging
import serial
import signal
import socket
import sys

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtGui import QPalette

from .serial_receiver import SerialReceiver
from .stream_processor import StreamProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file_path = importlib.resources.files('splot')  / "splot.ui"
        uic.loadUi(ui_file_path, self)
        self.show()

        # continually check serial port availability 3x/second
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_serial_ports)
        self.timer.start(300)

        self.serial_receiver = None
        self.stream_processor = None

        self.socket = None  # have to keep this around to properly close it

        # timer to update plots at 30 Hz
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_stream_plots)
        self.plot_timer.start(33)

        # set plot background and color based on system theme
        palette = QtWidgets.QApplication.palette()
        bgcolor = palette.color(QPalette.ColorRole.Window)
        fgcolor = palette.color(QPalette.ColorRole.WindowText)
        self.plot_series_color = palette.color(QPalette.ColorRole.WindowText)
        pg.setConfigOption('background', bgcolor.name())
        pg.setConfigOption('foreground', fgcolor.name())

        self.plots = []
        self.plot_cursor_lines = []
        self.plot_layout = pg.GraphicsLayoutWidget()
        # suppress constant debug messages on mac associated with trackpad
        self.plot_layout.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.plotVBoxLayout.addWidget(self.plot_layout)

        self.serialBaudRateComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.BAUDRATES])
        self.serialStopBitsComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.STOPBITS])
        self.serialParityComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.PARITIES])

        self.numberOfStreamsLabel.setVisible(False)
        self.numberOfStreamsSpinBox.setVisible(False)

    @QtCore.pyqtSlot(int)
    def on_serialPortComboBox_currentIndexChanged(self, index):
        logger.info(f"you changed the serial port to index {index}!")
        self.disconnect_from_serial()
        self.connect_to_serial()

    def connect_to_serial(self):
        port = self.serialPortComboBox.currentData()
        text = self.serialPortComboBox.currentText()
        if port is None:  # its either empty or a user-entered string
            if text == "(not connected)":
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
            serial_connection = serial.Serial(
                port,
                baudrate=int(self.serialBaudRateComboBox.currentText()),
                parity=self.serialParityComboBox.currentText(),
                stopbits=float(self.serialStopBitsComboBox.currentText()),
                timeout=0.010,
            )
            read_function = serial_connection.read

        self.serial_receiver = SerialReceiver(
            read_function=read_function,
            buffer_length=self.serialBufferSizeSpinBox.value(),
            read_chunk_size=self.serialReadChunkSizeSpinBox.value(),
        )
        self.stream_processor = StreamProcessor(
            serial_receiver=self.serial_receiver,
            plot_buffer_length=self.plotLengthSpinBox.value(),
            message_delimiter=self.messageDelimiterLineEdit.text(),
            binary=(self.dataFormatComboBox.currentText() == "binary"),
            binary_dtype_string=self.binaryDtypeStringLineEdit.text(),
            ascii_num_streams=self.numberOfStreamsSpinBox.value(),
        )
        self.serial_receiver.data_received.connect(self.stream_processor.process_new_data)
        self.serial_receiver.data_rate.connect(self.dataRateBpsValueLabel.setNum)
        self.create_plot_series(num_streams=self.stream_processor.get_output_dimensions()[1])
        self.serial_receiver.start()

        self.serialParametersFrame.setEnabled(False)

    def disconnect_from_serial(self):
        if self.serial_receiver is not None:
            self.serial_receiver.stop()
            self.serial_receiver.wait()
            self.serial_receiver.disconnect()  # disconnect all slots
            self.serial_receiver = None
        if self.socket is not None:
            self.socket.close()
        self.serialParametersFrame.setEnabled(True)

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
        # todo: clear all plots
        self.plots = []
        self.plot_layout.clear()
        for i in range(num_streams):
            plot = self.plot_layout.addPlot(x=[], y=[], row=i, col=0, pen=self.plot_series_color)
            line = pg.InfiniteLine(pos=0, angle=90, pen='red')
            plot.addItem(line)
            self.plots.append(plot)
            self.plot_cursor_lines.append(line)
            if i > 0:
                plot.setXLink(self.plots[0])

    def update_stream_plots(self):
        for i, plot in enumerate(self.plots):
            (series, ) = plot.listDataItems()
            j = self.stream_processor.write_ptr
            self.plot_cursor_lines[i].setValue(j)
            dat = self.stream_processor.plot_buffer[:, i].copy()
            dat[j] = np.nan # force plotting break
            series.setData(dat)

    def closeEvent(self, event):
        """This function is called when the main window is closed"""
        self.disconnect_from_serial()

    def on_plotLengthSpinBox_valueChanged(self, plot_buffer_length):
        if self.stream_processor:
            self.stream_processor.change_plot_buffer_length(plot_buffer_length)

    @QtCore.pyqtSlot(int)
    def on_dataFormatComboBox_currentIndexChanged(self, index):
        binary = (index == 0)
        self.binaryDtypeStringLineEdit.setVisible(binary)
        self.binaryDtypeStringLabel.setVisible(binary)
        self.numberOfStreamsLabel.setVisible(not binary)
        self.numberOfStreamsSpinBox.setVisible(not binary)
        self.messageDelimiterLineEdit.setText("0" if binary else "\\n")

    @QtCore.pyqtSlot(bool)
    def on_pausePushButton_clicked(self, checked):
        if self.stream_processor is not None:
            self.stream_processor.paused = checked

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()


if __name__ == "__main__":
    main()
