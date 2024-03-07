#!/usr/bin/env python3
import logging
import serial
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import serial.tools.list_ports
from PyQt6 import QtCore, QtWidgets, uic
import pyqtgraph as pg

from serial_receiver import SerialReceiver
from stream_processor import StreamProcessor


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # Call the inherited classes __init__ method
        uic.loadUi("splot.ui", self)  # Load the .ui file
        self.show()  # Show the GUI

        # continually check serial port availability 3x/second
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_serial_ports)
        self.timer.start(300)

        self.serial_receiver = None
        self.stream_processor = None

        # timer to update plots at 20 Hz
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_stream_plots)
        self.plot_timer.start(50)

        self.plots = []
        self.plot_layout = pg.GraphicsLayoutWidget()
        self.plot_layout.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.plotVBoxLayout.addWidget(self.plot_layout)

        self.serialBaudRateComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.BAUDRATES])
        self.serialStopBitsComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.STOPBITS])
        self.serialParityComboBox.addItems([str(x) for x in serial.serialutil.SerialBase.PARITIES])

    @QtCore.pyqtSlot(int)
    def on_serialPortComboBox_activated(self, index):
        logger.info(f"you changed the serial port to index {index}!")
        self.disconnect_from_serial()
        self.connect_to_serial()

    def connect_to_serial(self):
        port = self.serialPortComboBox.currentData()
        if port is None:
            return
        logger.info(f"Trying to connect to: {port}")

        serial_connection = serial.Serial(
            port,
            baudrate=int(self.serialBaudRateComboBox.currentText()),
            parity=self.serialParityComboBox.currentText(),
            stopbits=float(self.serialStopBitsComboBox.currentText()),
            timeout=0.010,
        )
        self.serial_receiver = SerialReceiver(
            read_function = serial_connection.read,
            buffer_length=self.serialBufferSizeSpinBox.value(),
            read_chunk_size=self.serialReadChunkSizeSpinBox.value(),
        )
        self.stream_processor = StreamProcessor(
            serial_receiver=self.serial_receiver,
            plot_buffer_length=self.plotLengthSpinBox.value(),
            message_delimiter=self.messageDelimiterLineEdit.text(),
            binary=True,
            binary_dtype_string=self.binaryDtypeStringLineEdit.text(),
            binary_message_length=self.messageLengthBytesSpinBox.value(),
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

        # handle deleted entries
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
            plot = self.plot_layout.addPlot(x=[], y=[], row=i, col=0)
            self.plots.append(plot)
            if i > 0:
                plot.setXLink(self.plots[0])

    def update_stream_plots(self):
        if not self.pausePushButton.isChecked():
            for i, plot in enumerate(self.plots):
                (series,) = plot.listDataItems()
                series.setData(self.stream_processor.plot_buffer[:, i])

    def closeEvent(self, event):
        """This function is called when the main window is closed"""
        self.disconnect_from_serial()

    def on_plotLengthSpinBox_valueChanged(self, plot_buffer_length):
        self.stream_processor.change_plot_buffer_length(plot_buffer_length)


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec()
