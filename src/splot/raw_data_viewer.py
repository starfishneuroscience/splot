from PyQt6 import QtCore, QtWidgets, QtGui


class RawDataViewer(QtWidgets.QWidget):
    def __init__(self, get_data_function, update_interval_ms=50):
        """Class for visualizing raw serial data in a textbox, as either ascii text
        or as a series of hex byte values.

        :param get_data_function: function that returns the most recent data received
            on the serial port
        :param update_interval_ms: how often the textbox should be updated.
        """
        super().__init__()

        self.setWindowTitle("Raw serial data (last 10000 bytes)")
        self.resize(400, 300)

        self.get_data_function = get_data_function
        self.update_interval_ms = update_interval_ms

        layout = QtWidgets.QVBoxLayout(self)
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["ascii", "hex", "decimal"])
        h_layout.addWidget(self.combo)

        self.button = QtWidgets.QPushButton("Pause")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.pause_pressed)
        h_layout.addWidget(self.button)
        self.paused = False

        self.text_edit = QtWidgets.QTextEdit()
        layout.addWidget(self.text_edit)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_text_box)
        self.timer.start(self.update_interval_ms)

    def update_text_box(self):
        data = self.get_data_function()  # returns bytes

        if self.combo.currentText() == "ascii":
            text = data.decode("ascii", errors="backslashreplace")
        elif self.combo.currentText() == "hex":
            text = " ".join([f"{byte:02X}" for byte in data])
        else:
            text = " ".join([f"{byte:3d}" for byte in data])

        self.text_edit.setText(text)
        self.text_edit.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def pause_pressed(self, checked: bool):
        if checked:
            self.timer.stop()
        else:
            self.timer.start(self.update_interval_ms)
