import logging
import time

import numpy as np
from PyQt6 import QtCore


logger = logging.getLogger(__name__)


class SerialReceiver(QtCore.QThread):
    """This class takes a serial connection and dumps all of its data into a ring buffer"""

    mutex = QtCore.QMutex()
    data_received = QtCore.pyqtSignal()
    data_rate = QtCore.pyqtSignal(int)  # emitted every second

    def __init__(self, read_function, buffer_length, read_chunk_size):
        # read_function must take an argument for the number of bytes to read
        super().__init__()
        self.running = False
        self.read_function = read_function
        self.read_chunk_size = read_chunk_size

        self.ring_buffer = np.empty((buffer_length,), dtype="B")
        self.write_ptr = 0

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

        bytes_in_last_second = 0
        last_data_rate_emitted_timestamp = time.time()

        while self.running:
            # get new data and add it to ring buffer
            try:
                read = self.read_function(self.read_chunk_size)
            except OSError:
                logger.error("Couldn't read from connection. Closing serial receiver.")
                self.stop()
                break

            if len(read):
                bytes_in_last_second += len(read)
                new_data = np.frombuffer(read, dtype="B")

                # update ring buffer
                n = len(read)
                p = len(self.ring_buffer)
                self.mutex.lock()
                if n + self.write_ptr < p:
                    self.ring_buffer[self.write_ptr : self.write_ptr + n] = new_data
                else:
                    n1 = p - self.write_ptr
                    n2 = n - n1
                    self.ring_buffer[self.write_ptr :] = new_data[:n1]
                    try:  # TODO: testing - this line throws exceptions periopdically
                        self.ring_buffer[:n2] = new_data[n1:]
                    except ValueError as e:
                        print(f"{len(read)=} {n1=} {n2=} {e}")
                self.write_ptr = (self.write_ptr + n) % p
                self.mutex.unlock()
                self.data_received.emit()

            if time.time() - last_data_rate_emitted_timestamp >= 1:
                last_data_rate_emitted_timestamp += 1
                self.data_rate.emit(bytes_in_last_second)
                bytes_in_last_second = 0

    def get_new_data(self, read_ptr):
        """Return ringbuffer, starting at read_ptr and ending at write_ptr, unrolled"""
        self.mutex.lock()
        if self.write_ptr > read_ptr:
            buffer = self.ring_buffer[read_ptr : self.write_ptr].copy()
        else:
            buffer = np.concatenate((self.ring_buffer[read_ptr:], self.ring_buffer[: self.write_ptr]))
        self.mutex.unlock()
        return buffer
