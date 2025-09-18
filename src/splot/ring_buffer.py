import numpy as np


class RingBuffer:
    def __init__(self, buffer_length, dtype="B", adaptive_dtype=False):
        self._buffer = np.empty((buffer_length,), dtype=dtype)
        self._adaptive_dtype = adaptive_dtype

        self._read_ptr = 0
        self._write_ptr = 0
        self._overflow_flag = False
        self._overflow_since_last_read = False
        self._full_buffer_valid = False

    def add(self, new_data):
        try:
            n = len(new_data)
        except TypeError:
            new_data = np.array([new_data], dtype=self._buffer.dtype)
            n = 1

        if n == 0:
            return

        try:
            new_data = np.array(new_data, dtype=self._buffer.dtype)
        except TypeError as e:
            if self._adaptive_dtype:
                self._buffer = np.zeros((len(self._buffer),), dtype=new_data.dtype)
                self._read_ptr = 0
                self._write_ptr = 0
                self._overflow_since_last_read = False
                self._overflow_flag = False
                self._full_buffer_valid = False
            else:
                raise TypeError(
                    f"Incompatible data type added to ringbuffer. "
                    f"\nRingbuffer type={self._buffer.dtype}"
                    f"\nnew_data.dtype ={new_data.dtype}"
                    f"\n{new_data=}"
                    f"\n{e=}"
                )

        if n > len(self._buffer):
            # received more data than fits in the buffer, so just store the most recent data
            self._buffer[:] = new_data[-len(self._buffer) :]
            self._write_ptr = 0
            self._read_ptr = 1
            self._overflow_flag = True
            self._overflow_since_last_read = True
            self._full_buffer_valid = True
        elif n + self._write_ptr < len(self._buffer):
            # new data fits without overrunning end of buffer
            self._buffer[self._write_ptr : self._write_ptr + n] = new_data
            self._write_ptr += n
            # check if we've overflowed (written past the read pointer)
            if self._write_ptr >= self._read_ptr > self._write_ptr - n:
                self._overflow_since_last_read = True
        else:
            # new data would overrun end of buffer; wrap it
            n1 = len(self._buffer) - self._write_ptr
            n2 = n - n1
            self._buffer[self._write_ptr :] = new_data[:n1]
            self._buffer[:n2] = new_data[n1:]
            old_write_ptr = self._write_ptr
            self._write_ptr = n2
            self._full_buffer_valid = True
            # check if we've overflowed (written past the read pointer)
            if self._read_ptr > old_write_ptr or self._read_ptr <= self._write_ptr:
                self._overflow_since_last_read = True

    def read_new(self, peek=False):
        if self._overflow_since_last_read:
            # return whole buffer in order, reset overflow flag
            data = np.roll(self._buffer, -self._write_ptr)
            if not peek:
                self._overflow_since_last_read = False
        elif self._read_ptr <= self._write_ptr:
            data = self._buffer[self._read_ptr : self._write_ptr]
        else:
            data = np.concatenate([self._buffer[self._read_ptr :], self._buffer[: self._write_ptr]])
        if not peek:
            self._read_ptr = self._write_ptr

        return data

    def get_valid_buffer(self):
        if self._full_buffer_valid:
            return self._buffer
        else:
            return self._buffer[: self._write_ptr]
