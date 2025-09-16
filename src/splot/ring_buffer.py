import numpy as np


class RingBuffer:
    def __init__(self, buffer_length, dtype="B", adaptive_dtype=False, ignore_incompatible_dtype=False):
        self._buffer = np.empty((buffer_length,), dtype=dtype)
        self._ignore_incompatible_dtype = ignore_incompatible_dtype
        self._adaptive_dtype = adaptive_dtype

        self._write_ptr = 0
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

                self._write_ptr = 0
                self._full_buffer_valid = False
            elif self._ignore_incompatible_dtype:
                return
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
            self._full_buffer_valid = True
        elif n + self._write_ptr < len(self._buffer):
            # new data fits without overrunning end of buffer
            self._buffer[self._write_ptr : self._write_ptr + n] = new_data
            self._write_ptr += n
        else:
            # new data would overrun end of buffer; wrap it
            n1 = len(self._buffer) - self._write_ptr
            n2 = n - n1
            self._buffer[self._write_ptr :] = new_data[:n1]
            self._buffer[:n2] = new_data[n1:]
            self._write_ptr = n2
            self._full_buffer_valid = True

    def get_valid_buffer(self):
        if self._full_buffer_valid:
            return self._buffer
        else:
            return self._buffer[: self._write_ptr]
