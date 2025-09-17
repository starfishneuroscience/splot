import numpy as np


class RingBuffer:
    def __init__(self, buffer_length, dtype="B", adaptive_dtype=False, ignore_incompatible_dtype=False):
        self._buffer = np.empty((buffer_length,), dtype=dtype)
        self._ignore_incompatible_dtype = ignore_incompatible_dtype
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

    def clear_overflow_flag(self):
        self._overflow_flag = False

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

    def seek_until(self, byte, go_to_next: bool = False):
        """Increment read pointer until we've found a particular byte, or there's no more data
        to read. If the read pointer is currently pointing to the requested byte, nothing, will
        happen unless go_to_next=True (in which case, the next `byte` will be sought out, if
        available).
        """
        if go_to_next and self._read_ptr != self._write_ptr:
            self._read_ptr = (self._read_ptr + 1) % len(self._buffer)

        while self._buffer[self._read_ptr] != byte and self._read_ptr != self._write_ptr:
            self._read_ptr = (self._read_ptr + 1) % len(self._buffer)

    def num_unread(self) -> int:
        return (self._write_ptr - self._read_ptr) % len(self._buffer)

    def seek_forward(self, n: int):
        if n > self.num_unread():
            self._read_ptr = self._write_ptr
        else:
            self._read_ptr = (self._read_ptr + int(n)) % len(self._buffer)


if __name__ == "__main__":
    # test ringbuffer
    rb = RingBuffer(100, dtype=int)
    print(f"{rb.read_new()=}")

    print("adding 0:49 to ring buffer")
    rb.add(np.arange(50))
    print(f"{rb.read_new()=}")
    print(f"{rb.read_new()=}")

    print("adding 0:59 to ring buffer")
    rb.add(np.arange(60))
    print(f"{rb.read_new()=}")
    print(f"{rb.read_new()=}")

    print("adding 0:120 to ring buffer")
    rb.add(np.arange(120))
    print(f"{rb.read_new()=}")
    print(f"{rb.read_new()=}")

    print("adding 0:149 to ring buffer, individually")
    for i in range(150):
        rb.add(i)
    print(f"{rb.read_new()=}")
    print(f"{rb.read_new()=}")

    print("adding 0:14 to ring buffer 10 times")
    for i in range(10):
        rb.add(np.arange(15))
    print(f"{rb.read_new(peek=True)=}")
    print(f"{rb.read_new()=}")
    print(f"{rb.read_new()=}")
