import logging
import re

import numpy as np
from numpy.lib import recfunctions as rfn


logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    This class processes data from a SerialReceiver's ringbuffer, converting
    it to a (n x k)-dimensional buffer that is used for plotting.
    """

    def __init__(
        self,
        serial_receiver,
        plot_buffer_length: int,
        message_delimiter: str | int,  # used for both binary and ascii
        binary: bool,
        binary_dtype_string: str,
        ascii_num_streams: int,
        paused: bool = False,
    ):
        # future parameters: long_or_wide,
        super().__init__()
        self.running = False
        self.paused = paused
        self.serial_receiver = serial_receiver

        self.binary = binary
        self.binary_dtype = None

        self.read_ptr = 0  # read pointer for *SerialReceiver* ringbuffer

        self.write_ptr = 0  # write pointer for this class's *plot_buffer*

        if self.binary:
            self.binary_dtype = np.dtype(binary_dtype_string)

            num_streams = np.sum([1 if len(x) <= 2 else np.prod(x[2]) for x in self.binary_dtype.descr])
            num_streams -= 1  # subtract one to ignore header byte
            self.message_delimiter = int(message_delimiter) % 256
        else:
            num_streams = ascii_num_streams
            # process escape characters correctly
            self.message_delimiter = bytes(message_delimiter, "utf-8").decode("unicode_escape")

        self.plot_buffer = np.full((plot_buffer_length, num_streams), np.nan, dtype=float)

        # compile regex to parse numbers out of arbitrary strings
        numeric_const_pattern = r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
        self.numeric_rx = re.compile(numeric_const_pattern, re.VERBOSE)

    def change_plot_buffer_length(self, size):
        self.plot_buffer = np.full((size, self.plot_buffer.shape[1]), np.nan, dtype=float)
        self.write_ptr = 0

    def change_binary_dtype(self, dtype_string):
        try:
            dt = np.dtype(dtype_string)
            # changing dtype probably changes number of fields; wipe buffer and make it the right size
            self.plot_buffer = np.full((self.plot_buffer.shape[0], len(dt) - 1), np.nan, dtype=float)
            self.write_ptr = 0
            self.binary_dtype = dt
        except Exception as e:
            logger.error(f'Failed to set binary data format "{dtype_string}".\n{e}')

    def get_output_dimensions(self):
        return self.plot_buffer.shape

    def process_new_data(self):
        """This slot should be connected to serial_receiver's data_received signal."""
        if self.paused:
            return

        # get new data
        new_data = self.serial_receiver.get_new_data(self.read_ptr)
        num_bytes_read = len(new_data)

        if not self.binary:
            new_data = new_data.tobytes().decode("ascii")
            if self.message_delimiter not in new_data:
                return
            messages = new_data.split(self.message_delimiter)

            # drop last message (if new_data ends in delimiter, last message will be empty "",
            # and if new_data doesnt end in delimiter, we dont know its a complete message)
            num_bytes_read -= len(messages[-1])
            messages = messages[:-1]

            for message in messages:
                if message == "":
                    continue
                nums = self.numeric_rx.findall(message)
                nums = np.array(nums, dtype=float)
                logger.debug(f"parsed {message=} to {nums=}")
                if len(nums) >= self.plot_buffer.shape[1]:
                    self.plot_buffer[self.write_ptr] = nums[: self.plot_buffer.shape[1]]
                else:
                    self.plot_buffer[self.write_ptr, : len(nums)] = nums
                    self.plot_buffer[self.write_ptr, len(nums) :] = np.nan
                self.write_ptr = (self.write_ptr + 1) % self.plot_buffer.shape[0]

        elif self.binary:
            expected_length = self.binary_dtype.itemsize

            # preliminary pass - split on delimiter
            delimiter_indices = np.where(new_data == self.message_delimiter)[0]
            if len(delimiter_indices) == 0:
                return
            # remove delimiters that would make messages too short (must be part of data)
            valid_delimiter_indices = [delimiter_indices[0]]
            for index in delimiter_indices:
                if index - valid_delimiter_indices[-1] >= expected_length:
                    valid_delimiter_indices += [index]
            if valid_delimiter_indices[0] == 0:
                valid_delimiter_indices.pop(0)
            messages = np.split(new_data, valid_delimiter_indices)

            # check if last message is truncated, and if so, remove it
            if len(messages[-1]) < expected_length:
                num_bytes_read -= len(messages[-1])
                messages = messages[:-1]

            # remove any invalid messages with extra bytes
            message_lengths = np.array([len(msg) for msg in messages], dtype=int)
            if any(message_lengths != expected_length):
                bad_message_lengths = message_lengths[message_lengths != expected_length]
                logger.error(
                    f"{len(bad_message_lengths)} bad message lengths detected! {bad_message_lengths}. "
                    "Dropping those messages."
                )
                messages = [m for m in messages if len(m) == expected_length]

            if len(messages) == 0:
                # we received either an incomplete packet, or 1+ invalid packets
                return

            # messages to streams (e.g. 8 byte message to 2 uint8s, 1 uint16, 1 uint32):
            # the numpy parser is amazing, see docs:
            #   https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
            data = np.frombuffer(np.concatenate(messages), self.binary_dtype)
            data = rfn.structured_to_unstructured(data)
            data = data[:, 1:]  # drop 1st column, it's the delimiter which is constant by definition

            # if we're saving to file, dump new data to file here
            # - avro
            # - straight binary
            # - csv

            # update ring-buffer that plot UI will use
            n = len(messages)
            if n >= self.plot_buffer.shape[0]:
                # If data is larger than plot_buffer, just overwrite the whole buffer with the most recent data
                self.plot_buffer[:] = data[-self.plot_buffer.shape[0] :]
            elif self.write_ptr + n <= self.plot_buffer.shape[0]:
                # new data is smaller than plot_buffer and doesnt wrap around
                self.plot_buffer[self.write_ptr : self.write_ptr + n] = data
            else:
                # new data is smaller than plot_buffer and wraps around end of buffer
                n1 = self.plot_buffer.shape[0] - self.write_ptr
                n2 = n - n1
                self.plot_buffer[self.write_ptr :] = data[:n1]
                self.plot_buffer[:n2] = data[n1:]
            self.write_ptr = (self.write_ptr + n) % self.plot_buffer.shape[0]

        self.read_ptr = (self.read_ptr + num_bytes_read) % len(self.serial_receiver.ring_buffer)  # TODO: hack
