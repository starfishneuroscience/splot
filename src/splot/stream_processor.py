import csv
import datetime
import json
import logging
import re
import time

import numpy as np
from numpy.lib import recfunctions as rfn

from .ring_buffer import RingBuffer


logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    This class processes data from a SerialReceiver's ringbuffer, converting
    it to a structured array in a ringbuffer that is used for plotting.
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
        self.running = False
        self.paused = paused
        self.serial_receiver = serial_receiver

        self.binary = binary
        # binary_dtype does not include separator byte
        self.binary_dtype = None
        self.ascii_num_streams = ascii_num_streams

        self.save_timestamps = False
        self.read_ptr = 0  # read pointer for *SerialReceiver* ringbuffer

        self.save_file = None  # handle to file for saving data
        self.csv_writer = None

        if self.binary:
            self.message_delimiter = int(message_delimiter) % 256
            try:
                self.binary_dtype = np.dtype(binary_dtype_string)
            except Exception as e:
                logger.error(f'Failed to set binary data format "{binary_dtype_string}".\n')
                raise e

            self.complete_message_dtype = np.dtype(self.binary_dtype.descr + [("timestamp_usec", "u8")])

        else:
            self.message_delimiter = bytes(message_delimiter, "utf-8").decode("unicode_escape")
            dtype_list = [(f"f{i}", "f8") for i in range(self.ascii_num_streams)] + [("timestamp_usec", "u8")]
            self.complete_message_dtype = np.dtype(dtype_list)

        self.plot_buffer = RingBuffer(plot_buffer_length, dtype=self.complete_message_dtype)

        # compile regex to parse numbers out of arbitrary strings
        numeric_const_pattern = r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
        self.numeric_rx = re.compile(numeric_const_pattern, re.VERBOSE)

    def change_plot_buffer_length(self, length):
        self.plot_buffer = RingBuffer(length, dtype=self.complete_message_dtype)
        self.write_ptr = 0

    def get_num_streams(self):
        return len(self.binary_dtype) if self.binary else self.ascii_num_streams

    def start_saving(self, save_location: str, save_timestamps: bool, series_names: list[str]) -> None:
        # start recording data
        filename = datetime.datetime.now().strftime("serialcapture_%Y-%m-%d_%H-%M-%S")
        full_path = save_location + "/" + filename + (".bin" if self.binary else ".csv")

        self.save_timestamps = save_timestamps
        self.save_file = open(full_path, "wb" if self.binary else "w")

        # write headers
        if self.binary:
            header = {
                "dtype": self.complete_message_dtype.descr if self.save_timestamps else self.binary_dtype.descr,
                "series_names": series_names,
            }
            byte_str = bytes(json.dumps(header), "utf-8")
            self.save_file.write(byte_str)
        else:
            writer = csv.writer(self.save_file)
            writer.writerow(series_names + (["timestamp_usec"] if save_timestamps else []))

        return full_path

    def stop_saving(self) -> None:
        if self.save_file:
            self.save_file.close()
            self.save_file = None

    def process_new_data(self):
        """This slot should be connected to serial_receiver's data_received signal."""
        if self.paused:
            return

        # get new data
        new_data = self.serial_receiver.get_new_data(self.read_ptr)
        num_bytes_read = len(new_data)

        timestamp = time.time_ns() // 1000

        if not self.binary:
            try:
                new_data = new_data.tobytes().decode("ascii")
            except Exception as e:
                logger.error(f"Couldn't decode data as ascii, exception: {e}")
                # update read pointer and ignore the data we couldn't decode
                self.read_ptr = (self.read_ptr + num_bytes_read) % len(self.serial_receiver.ring_buffer)

            if self.message_delimiter not in new_data:
                return
            messages = new_data.split(self.message_delimiter)

            # drop last message (if new_data ends in delimiter, last message will be empty "",
            # and if new_data doesnt end in delimiter, we dont know its a complete message)
            num_bytes_read -= len(messages[-1])
            messages = messages[:-1]

            # convert to tuple containing numeric values as strings
            message_numbers = [tuple(self.numeric_rx.findall(msg)) for msg in messages]
            message_lengths = [len(nums) for nums in message_numbers]

            # drop any invalid (wrong-length) messages and warn about it
            num_bad_messages = np.sum([length != self.ascii_num_streams for length in message_lengths])
            if num_bad_messages:
                logger.warning(f"Dropping {num_bad_messages} messages of wrong length.")
            message_numbers = [x for x in message_numbers if len(x) == self.ascii_num_streams]

            structured_data = np.array(
                [(*nums, timestamp) for nums in message_numbers], dtype=self.complete_message_dtype
            )

            self.plot_buffer.add(structured_data)

            # if we're saving to file, dump new data to file here.
            if self.save_file is not None:
                for msg in message_numbers:
                    line = ",".join(msg)
                    if self.save_timestamps:
                        line += "," + str(timestamp)
                    self.save_file.write(line + "\n")

        elif self.binary:
            expected_length = self.binary_dtype.itemsize  # does not include delimiter byte

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

            # drop delimiter byte from each message
            messages = [x[1:] for x in messages]

            # check if last message is truncated, and if so, remove it
            if len(messages[-1]) < expected_length:
                num_bytes_read -= len(messages[-1])
                messages = messages[:-1]

            # remove any invalid messages with extra bytes
            message_lengths = np.array([len(msg) for msg in messages])
            if any(message_lengths != expected_length):
                bad_message_lengths = message_lengths[message_lengths != expected_length]
                logger.error(f"Dropping {len(bad_message_lengths)} bad messages. Lengths: {set(bad_message_lengths)}.")
                messages = [m for m in messages if len(m) == expected_length]

            if len(messages) == 0:
                # we received either an incomplete packet, or 1+ invalid packets
                return

            # messages to streams (e.g. 8 byte message to 2 uint8s, 1 uint16, 1 uint32):
            # the numpy parser is amazing, see docs:
            #   https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
            structured_data = np.frombuffer(np.concatenate(messages), self.binary_dtype)

            # add timestamp
            structured_data = rfn.append_fields(
                structured_data,
                names="timestamp_usec",
                data=np.repeat(np.uint64(timestamp), len(structured_data)),
            )

            # if we're saving to file, dump new data to file here
            if self.save_file is not None:
                self.save_file.write(structured_data.tobytes())

            # update ring-buffer that plot UI will use
            self.plot_buffer.add(structured_data)

        self.read_ptr = (self.read_ptr + num_bytes_read) % len(self.serial_receiver.ring_buffer)  # TODO: hack
