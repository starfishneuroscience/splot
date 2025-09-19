import csv
import datetime
import json
import logging
import re
import serial
import socket
import time

import numpy as np
from numpy.lib import recfunctions as rfn
import zmq

from .ring_buffer import RingBuffer


logger = logging.getLogger(__name__)


def start_stream_processor(rpc_conn):
    stream_processor = StreamProcessor(rpc_conn=rpc_conn)
    stream_processor.run()


class StreamProcessor:
    """
    This class processes data from a serial connection and keeps a running message buffer.

    It is intended to run in a separate process and interface with the plotting process
    via multiprocessing.Pipe.

    After creating this object with a multiprocessing pipe connection, you must configure
    it via `configure_message_format` and `connect_to_serial`
    """

    def __init__(self, rpc_conn):
        self.rpc_conn = rpc_conn

        self.message_buffer_length = 100_000
        self.message_delimiter = None
        self.binary = None
        self.binary_dtype_string = None
        self.ascii_num_streams = None
        self.save_timestamps = None

        self.binary_dtype = None  # used for parsing binary data
        self.complete_message_dtype = None  # used for ringbuffer for storing messages
        self.message_buffer = None

        # compile regex to parse numbers out of arbitrary strings
        numeric_const_pattern = r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
        self.numeric_rx = re.compile(numeric_const_pattern, re.VERBOSE)

        self.save_file = None  # handle to file for saving data
        self.csv_writer = None

        self.zmq_forwarding_conn = None
        self.zmq_listener_conn = None

        self.serial_conn = None
        self.serial_read_function = None

        self.bytes_received = None

    def connect_to_serial(self, port, is_socket, baudrate=None, parity=None, stopbits=None):
        self.disconnect_from_serial()

        if is_socket:
            logger.info(f"Trying to connect to socket: {port}")
            socket_type = socket.SOCK_DGRAM if port.startswith("udp://") else socket.SOCK_STREAM
            if port.startswith("udp://"):
                port = port[6:]
            self.serial_conn = socket.socket(socket.AF_INET, socket_type)
            host, port = port.rsplit(":")
            self.serial_conn.connect((host, int(port)))
            self.serial_read_function = self.serial_conn.recv
            logger.info(f"Connected to socket: {port}")
        else:
            logger.info(f"Trying to connect to serial port: {port}")
            self.serial_conn = serial.Serial(
                port,
                baudrate=baudrate,
                parity=parity,
                stopbits=stopbits,
                timeout=0.001,
            )
            self.serial_read_function = self.serial_conn.read
            logger.info(f"Connected to serial port: {port}")

        self.bytes_received = 0

    def disconnect_from_serial(self):
        if self.serial_conn is not None:
            # close the port; if its already failed, may thow an exception
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
            self.serial_read_function = None
            self.bytes_received = None

    def configure_message_format(
        self,
        message_delimiter: str | int,
        binary: bool,
        binary_dtype_string: str,
        ascii_num_streams: int,
    ):
        self.message_delimiter = message_delimiter
        self.binary = binary
        self.binary_dtype_string = binary_dtype_string
        self.ascii_num_streams = ascii_num_streams
        self.init_message_buffer_and_dtypes()

    def set_message_buffer_length(self, length):
        self.message_buffer = RingBuffer(length, dtype=self.complete_message_dtype)

    def init_message_buffer_and_dtypes(self):
        if self.binary:
            try:
                self.binary_dtype = np.dtype(self.binary_dtype_string)
            except Exception as e:
                logger.error(f'Failed to set binary data format "{self.binary_dtype_string}".\n')
                raise e
            self.complete_message_dtype = np.dtype(self.binary_dtype.descr + [("timestamp_usec", "u8")])

        else:
            dtype_list = [(f"f{i}", "f8") for i in range(self.ascii_num_streams)] + [("timestamp_usec", "u8")]
            self.complete_message_dtype = np.dtype(dtype_list)

        self.message_buffer = RingBuffer(self.message_buffer_length, dtype=self.complete_message_dtype)

    def get_num_streams(self):
        return len(self.binary_dtype.descr) if self.binary else self.ascii_num_streams

    def start_saving(self, save_location: str, save_timestamps: bool, series_names: list[str]) -> str:
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

        logger.info(f"Saving data to: {full_path}")
        return full_path

    def stop_saving(self) -> None:
        if self.save_file:
            logger.info("Closing save file.")
            self.save_file.close()
            self.save_file = None

    def start_zmq_forwarding(self, port):
        self.stop_zmq_forwarding()
        self.zmq_forwarding_conn = zmq.Context().socket(zmq.PUB)
        self.zmq_forwarding_conn.bind(f"tcp://*:{port}")
        logger.info(f"Bound tcp://*:{port}, raw serial data will be published on this port")

    def stop_zmq_forwarding(self):
        if self.zmq_forwarding_conn:
            self.zmq_forwarding_conn.close()
            self.zmq_forwarding_conn = None
            logger.info("Stopping serial->zmq forwarding")

    def start_zmq_listener(self, port):
        self.zmq_listener_conn = zmq.Context().socket(zmq.SUB)
        self.zmq_listener_conn.bind(f"tcp://*:{port}")
        self.zmq_listener_conn.setsockopt(zmq.RCVTIMEO, 0)
        self.zmq_listener_conn.subscribe(b"")
        logger.info(f"Bound tcp://*:{port}; incoming data will be forwarded to serial")

    def stop_zmq_listener(self):
        if self.zmq_listener_conn:
            self.zmq_listener_conn.close()
            self.zmq_listener_conn = None
            logger.info("Stopping zmq->serial forwarding")

    def handle_rpc_requests(self):
        if self.rpc_conn is None:
            raise RuntimeError("Cannot communicate with main process.")

        while self.rpc_conn.poll(timeout=0):
            command = self.rpc_conn.recv()
            try:
                # execute the command and send back the return value
                method = getattr(self, command["method"])
                value = method(*command["args"], **command["kwargs"])
                self.rpc_conn.send(value)
            except Exception as err:
                self.rpc_conn.send(err)

    def run(self):
        self.running = True
        leftover_bytes = bytearray()

        while self.running:
            self.handle_rpc_requests()

            if self.serial_read_function is None:
                continue

            # read and transmit any incoming zmq messages that need to go out over serial
            if self.zmq_listener_conn is not None:
                while True:
                    try:
                        data = self.zmq_listener_conn.recv(flags=zmq.NOBLOCK)
                        self.serial_conn.write(data)
                    except zmq.ZMQError:
                        break

            # get all available new data from serial
            try:
                read = self.serial_read_function(1_000_000)
            except OSError:
                logger.error("Couldn't read from connection. Disconnecting.")
                self.disconnect_from_serial()
                continue

            if len(read) == 0:
                continue

            self.bytes_received += len(read)

            # emit received serial data over zmq
            if self.zmq_forwarding_conn is not None:
                self.zmq_forwarding_conn.send(read)

            # process new + leftover data into messages
            buffer = leftover_bytes + read  # prepend previous bytes
            timestamp = time.time_ns() // 1000
            if self.binary:
                leftover_bytes = self.process_binary(buffer, timestamp)
            else:
                leftover_bytes = self.process_ascii(buffer, timestamp)

    def process_ascii(self, buffer, timestamp):
        try:
            new_data = buffer.decode("ascii")
        except Exception as e:
            logger.error(f"Couldn't decode data as ascii, exception: {e}")
            return bytearray(b"")

        if self.message_delimiter not in new_data:
            return buffer

        messages = new_data.split(self.message_delimiter)

        # drop last message (if new_data ends in delimiter, last message will be empty "",
        # and if new_data doesnt end in delimiter, we dont know its a complete message)
        num_bytes_read = len(buffer) - len(messages[-1])
        messages = messages[:-1]

        # convert to tuple containing numeric values as strings
        message_numbers = [tuple(self.numeric_rx.findall(msg)) for msg in messages]
        message_lengths = [len(nums) for nums in message_numbers]

        # drop any invalid (wrong-length) messages and warn about it
        bad_message_lengths = [length for length in message_lengths if length != self.ascii_num_streams]
        if len(bad_message_lengths) > 0:
            logger.warning(f"Dropping {len(bad_message_lengths)} bad messages. Lengths: {set(bad_message_lengths)}")
        message_numbers = [x for x in message_numbers if len(x) == self.ascii_num_streams]

        structured_data = np.array([(*nums, timestamp) for nums in message_numbers], dtype=self.complete_message_dtype)

        self.message_buffer.add(structured_data)

        # if we're saving to file, dump new data to file here.
        if self.save_file is not None:
            for msg in message_numbers:
                line = ",".join(msg)
                if self.save_timestamps:
                    line += "," + str(timestamp)
                self.save_file.write(line + "\n")

        # return leftover (unprocessed) data
        return buffer[num_bytes_read:]

    def process_binary(self, buffer, timestamp):
        new_data = np.array(buffer)
        expected_length = self.binary_dtype.itemsize  # does not include delimiter byte

        # find delimiters so we can define message bytes based on them
        delimiter_indices = np.where(new_data == self.message_delimiter)[0]
        if len(delimiter_indices) == 0:
            # if we've got more than expected_size w no delimiter, it's all invalid, ignore it
            if len(new_data) > expected_length:
                return bytearray(b"")
            else:
                return buffer

        valid_delimiter_indices = [delimiter_indices[0]]
        for i in delimiter_indices[1:]:
            if i - valid_delimiter_indices[-1] >= expected_length:
                valid_delimiter_indices += [i]

        valid_mask = np.full((len(new_data)), False)
        # assume delimiters at right end (so that we can parse a message with a header w/o waiting for next)
        break_indices = np.concatenate([valid_delimiter_indices, [len(new_data)]])
        for i, j in zip(break_indices[:-1], break_indices[1:]):
            if j - i == expected_length + 1:
                valid_mask[(i + 1) : j] = True

        if sum(valid_mask) == 0:
            # we received either an incomplete packet, or 1+ invalid packets
            return buffer[valid_delimiter_indices[-1] :]

        # messages to streams (e.g. 8 byte message to 2 uint8s, 1 uint16, 1 uint32):
        # the numpy parser is amazing, see docs:
        #   https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
        structured_data = np.frombuffer(new_data[valid_mask], self.binary_dtype)

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
        self.message_buffer.add(structured_data)

        last_valid_byte = np.where(valid_mask)[0].max()
        return buffer[last_valid_byte + 1 :]

    def get_new_messages(self):
        return self.message_buffer.read_new()

    def get_bytes_received(self):
        return self.bytes_received

    def close(self):
        self.running = False
        self.stop_saving()
        self.disconnect_from_serial()
