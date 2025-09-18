import json
import numpy as np


def read_serial_capture_binary(filename):
    """Load a splot serial_capture data file, which begins with a json header,
    immediately followed by binary data (in the format specified in the header).

    Returns:
        a list containing:
         - either a structured numpy recarray (if structured==True), or a
        regular ndarray (structured==False).
         - the names of the individual scalar fields
    """

    with open(filename, "rb") as file:
        # assume header can't be longer than 4kb
        chunk_string = file.read(4096).decode("utf-8", errors="ignore")
        decoder = json.JSONDecoder()
        json_object, binary_start_index = decoder.raw_decode(chunk_string)

        # json requires lists, numpy requires tuples; convert to tuples
        dtype = [tuple(x) for x in json_object["dtype"]]

        binary_dtype = np.dtype(dtype)
        columns = ["delimiter"] + json_object["series_names"]
        columns = [name if name != "" else f"var{i}" for i, name in enumerate(columns)]

        file.seek(binary_start_index)
        structured_data = np.fromfile(file, binary_dtype)

    return structured_data, columns
