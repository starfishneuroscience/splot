import json
import numpy as np
from numpy.lib import recfunctions as rfn


def read_serial_capture_binary(filename, structured=False):
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

        binary_dtype = np.dtype(json_object["dtype_string"])
        columns = ["delimiter"] + json_object["series_names"]
        columns = [name if name != "" else f"var{i}" for i, name in enumerate(columns)]

        file.seek(binary_start_index)
        structured_data = np.fromfile(file, binary_dtype)

    if structured:
        return structured_data, columns

    # flatten a potentially nested data type down to scalars
    data = rfn.structured_to_unstructured(structured_data)
    data = data[:, 1:]  # drop 1st column, it's the delimiter which is constant by definition

    return data, columns
