# splot

## Purpose
splot is intended to be a swiss army knife for plotting and assessing data being passed over a serial connection. Currently, it supports data being sent via a computer serial port (e.g. `COM1` on windows, or `/dev/ttyusb0` on mac or linux), or over a network via a tcp/udp socket.

Data is often sent in different formats, so splot is intended to have the flexibility to parse various message encodings. Currently it supports:
1. *binary encoded messages* with a single-byte delimiter between messages. The user can specify which bytes of the message belong to which data series. For example, if every message consists of a 0 header/delimiter byte, a 2-byte unsigned integer, a 4-byte float, an 8-byte double, and a signed 1-byte integer, one could specify this as "u1,u2,f4,f8,i1" (see https://numpy.org/doc/stable/user/basics.rec.html#structured-datatypes for details and more examples). 
2. *ascii encoded messages*, with a single-byte delimiter between messages (typically a newline, '\n'). The user specifies the number of data series expected in each message `n`, and the first `n` numbers in each message will be plotted. If less than `n` values are present, the remaining values are filled with NaNs.

## How do I install it?
```sh
pip install "git+https://github.com/starfishneuroscience/splot.git"
```

and then to run it, run `splot` at the command line.

## Possible future directions:
- UI:
    - allow disabling of certain streams for plotting
    - filters and averaging of signals
- Error detection:
    - add option for setting a 'counter' field and create a 'data-dropped flag' or online estimator for dropped messages
    - add checksum checking for each message, and an indicator for how often bad data is seen
- Message parsing:
    - allow single messages to contain multiple values for a single series (e.g., message consists of 3 consecutive readings from one sensor, followed by 1 reading from another lower-speed sensor).
    - allow multi-byte message-delimiters
    - handle uart with different frame sizes (e.g., 9- or 10-bit frames)
    - handle other serialization formats, e.g. JSON or protobuf

## Changelog:
 - PR #1:
    - handle sockets, not just serial ports. you can type in a tcp/udp address into "source" and it will connect automatically.
    - handle ascii messages with numerical fields
    - add mouseover documentation for dtype_string
    - in binary, message size should be inferred from dtype_string
    - make different config options available on UI when parsing different message types
 - PR #2:
    - Restructure as python package to allow easy install. Still works with local editable install (`pip install -e <repo path>`).
