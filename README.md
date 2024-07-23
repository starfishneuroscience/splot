# splot

## Purpose
splot is intended to be a performant, swiss army knife for plotting and assessing data being passed over a serial connection. Currently, it supports data being sent via a computer serial port (e.g. `COM1` on windows, or `/dev/ttyusb0` on mac or linux), or over a network via a tcp/udp socket.

Data is often sent in different formats, so splot is intended to have the flexibility to parse various message encodings. Currently it supports:
1. *binary encoded messages* with a single-byte delimiter between messages. The user can specify which bytes of the message belong to which data series. For example, if every message consists of a 0 header/delimiter byte, a 2-byte unsigned integer, a 4-byte float, an 8-byte double, and a signed 1-byte integer, one could specify this as "u1,u2,f4,f8,i1" (see https://numpy.org/doc/stable/user/basics.rec.html#structured-datatypes for details and more examples).
2. *ascii encoded messages*, with a single-byte delimiter between messages (typically a newline, '\n'). The user specifies the number of data series expected in each message `n`, and the first `n` numbers in each message will be plotted. If less than `n` values are present, the remaining values are filled with NaNs.

## Screenshots (what does it look like?)
https://github.com/starfishneuroscience/splot/assets/108433203/f57e02ae-caa8-45b6-ac78-059420b13914

## How do I install it?
```sh
pip install "git+https://github.com/starfishneuroscience/splot.git"
```
and then to run it, run `splot` at the command line.

## License
See the `LICENSE` file.

## Performance
splot has been tested with 12M baud serial connections with net data rates exceeding 1 MB/sec (66% utilization). Over a TCP connection, splot can smoothly parse and plot 1.4 MB/sec of ascii data on a Macbook pro (M1).

## Saving data
splot can save binary data or text/ascii data.

When receiving binary data, splot can save a file consisting of a json header including the data format (as specified as a numpy dtype string) and the data series names, followed by the binary data. It will only save complete messages; incomplete messages will be ignored. A helper function `read_serial_capture_binary` can be used to read this file and format it as a structured recarray, or an (unstructured) ndarray , e.g.,
```py
import splot
arr, series_names = splot.read_serial_capture_binary(
    filename='/home/x/data/serialcapture_2024-06-17_17-49-54.bin',
    structured=True,
)
```

When receiving ascii data, splot simply records it as a csv file.

### Possible future directions
- Development:
    - Implementing a good test infrastructure for automated testing (possibly in CI)
- UI:
    - switch between stacked plots and single plot with overlaid series
- Data processing:
    - filters and averaging of signals (averaging is actually built into pyqtgraph; right click on a plot!)
    - Oscilloscope mode / triggered plotting
    - Error detection:
        - add option for setting a 'counter' field and create a 'data-dropped flag' or online estimator for dropped messages
        - add checksum checking for each message, and an indicator for how often bad data is seen
- Message parsing:
    - allow single messages to contain multiple values for a single series (e.g., message consists of 3 consecutive readings from one sensor, followed by 1 reading from another lower-speed sensor).
    - allow multi-byte message-delimiters for binary (already supported for ascii)
    - handle uart with different frame sizes (e.g., 9- or 10-bit frames)
    - handle other serialization formats, e.g. JSON or protobuf
    - long vs wide data formats
- Serial interface:
    - Serial send functionality (possibly out of scope?)

## Similar projects
There are a number of similar projects out there from which splot takes inspiration, for example:
 - https://github.com/CieNTi/serial_port_plotter
 - https://github.com/nathandunk/BetterSerialPlotter
 - https://github.com/mich-w/QtSerialMonitor
 - https://github.com/hacknus/serial-monitor-rust

## Changelog:
- PR #14:
    - Changing data formats didn't work while connected, but works now.
- PR #8:
    - enable saving data
    - revert to same color on all plots
    - add name for each plot
    - make names, visility, and type (analog or bitmask) of plots persistent
- PR #6:
    - Pause now respected when connecting/disconnecting
    - Unique colors for each plot
    - Allow disabling of certain streams for plotting
    - Remove margins between plots and x-axes of all plots except bottom one, so that plots can be bigger
    - Allow plotting of data series as rasters (interpreting values as bit-masks)
 - PR #4:
    - Pause didn't pause processing, just updating the plots. Pause now inhibits stream processor, so plot buffers dont update.
    - Show vertical bar for current plot position
    - Color scheme was bad for 'light' system theme. Now correctly pulls theme colors and uses them.
    - Add persistent settings when closing and re-opening app via QSettings.
 - PR #3:
    - Fix some bad bugs in ASCII parsing (would re-read same buffer if no new data were present! wouldnt parse floats correctly!)
    - UI cleanup: alignment/sizing, make plot colors match system theme.
    - Add example TCP server for testing ascii parsing.
 - PR #2:
    - Restructure as python package to allow easy install. Still works with local editable install (`pip install -e <repo path>`).
 - PR #1:
    - handle sockets, not just serial ports. you can type in a tcp/udp address into "source" and it will connect automatically.
    - handle ascii messages with numerical fields
    - add mouseover documentation for dtype_string
    - in binary, message size should be inferred from dtype_string
    - make different config options available on UI when parsing different message types
