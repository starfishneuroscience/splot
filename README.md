# splot

## Purpose
Swiss army knife to plot and assess data being passed over a serial interface.

## Possible future directions:
- UI:
    - allow disabling of certain streams for plotting
    - filters and averaging of signals
- Error detection:
    - add option for setting a 'counter' field and create a 'data-dropped flag' or online estimator for dropped messages
    - add checksum checking for each message, and an indicator for how often bad data is seen
- Message parsing:
    - in binary, allow multi-byte message-delimiters
    - handle uart with different frame sizes
    - handle protobuf
    - handle json


## Changelog:
 - PR #1:
    - handle sockets, not just serial ports. you can type in a tcp/udp address into "source" and it will connect automatically or fail and alert you
    - handle ascii messages with numerical fields
    - add mouseover documentation for dtype_string
    - in binary, message size should be inferred from dtype_string
    - make different config options available on UI when parsing different message types
