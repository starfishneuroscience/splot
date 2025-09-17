import socket
import numpy as np
import time

server_socket = socket.socket()
server_socket.bind(("localhost", 12344))
server_socket.listen()
while True:
    conn, address = server_socket.accept()  # accept new connection
    counter = 0
    while True:
        delimiter = 42
        x1 = int(np.random.rand() * 255)
        x2 = int(np.sin(time.time() * 1) * 2**15)
        x3 = int(np.sin(time.time() * 10) * 2**15)
        x4 = int(np.sin(time.time() * 100) * 2**15)
        x5 = x1 + x2 + x3 + x4

        # u1, u1, 3i2, i4
        data = delimiter.to_bytes(1) + x1.to_bytes(1)
        data += x2.to_bytes(2, signed=True, byteorder="little")
        data += x3.to_bytes(2, signed=True, byteorder="little")
        data += x4.to_bytes(2, signed=True, byteorder="little")
        data += x5.to_bytes(4, signed=True, byteorder="little")

        try:
            conn.send(data)
        except Exception as e:
            print(e)
            break

        # print(f"Sent: {data}")
        counter += 1
        time.sleep(0.00005)
