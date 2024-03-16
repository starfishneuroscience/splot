import socket
import numpy as np
import time

server_socket = socket.socket()
server_socket.bind(('localhost', 12345))
server_socket.listen()
while True:
    conn, address = server_socket.accept()  # accept new connection
    counter = 0
    while True:
        x1 = np.random.rand() - 0.5
        x2 = np.sin(time.time() * 100)
        x3 = np.sin(time.time() * 10) + 0.1*x1
        x4 = round(time.time() * 10) % 2
        data = f"{counter}, {x1}, {x2:.5e}units {x3} junk {x4}\r\n"

        try:
            conn.send(data.encode('ascii'))
        except Exception as e:
            print(e)
            break

        print(f"Sent: {data}")
        counter += 1
        time.sleep(.0001)
