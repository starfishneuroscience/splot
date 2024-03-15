import socket
import numpy
import time

server_socket = socket.socket()
server_socket.bind(('localhost', 1234))
server_socket.listen()
while True:
    conn, address = server_socket.accept()  # accept new connection
    counter = 0
    while True:
        rand = numpy.random.rand() - 0.5
        data = f"{counter}, {rand}, {rand:.5e}units {round(rand*1000)} junk\r\n"

        try:
            conn.send(data.encode('ascii'))
        except Exception as e:
            print(e)
            break

        print(f"Sent: {data}")
        counter += 1
        time.sleep(.0001)
