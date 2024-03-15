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
        rand = numpy.random.rand(3)
        data = f"{counter}, {rand[0]}, {rand[1]}, {rand[2]}\r\n"

        try:
            conn.send(data.encode())
        except Exception as e:
            print(e)
            break

        print(f"Sent: {data}")
        counter += 1
        time.sleep(.1)
