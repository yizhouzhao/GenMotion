# import c4d
import socket
import time

def SimpleServer(HOST = '127.0.0.1', PORT = 3005):
    """
    Generate a simple TCP socket server in Cinema 4D

    Args:
        HOST (str): local host address;
        POST (int): server port;
    """
    sock = socket.socket()
    sock.bind((HOST,PORT))

    sock.listen()
    conn, addr = sock.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(10240).decode("utf-8") 
            #if not data:
            #    break
            if data:
                try:
                    print("received:", data) 
                    exec(data)
                    
                    # conn.sendall("".encode()) # data.encode()
                except Exception as e:
                    print(e)
                    print("wrong command:", data)
                    conn.sendall("wrong command".encode()) # data.encode()

            #time.sleep(1)
    
    print("Disconnected")

class UserThread(c4d.threading.C4DThread):
    """
    A user thread to hold the socket server in Cinema 4D
    """
    def Main(self):
        # Put in your code here
        # which you want to run
        # in another thread
        print("start server......")
        SimpleServer()

if __name__ == '__main__':

    # server_thread = threading.C4DThread(target=simpleServer)
    # server_thread.start()

    thread = UserThread()
    thread.Start()
