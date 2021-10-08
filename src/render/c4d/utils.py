# Select object by name
def SelectObjectByName(obj_name):
    obj = doc.SearchObject(obj_name)
    doc.SetSelection(obj, c4d.SELECTION_NEW)

# Select relative/absolute position, rotation, scale
# obj.SetRelPos(c4d.Vector(0,100,0))

# Get keyframe time 
# doc.GetTime().Get()

# Set keyframe time
# doc.SetTime(c4d.BaseTime(50,30))


# Animation reference
# https://forums.cgsociety.org/t/c4d-animation-via-python/1546556/3
# https://plugincafe.maxon.net/topic/11698/beginner-how-can-i-set-a-key-frame-and-value-to-a-cube-by-python/2

# -----------------------------------------------------------------
# Open socket
import socket
import threading
import socketserver

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print("Received: {}".format(response))

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 10001

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)

        #client(ip, port, "Hello World 1")
        #client(ip, port, "Hello World 2")
        #client(ip, port, "Hello World 3")

        # server.shutdown()

# -----------------------------------------------------------------