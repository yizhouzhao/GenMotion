

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

import c4d
import socket
import time

def simpleServer():
    HOST = '127.0.0.1'
    PORT = 3005
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

    def Main(self):
        # Put in your code here
        # which you want to run
        # in another thread
        print("start server......")
        simpleServer()

if __name__ == '__main__':

    # server_thread = threading.C4DThread(target=simpleServer)
    # server_thread.start()

    thread = UserThread()
    thread.Start()

###### --------------------------------------------
###                  c4d controller
###### --------------------------------------------

class C4DController():
    def __init__(self, PORT = 12345):
        # connect to Maya server
        HOST = '127.0.0.1'  # Symbolic name meaning the local host
        #PORT = PORT  # Arbitrary non-privileged port

        ADDR = (HOST, PORT)

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def SendCommand(self, command: str):
        command = command.encode()
        self.client.sendall(bytes(command, 'utf-8'))

    def RegisterEvent(self):
        self.client.sendall(bytes("c4d.EventAdd()", 'utf-8'))

    def SetOneKeyFrame(self, curve_name: str, frame: int, value: float):
        # Retrieves the current time
        self.SendCommand("keyTime = c4d.BaseTime({}, doc.GetFps())".format(str(frame)))
        self.SendCommand("added = {}.AddKey(keyTime)".format(curve_name))
        self.SendCommand("added[\"key\"].SetValue({}, 0)".format(curve_name))
        self.SendCommand("added[\"key\"].SetInterpolation({},c4d.CINTERPOLATION_SPLINE)".format(curve_name))
        self.SendCommand("{}.SetKeyDefault(doc, added_X[\"nidx\"])".format(curve_name))
        
        # Sets the key to default status AutoTangent etc...

# ---- animation

import c4d
# set documentation
doc = c4d.documents.GetActiveDocument()
root_name = "f_avg_root"
root = doc.SearchObject(root_name)
# Creates the track in memory. Defined by it's DESCID    
root_trX = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)))
root_trY = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)))
root_trZ = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0)))
# Creates the track in memory. Defined by it's DESCID    
root_rX = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)))
root_rY = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)))
root_rZ = c4d.CTrack(root, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0)))
# Gets Curves for the track
root_curveX = root_trX.GetCurve()
root_curveY = root_trY.GetCurve()
root_curveZ = root_trZ.GetCurve()
# Gets Curves for the track
root_curveRX = root_rX.GetCurve()
root_curveRY = root_rY.GetCurve()
root_curveRZ = root_rZ.GetCurve()

joint_name = "f_avg_" + SMPL_H_SKELETON[0]

def RegisterJoints(joint):
    register_command = [
        "{} = doc.SearchObject(\"{}\")".format(joint,joint),
        "{}_rX = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)))".format(joint, joint),
        "{}_rY = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)))".format(joint, joint),
        "{}_rZ = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0)))".format(joint, joint),
        "{}_curveRX = {}_rX.GetCurve()".format(joint, joint),
        "{}_curveRY = {}_rY.GetCurve()".format(joint, joint),
        "{}_curveRZ = {}_rZ.GetCurve()".format(joint, joint),
    ]



# Retrieves the current time
keyTime = c4d.BaseTime(0, doc.GetFps())

# Adds the keys
added_X = root_curveX.AddKey(keyTime)
added_Y = root_curveY.AddKey(keyTime)
added_Z = root_curveZ.AddKey(keyTime)

# Sets the value of the key
added_X["key"].SetValue(root_curveX, 0)

# Changes it's interpolation
added_X["key"].SetInterpolation(root_curveX,c4d.CINTERPOLATION_SPLINE)

# Sets the key to default status AutoTangent etc...
root_curveX.SetKeyDefault(doc, added_X["nidx"])


# Retrieves the current time
keyTime = c4d.BaseTime(10, doc.GetFps())

# Adds the keys
added_X = root_curveX.AddKey(keyTime)
added_Y = root_curveY.AddKey(keyTime)
added_Z = root_curveZ.AddKey(keyTime)

# Sets the value of the key
added_X["key"].SetValue(root_curveX, 100)

# Changes it's interpolation
added_X["key"].SetInterpolation(root_curveX,c4d.CINTERPOLATION_SPLINE)

# Sets the key to default status AutoTangent etc...
root_curveX.SetKeyDefault(doc, added_X["nidx"])


# Inserts track to the object
root.InsertTrackSorted(root_trX)

# Inserts the object in document
doc.InsertObject(root)

# Pushes an update event to Cinema 4D
c4d.EventAdd()    