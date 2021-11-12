import socket

class C4DController():
    def __init__(self, PORT = 3014):
        # connect to c4d server
        HOST = '127.0.0.1'  # Symbolic name meaning the local host
        #PORT = PORT  # Arbitrary non-privileged port

        ADDR = (HOST, PORT)

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def SendCommand(self, command: str):
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