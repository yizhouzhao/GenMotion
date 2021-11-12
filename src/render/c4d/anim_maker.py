import time

from .utils import C4DController
from .params import *

class C4DAnimMaker():
    def __init__(self, PORT) -> None:
        self.PORT = PORT
        self.controller = C4DController(PORT = self.PORT)

    def Initialize(self):
        preparation_command = ["import c4d",
            "# set documentation",
            "doc = c4d.documents.GetActiveDocument()"
            ]
        self.controller.SendCommand(preparation_command)

    def RegisterJointPosition(self, joint):
        register_command = [f"{joint} = doc.SearchObject({joint})",
            f"{joint}.SetRotationOrder(5)",
            "# Creates the track in memory. Defined by it's DESCID    ",
            f"{joint}_trX = c4d.CTrack({joint}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)))",
            f"{joint}_trY = c4d.CTrack({joint}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)))",
            f"{joint}_trZ = c4d.CTrack({joint}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_POSITION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0)))",
            f"{joint}_curveX = {joint}_trX.GetCurve()",
            f"{joint}_curveY = {joint}_trY.GetCurve()",
            f"{joint}_curveZ = {joint}_trZ.GetCurve()",
        ]

        self.controller.SendCommand("\n".join(register_command))


    def RegisterJointRotation(self, joint):
        register_command = [
            "{} = doc.SearchObject(\"{}\")".format(joint,joint),
            "{}.SetRotationOrder(5)".format(joint),
            "{}_rX = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)))".format(joint, joint),
            "{}_rY = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)))".format(joint, joint),
            "{}_rZ = c4d.CTrack({}, c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0), c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0)))".format(joint, joint),
            "{}_curveRX = {}_rX.GetCurve()".format(joint, joint),
            "{}_curveRY = {}_rY.GetCurve()".format(joint, joint),
            "{}_curveRZ = {}_rZ.GetCurve()".format(joint, joint),
        ]
        
        self.controller.SendCommand("\n".join(register_command))

    def SetOneKeyFrame(self, curve_name: str, frame: int, value: float):
        # Retrieves the current time
        frame_command = ["keyTime = c4d.BaseTime({}, doc.GetFps())".format(str(frame)),
        "added = {}.AddKey(keyTime)".format(curve_name),
        "added[\"key\"].SetValue({}, {})".format(curve_name, str(value)),
        "added[\"key\"].SetInterpolation({},c4d.CINTERPOLATION_SPLINE)".format(curve_name),
        "{}.SetKeyDefault(doc, added[\"nidx\"])".format(curve_name)]
        
        # print
        self.controller.SendCommand("\n".join(frame_command))
    

class C4DSMPLHAnimMaker(C4DAnimMaker):
    def __init__(self, PORT, joint_prefix = "f_avg_", root_name = "root") -> None:
        super().__init__(PORT)
        self.joint_prefix = joint_prefix
        self.root = root_name

    def RegisterJoints(self, register_joint = True):
        if register_joint:
            self.RegisterJointPosition(self.root)
            self.RegisterJointRotation(self.root)

        

