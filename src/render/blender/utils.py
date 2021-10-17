import bpy
from tqdm import tqdm
import numpy as np
from typing import Any, List, Optional
from dataset.hdm05_params import ASF_JOINT2DOF
from dataset.amass_params import SMPL_X_SKELTON2

def import_fbx(file_path: str):
    bpy.ops.import_scene.fbx(filepath=file_path)
        
def import_smplx(gender: str):
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = gender
    bpy.ops.scene.smplx_add_gender()

def clear_all_animation():
    bpy.context.active_object.animation_data_clear()

def set_joint_location_keyframe(joint: Any, data: List[float], frame: int):
    joint.location = data
    joint.keyframe_insert(data_path="location", frame=frame)

def set_joint_rotation_keyframe(joint: Any, data: List[float], frame: int, mode: str ="euler", axis: Optional[List[str]]=None):
    if mode == "euler":
        if axis:
            for i in range(len(axis)):
                if axis[i] == "rx":
                    joint.rotation_euler[0] = data[i]
                elif axis[i] == "ry":
                    joint.rotation_euler[1] = data[i]
                elif axis[i] == "rz":
                    joint.rotation_euler[2] = data[i]
        else:
            joint.rotation_euler = data
        joint.keyframe_insert(data_path="rotation_euler", frame=frame)
    elif mode == "quaternion":
        joint.rotation_quaternion = data
        joint.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def set_amass_animation(data):
    # set frame properties
    total_frame = int(data["mocap_time_length"] * data["mocap_frame_rate"])
    bpy.data.scenes["Scene"].frame_start = 0
    bpy.data.scenes["Scene"].frame_end = total_frame
    bpy.data.objects['SMPLX-neutral'].select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.context.scene.render.fps = round(float(data["mocap_frame_rate"]))
    
    # change rotation mode to xyz-euler
    character = bpy.data.objects["SMPLX-neutral"]
    root = character.pose.bones["root"]
    root.rotation_mode = "XYZ"
    for i, joint_name in SMPL_X_SKELTON2.items():
        character.pose.bones[joint_name].rotation_mode = "XYZ"
        
    for frame in tqdm(range(total_frame)):
        # set root location and rotation
        set_joint_location_keyframe(root, data["trans"][frame], frame)
        set_joint_rotation_keyframe(root, data["root_orient"][frame], frame)
        
        # set skeleton joints
        for i, joint_name in SMPL_X_SKELTON2.items():
            set_joint_rotation_keyframe(character.pose.bones[joint_name], data["poses"][frame][i * 3: (i + 1) * 3], frame)  

def set_amc_animation(amc_file_path: str):
    with open(amc_file_path, "rb") as f:
        cur_frame = 0
        character = bpy.data.objects["Armature"]
        character.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")
        for line in tqdm(f.readlines()):
            if line.strip().isdigit():
                cur_frame = int(line)
            elif cur_frame > 0 and cur_frame % 100 == 0:
                data = line.decode("utf-8").strip().split()
                joint_name = data[0]
                joint = character.pose.bones[joint_name]
                character.pose.bones[joint_name].rotation_mode = "XYZ"
                if joint_name == "root":
                    set_joint_location_keyframe(joint, np.float_(data[1:4]), cur_frame)
                    set_joint_rotation_keyframe(joint, np.float_(data[4:]) * np.pi / 180, cur_frame)
                else:
                    set_joint_rotation_keyframe(joint, np.float_(data[1:]) * np.pi / 180, cur_frame, axis=ASF_JOINT2DOF[joint_name])