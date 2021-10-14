import bpy
from tqdm import tqdm
import numpy as np

# smpl-x skeleton dict
SMPL_X_JOINTS = { 
    0: 'pelvis',                       
    1: 'left_hip',             
    2: 'right_hip',
    3: 'spine1', 
    4: 'left_knee',      
    5: 'right_knee',
    6: 'spine2', 
    7: 'left_ankle', 
    8: 'right_ankle',
    9: 'spine3',  
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck', 
    13: 'left_collar',      
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder', 
    17: 'right_shoulder',
    18: 'left_elbow',  
    19: 'right_elbow',
    20: 'left_wrist',  
    21: 'right_wrist',
    22: 'jaw',      
    23: 'left_eye_smplhf',    
    24: 'right_eye_smplhf',
    25: 'left_index1',  
    26: 'left_index2',  
    27: 'left_index3',
    28: 'left_middle1', 
    29: 'left_middle2', 
    30: 'left_middle3',
    31: 'left_pinky1',  
    32: 'left_pinky2',  
    33: 'left_pinky3',
    34: 'left_ring1',   
    35: 'left_ring2',   
    36: 'left_ring3',
    37: 'left_thumb1',  
    38: 'left_thumb2',  
    39: 'left_thumb3',
    40: 'right_index1',  
    41: 'right_index2',  
    42: 'right_index3',
    43: 'right_middle1', 
    44: 'right_middle2', 
    45: 'right_middle3',
    46: 'right_pinky1',  
    47: 'right_pinky2',  
    48: 'right_pinky3',
    49: 'right_ring1',   
    50: 'right_ring2',   
    51: 'right_ring3',
    52: 'right_thumb1',  
    53: 'right_thumb2',  
    54: 'right_thumb3'
}

def import_fbx(file_path: str):
    bpy.ops.import_scene.fbx(filepath=file_path)
        
def import_smplx(gender: str):
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = gender
    bpy.ops.scene.smplx_add_gender()

def clear_all_animation():
    bpy.context.active_object.animation_data_clear()

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
    for i, joint_name in SMPL_X_JOINTS.items():
        character.pose.bones[joint_name].rotation_mode = "XYZ"
        
    for frame in tqdm(range(total_frame)):
        # set root location and rotation
        root.location = data["trans"][frame]
        root.keyframe_insert(data_path="location", frame=frame)
        root.rotation_euler = data["root_orient"][frame]
        root.keyframe_insert(data_path="rotation_euler", frame=frame)
        
        # set skeleton joints
        for i, joint_name in SMPL_X_JOINTS.items():
            joint = character.pose.bones[joint_name]
            joint.rotation_euler = data["poses"][frame][i * 3: (i + 1) * 3]
            joint.keyframe_insert(data_path="rotation_euler", frame=frame)   