# This code was tested on pybullet=3.2.5
import os
import pybullet as p
import pybullet_data
import numpy as np
from numpy.random import uniform
import math

from typing import Tuple


def normalized_angle(x: float) -> float:
    """Map angle in range (-pi,pi)."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_joint_limits(body_id, joint_ids):
    """Query joint limits as (lo, hi) tuple, each with length same as
    `joint_ids`."""
    joint_limits = []
    for joint_id in joint_ids:
        joint_info = p.getJointInfo(body_id, joint_id)
        joint_limit = joint_info[8], joint_info[9]
        joint_limits.append(joint_limit)
    joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
    return joint_limits


class Robot:
    '''
    Robot configuration and motions
    '''
    def __init__(self, uid, control_dt):
        # Simulation config
        self.uid = uid
        self.control_dt = control_dt

        # Kinematics config
        self.joint_index_last             = 13
        self.joint_index_endeffector_base = 7
        self.joint_indices_arm:np.ndarray = np.array(range(1, self.joint_index_endeffector_base)) # Without base
        self.joint_limits:np.ndarray      = get_joint_limits(uid, range(self.joint_index_last+1))
        self.joint_range:np.ndarray       = (np.array(self.joint_limits[1]) - np.array(self.joint_limits[0]))
        
        self.rest_pose:np.ndarray = np.array((
            0.0,    # Base  (Fixed)
            0.0,    # Joint 1
            -2.094, # Joint 2
            1.57,   # Joint 3
            -1.047, # Joint 4
            -1.57,  # Joint 5
            0,      # Joint 6
            0.0,    # EE Base (Fixed)
            0.785,  # EE Finger
        ))

        # Motion config
        self.CLOSED_LOOP = True
        self.motion_durations = [0.1, 0.5, 0.3, 0.5]    # Duration for each motion
        self.reset()                                    # Reset with init


    def reset(self):
        # Init motion
        self.current_pose:np.ndarray = np.copy(self.rest_pose)     # The current pose that "the controller should follow"
        
        for i in self.joint_indices_arm:
            p.resetJointState(self.uid, i, self.current_pose[i])
        self._finger_control(self.current_pose[self.joint_index_endeffector_base+1])

        self.motion_trajectory = []                     # The future trajectory
        self.motion_time = 0.
        self.motion_current = 0


    def set_object(self, obj_pos, obj_yaw):
        # Set gripper pos & orn
        self.obj_pos = list(obj_pos)
        self.obj_pos[2] += 0.200                        # End-effector will be at 20 cm above the object                       
        self.obj_pos_ready = list(obj_pos) 
        self.obj_pos_ready[2] += 0.250                  # This is to prevent collision during the motion
        self.obj_orn = p.getQuaternionFromEuler(
            [-math.pi, 0, -math.pi+obj_yaw-math.pi/2]   # Downward [roll, pitch, yaw + obj_yaw]
        )  

        
    def execute(self):
        # Init goal pose from previous pose
        goal_pose = np.copy(self.current_pose) # SHOULD BE COPIED
        
        # Set goal pose of current motion
        # 0. Rest position
        # 1. Move end-effector to the object
        # 2. Grap
        # 3. Rest position
        if self.motion_current == 0:
            goal_pose = np.copy(self.rest_pose)                                             # Update goal_pose to rest_pose
        if self.motion_current == 1:                                                                
            ik_pose = p.calculateInverseKinematics(self.uid,                                # Get InverseKinematics to the target
                                                   self.joint_index_endeffector_base,     
                                                   self.obj_pos_ready, self.obj_orn)   
            # NOTE: This is awkward... 
            # but the index of joint and ik_pose does not match, 
            # as the ik solver ignores fixed joints...
            goal_pose[self.joint_indices_arm] = ik_pose[:len(self.joint_indices_arm)]       # Update goal_pose to the target
        if self.motion_current == 2:
            ik_pose = p.calculateInverseKinematics(self.uid,                                # Get InverseKinematics to the target
                                                   self.joint_index_endeffector_base,     
                                                   self.obj_pos, self.obj_orn)
            goal_pose[self.joint_indices_arm] = ik_pose[:len(self.joint_indices_arm)]       # Update goal_pose to the target
            goal_pose[self.joint_index_endeffector_base + 1] = 0                            # Close the gripper
        if self.motion_current == 3:
            goal_pose[self.joint_indices_arm] = self.rest_pose[self.joint_indices_arm]      # Update goal_pose to rest_pose
            goal_pose[self.joint_index_endeffector_base + 1] = 0                            # Close the gripper
        

        # Generating the trajectory of each joint
        if self.CLOSED_LOOP:    # Regenerate everytime if closed-loop
            self.motion_trajectory = self._interpolate_trajectory(
                self.current_pose, goal_pose, 
                self.motion_time, self.motion_durations[self.motion_current], 
                self.control_dt)
        else:                   # Generate once if open-loop (may be inaccurate)
            if self.motion_time == 0:
                self.motion_trajectory = self._interpolate_trajectory(
                    self.current_pose, goal_pose, 
                    self.motion_time, self.motion_durations[self.motion_current], 
                    self.control_dt)

        # Take next pose from trajectory
        if len(self.motion_trajectory) != 0:
            self.current_pose = self.motion_trajectory.pop(0)

        # Feed pose value to the controller
        p.setJointMotorControlArray(self.uid, self.joint_indices_arm,                               # Arm
                                    p.POSITION_CONTROL, 
                                    self.current_pose[self.joint_indices_arm])    
        self._finger_control(self.current_pose[self.joint_index_endeffector_base + 1])              # Endeffector

        # Update simulation time
        self.motion_time += self.control_dt
        # Motion transition
        if self.motion_time > self.motion_durations[self.motion_current]:
            self.motion_current += 1
            self.motion_time = 0

        # Return True if task is done
        if self.motion_current >= len(self.motion_durations):
            return True

        # Return False if task is undergoing
        return False


    def _finger_control(self, target):
        '''
        Control the finger joints to target position.
        This is to imitate the mimic joint in ROS.
        '''
        # Just a hardcoded control...
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+1, p.POSITION_CONTROL, 
                                targetPosition = target)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+4, p.POSITION_CONTROL, 
                                targetPosition = target)
        # Get the current joint pos and vel
        finger_left = p.getJointState(self.uid, self.joint_index_endeffector_base+1)
        finger_right = p.getJointState(self.uid, self.joint_index_endeffector_base+4)
        # Propagate it to the other joints.
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+2, p.POSITION_CONTROL, 
                                targetPosition = finger_left[0], 
                                targetVelocity = finger_left[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+3, p.POSITION_CONTROL, 
                                targetPosition = finger_left[0], 
                                targetVelocity = finger_left[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+5, p.POSITION_CONTROL, 
                                targetPosition = finger_right[0], 
                                targetVelocity = finger_right[1],
                                positionGain=1.2)
        p.setJointMotorControl2(self.uid, self.joint_index_endeffector_base+6, p.POSITION_CONTROL, 
                                targetPosition = finger_right[0], 
                                targetVelocity = finger_right[1],
                                positionGain=1.2)


    @staticmethod
    def _interpolate_trajectory(current:np.ndarray, goal:np.ndarray, motion_time:float, motion_duration:float, control_dt:float) -> Tuple[np.ndarray, ...]:
        '''
        This function returns linear-interpolated (dividing straight line)
        trajectory between current and goal pose.
        Acc, Jerk is not considered.
        '''
        # Interpolation steps
        steps = math.ceil((motion_duration-motion_time)/control_dt)
        
        # Calculate difference
        delta = [ goal[i] - current[i] for i in range(len(current)) ]
        
        # Linear interpolation
        trajectory:Tuple[np.ndarray, ...] = ([
            np.array([
                current[j] + ( delta[j] * float(i)/float(steps) ) 
                for j in range(len(current))
            ])
            for i in range(1, steps+1)
        ])

        return trajectory




class ObjectTarget:
    """
    Object target configuration
    """
    def __init__(self, uid):
        self.uid = uid
        self.update()
    
    def reset(self):
        # generate new pose
        x = 0.65 + uniform(-0.15, 0.15)
        y = uniform(-0.15, 0.15)
        pos = (x, y, 0.1)
        yaw = uniform(-1, 1) * math.pi
        orn = p.getQuaternionFromEuler([0, 0, yaw])

        # update in bullet
        p.resetBasePositionAndOrientation(self.uid, pos, orn)

        # update target info
        self.update()
        
    def update(self):
        pos, orn = p.getBasePositionAndOrientation(self.uid)
        orn_eul = p.getEulerFromQuaternion(orn)
        self.pos = pos
        self.yaw = orn_eul[2]



def main():
    """
    Entry point
    """
    # Initiate PyBullet
    p.connect(p.GUI)

    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    if (egl):
        eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")


    # Load URDF, basePosition -> x, y, z
    pb_data_path = pybullet_data.getDataPath()
    project_path = os.path.dirname(os.path.abspath(__file__))

    robot_uid = p.loadURDF(os.path.join(project_path, "urdf/ur5_rg2.urdf"), useFixedBase=True)
    table_uid = p.loadURDF(os.path.join(pb_data_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
    tray_uid = p.loadURDF(os.path.join(pb_data_path, "tray/traybox.urdf"), basePosition=[0.65, 0, 0])
    target_uid = p.loadURDF(os.path.join(pb_data_path, "random_urdfs/000/000.urdf"), basePosition=[0.7, 0, 0.1])

    # Config bullet
    CONTROL_DT = 1./240.        # 240Hz default
    p.setTimeStep(CONTROL_DT)   # pybullet time step
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

    # Init Robot and Target
    robot = Robot(robot_uid, CONTROL_DT)
    obj_target = ObjectTarget(target_uid)

    while True:
        # For smooth rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # This replaces perception
        obj_target.update()
        robot.set_object(obj_target.pos, obj_target.yaw)

        # Entire pipeline here!
        is_done = robot.execute()
        if is_done:
            robot.reset()
            obj_target.reset()

        p.stepSimulation()




if __name__ == "__main__":
    main()