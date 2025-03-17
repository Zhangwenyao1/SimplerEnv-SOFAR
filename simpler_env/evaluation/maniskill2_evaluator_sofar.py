"""
Evaluate a model on ManiSkill2 environment.
"""

import debugpy
print("Waiting for debugger attach")
debugpy.listen(5681)
debugpy.wait_for_client()

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import plan.src.utils.config as config
import torch
from GSNet.gsnet_simpler import grasp_inference
import numpy as np
from transforms3d.euler import quat2euler
import open3d as o3d

from simpler_sofar import sofar
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict, get_depth_from_maniskill2_obs_dict, \
                                                    get_camera_extrinsics_from_maniskill2_obs_dict, get_pointcloud_in_camera, get_base_pose
from simpler_env.utils.visualization import write_video

from PIL import Image
from copy import deepcopy
from transforms3d.quaternions import mat2quat
from scipy.spatial.transform import Rotation as R
from SoFar.depth.utils import transform_point_cloud_nohw, inverse_transform_point_cloud


from plan.src.plan import pb_ompl
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.config import DotDict
from plan.src.utils.utils import to_list, to_torch
from plan.src.utils.robot_model import RobotModel
from plan.src.utils.scene import Scene
from plan.src.utils.ik import IK
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.constants import ARM_URDF_FULL_GOOGLE_ROBOT, ARM_URDF_FULL_WIDOWX, ROBOT_JOINTS_WIDOWX, ROBOT_JOINTS_GOOGLEROBOT, FRANKA_COLLISION_FILE, FRANKA_CUROBO_FILE







class Planner:
    def __init__(self, config, fix_joints=[], planner="RRTConnect"):
        self.config = config

        # load robot
        robot = RobotModel(config.urdf)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, config, fix_joints=fix_joints)
        self.pb_ompl_interface.set_planner(planner)

    def clear_obstacles(self):
        raise NotImplementedError
        self.obstacles = []

    def plan(self, start=None, goal=None, interpolate_num=None, fix_joints_value=dict(), time=None, first=None, only_test_start_end=False):
        if start is None:
            start = [0,0,0,-1,0,1.5,0, 0.02, 0.02]
        if goal is None:
            goal = [1,0,0,-1,0,1.5,0, 0.02, 0.02]
        # goal = [0,1.5,0,-0.1,0,0.2,0, 0.02, 0.02]

        self.pb_ompl_interface.fix_joints_value = fix_joints_value
        start, goal = to_list(start), to_list(goal)
        for name, pose in [('start', start), ('goal', goal)]:
            if not self.pb_ompl_interface.is_state_valid(pose):
                print(f'unreachable {name}')
                return False, None
        if only_test_start_end:
            return True, None

        res, path = self.pb_ompl_interface.plan(start, goal, interpolate_num=interpolate_num, fix_joints_value=fix_joints_value, allowed_time=time, first=first)
        if res:
            path = np.array(path)
        return res, path

    def close(self):
        pass




def get_grasp_pose(task_description, intrinsic, object_mask, obj_pts_cam, sce_pts_cam, sce_pts_base, extrinsics, relative_translation_table, relative_rotation_table):
    
    if "drawer" in task_description:
        object_pc_base = transform_point_cloud_nohw(obj_pts_cam, np.array(extrinsics))
        object_pc_base[:, 0] += np.random.normal(loc=0.0, scale=0.01, size=object_pc_base.shape[0])
        obj_pts_cam = inverse_transform_point_cloud(object_pc_base, np.array(extrinsics))

    try:
        gg_group, gg_goal_group = grasp_inference(task_description, intrinsic, object_mask, obj_pts_cam, sce_pts_cam, sce_pts_base, extrinsics, relative_translation_table, relative_rotation_table)
        print('Grasp Inference Completed')
    except Exception as e:
        print(f"An error occurred: {e}")
    if gg_group is None:
        return None, None
    print('len of gg_group', len(gg_group))


    gg_group_list = []
    gg_goal_group_list = []
    for i in range(len(gg_group)):
        gg_group_list.append(gg_group[i].transform(extrinsics))
        gg_goal_group_list.append(gg_goal_group[i].transform(extrinsics))

    if "drawer" in task_description:

        sorted_indices = sorted(range(len(gg_group)), key=lambda i: gg_group_list[i].translation[0])

        gg_sorted = [gg_group_list[i] for i in sorted_indices]
        gg_goal_sorted = [gg_goal_group_list[i] for i in sorted_indices]
        gg_group = gg_sorted
        gg_goal_group = gg_goal_sorted
        return gg_group, gg_goal_group
    else:
        return gg_group_list, gg_goal_group_list
        
    

def filter_pc(sce_pts_base, obs, robot_urdf, robot_joints):
    rm = RobotModel(robot_urdf)
    init_qpos = to_torch(obs['agent']['qpos'][None]).float()
    init_qpos = {k: init_qpos[:, i] for i, k in enumerate(robot_joints)}
    robot_pc, link_trans, link_rot, link_pc = rm.sample_surface_points_full(init_qpos, n_points_each_link=2**11, with_fk=True)
    robot_pc = robot_pc[0]
    state_pc = o3d.geometry.PointCloud()
    state_pc.points = o3d.utility.Vector3dVector(sce_pts_base)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_pc)

    # 使用 KDTree 查找与 robot_pc 重叠的点
    kd_tree = o3d.geometry.KDTreeFlann(state_pc)
    indices_to_remove = []
    for point in robot_pcd.points:
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)  # 设置合适的半径
        indices_to_remove.extend(idx)
    # 移除重复点
    state_pc = state_pc.select_by_index(indices_to_remove, invert=True)
    scene_pc_filter = torch.tensor(np.asarray(state_pc.points))

    return scene_pc_filter

def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=30,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    
    control_mode = "arm_pd_ee_pose_gripper_pd_joint_pos"
    control_mode = "arm_pd_joint_pos_gripper_pd_joint_pos"

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 
    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)


    # Initialize model
    timestep = 0
    success = "failure"
    
    print('Task Start')
    images = []
    for _ in range(3):
        images, env, obs, done, info = sofar_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options)
        if done:
            break
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images.append(image)
    success = "success" if done else "failure"
    new_task_description = env.get_language_instruction()
    if new_task_description != task_description:
        task_description = new_task_description
        print(task_description)
        
    is_final_subtask = env.is_final_subtask()
    timestep += 1
    
        
    _, _, _, _, info = env.step(np.zeros(11))    
    episode_stats = info.get("episode_stats", {})
    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    ckpt_path_basename = 'motion_planning'
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)
    print('=============video save================')
    print(video_path)
    print('=============video save================')
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    return success == "success"

def maniskill2_evaluator_sofar(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr



def sofar_execution(images, env, obs, obs_camera_name, task_description, additional_env_build_kwargs, env_reset_options):
    
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images.append(image)
    
    depth = get_depth_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    intrinsic, extrinsics = get_camera_extrinsics_from_maniskill2_obs_dict(env,
    obs, camera_name = obs_camera_name)
    base_pose = get_base_pose(obs)
    
    extrinsics = np.linalg.inv(np.array(extrinsics) @ np.array(base_pose))
    try:
        sce_pts_cam, sce_pts_base, obj_pts_cam, obj_pts_base, object_mask, relative_translation_table, relative_rotation_table = sofar(image, depth, intrinsic, extrinsics, task_description)
    except:
        return images, env , obs, None, None
    graspness_threshold = 0.01
    try:
        gg_group, gg_goal_group = get_grasp_pose(task_description, intrinsic, object_mask, obj_pts_cam, sce_pts_cam, sce_pts_base, extrinsics, relative_translation_table, relative_rotation_table)
    except Exception as e:
        print(f'get_grasp_pose error {e}')
        return images, env , obs, None, None

    init = np.array(obs['agent']['qpos'][:7]) 
    robot_urdf = ARM_URDF_FULL_GOOGLE_ROBOT
    robot_joints = ROBOT_JOINTS_GOOGLEROBOT
    vis = Vis(robot_urdf)
    # filter the scene pcd
    scene_pc_filter = filter_pc(sce_pts_base, obs, robot_urdf, robot_joints)
    cfg = config.DotDict(
        urdf=robot_urdf,
        pc = torch.tensor([[0,0,0]]) if "drawer" in task_description else scene_pc_filter,
        curobo_file = "./plan/robot_models/google_robot/curobo/google_robot.yml",
        robot_type = "google_robot",
    )
    
    if gg_group is None:
        return images, env, obs, None, None

    for i in range(len(gg_group)):
    # get the grasp pose and pose pose
        gg = gg_group[i]  
        if gg.translation[0] < 0.2:        
            continue 
        if "pick" in task_description:
            # Define the rotation angle in degrees and convert to radians
            if 'lr_switch' in additional_env_build_kwargs.keys():
                if additional_env_build_kwargs['lr_switch']==True:
                    angle_deg = 30
                    angle_rad = np.radians(angle_deg)

                    # Compute the rotation matrix around the y-axis
                    rotation_matrix = np.array([
                        [np.cos(angle_rad), 0, np.sin(angle_rad)],
                        [0, 1, 0],
                        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
                    ])
                    gg.rotation_matrix = rotation_matrix @ gg.rotation_matrix    
            gg_goal = deepcopy(gg)
            gg_goal.translation[0] -=0.05
            gg_goal.translation[2] += 0.05
        else:
            gg_goal = gg_goal_group[i]

        print("\nStar Planning First Phase")
        
        robot_state = np.array(obs['agent']['qpos'])
        goal = mat2quat(gg.rotation_matrix)
        goal = np.concatenate([gg.translation, goal])
        ik = IK(robot='google_robot')
        align = np.array([
                [0,0,1],
                [0,-1,0],
                [1,0,0],
            ],
        )
        rot = gg.rotation_matrix @ align
        delta_rot = R.from_euler('xyz', [-0.001328879239766479, -0.00443973089771006, -0.0008686511567993004], degrees=False).as_matrix()
        delta_trans = np.array([4.3374896E-03, 1.8852949E-03, 1.6353297E-01])
        trans_2 = gg.translation - np.einsum('ab,b->a', rot, delta_trans)
        rot_2 = np.einsum('ba,bc->ac', delta_rot, rot)
        goal = ik.ik(trans_2, rot_2, joints=ik.robot_to_arm_joints(init))
        try:
            for _ in range(3):
                goal =  ik.ik(trans_2, rot_2, joints=ik.robot_to_arm_joints(init))
                if goal is not None:
                    break
        except Exception as e:
            print(f"Error encountered: {e}. Skipping to next iteration.")
            continue
        if goal is None:
            print("Grasp Path IK No Solution")
            continue

        print('\nPlace Path Starting')
        place_init_qpos = goal
        rot = gg_goal.rotation_matrix @ align
        delta_rot = R.from_euler('xyz', [-0.001328879239766479, -0.00443973089771006, -0.0008686511567993004], degrees=False).as_matrix()
        delta_trans = np.array([4.3374896E-03, 1.8852949E-03, 1.6353297E-01])
        trans_2 = gg_goal.translation - np.einsum('ab,b->a', rot, delta_trans)
        rot_2 = np.einsum('ba,bc->ac', delta_rot, rot)
        
        try:
            for _ in range(3):
                place_goal_qpos = ik.ik(trans_2, rot_2, joints=ik.robot_to_arm_joints(place_init_qpos))
                if place_goal_qpos is not None:
                    break
        except Exception as e:
            print(f"Error encountered: {e}. Skipping to next iteration.")
            continue
        if place_goal_qpos is None:
            print("Place Path IK No Solution")
            continue
        for _ in range(10):
            planner = Planner(cfg, planner='AITstar', fix_joints=['joint_finger_right', 'joint_finger_left', 'joint_head_pan', 'joint_head_tilt'])
            res_grasp, grasp_path = planner.plan(robot_state[:7], goal, interpolate_num=30, fix_joints_value={'joint_finger_right': 0, 'joint_finger_left': 0, 'joint_head_pan': 0, 'joint_head_tilt': 0})
            if res_grasp : # and isinstance(grasp_path, np.ndarray)
                print('\nGrasp Path Completed')
                break
        if grasp_path is None or res_grasp == False: #  or not isinstance(grasp_path, np.ndarray)
            continue
        

        for _ in range(10):
            planner = Planner(cfg, planner='AITstar', fix_joints=['joint_finger_right', 'joint_finger_left', 'joint_head_pan', 'joint_head_tilt'])
            res_place, place_path = planner.plan(place_init_qpos[:7], place_goal_qpos, interpolate_num=30, fix_joints_value={'joint_finger_right': 0.5, 'joint_finger_left': 0.5, 'joint_head_pan': -0.00285961, 'joint_head_tilt': 0.7851361})
            if res_place: # and isinstance(place_path, np.ndarray)
                print('\nPlace Path Completed')
                break
        if place_path is None or res_place == False: # or not isinstance(place_path, np.ndarray)
            continue
        else:
            break
        # import pdb; pdb.set_trace()
    try:
        if isinstance(grasp_path, np.ndarray):    
            grasp_path[:, 7] = -0.00285961
            grasp_path[:, 8] = 0.7851361
            grasp_path[:, 9] = 0 
            grasp_path[:, 10] = 0
            
            num_copies = 5
            repeated_elements = np.tile(grasp_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [index/num_copies, index/num_copies]
            grasp_path = np.vstack([grasp_path, repeated_elements])
            for index in range(len(grasp_path)):
                obs, reward, done, truncated, info = env.step(grasp_path[index])   
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)
                   
        else:
            return images, env, obs, None, None
    except:
        return images, env, obs, None, None
    try:
        if isinstance(place_path, np.ndarray):    
            place_path[:, 7] = -0.00285961
            place_path[:, 8] = 0.7851361
            place_path[:, 9] = 1
            place_path[:, 10] = 1
            num_copies = 5
            repeated_elements = np.tile(place_path[-1], (num_copies, 1))
            for index in range(num_copies):
                repeated_elements[index, -2:] = [1-index/num_copies, 1-index/num_copies]
            place_path = np.vstack([place_path, repeated_elements])
            for index in range(len(place_path)):
                obs, reward, done, truncated, info = env.step(place_path[index])
                img = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
                images.append(img)
        else:
            return images, env, obs, None, None
    except:
        return images, env, obs, None, None
    return images, env, obs, done, info