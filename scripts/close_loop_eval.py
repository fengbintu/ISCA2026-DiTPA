from libero.libero import benchmark
import tqdm
import logging
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from utils.data_utils import process_traj_v3
from moviepy.editor import ImageSequenceClip
from PIL import Image
# import sapien.core as sapien
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat
from moviepy.editor import ImageSequenceClip

def get_task_suite_name(dataset_name):
    if dataset_name == 'libero_10_no_noops':
        return 'libero_10'
    if dataset_name == 'libero_spatial_no_noops':
        return 'libero_spatial'
    if dataset_name == 'libero_object_no_noops':
        return 'libero_object'
    if dataset_name == 'libero_goal_no_noops':
        return 'libero_goal'

class PytorchDiffInference(nn.Module):
    def __init__(self, model, prediction_type='epsilon',sequence_length = 15, 
                 use_wrist_img=False, device="cuda", 
                 stride=1, num_pred_action=4,
                 use_action_head_diff=0):
        super().__init__()

        self.device = torch.device(device)
        use_depth_img = False
        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img
        self.sequence_length = sequence_length
        self.num_pred_action = num_pred_action
        self.use_action_head_diff = use_action_head_diff
        self.stride = stride
        self.model = model

        try:
            if hasattr(self.model, 'module'):
                self.use_wrist_img = self.model.module.use_wrist_img
                self.use_depth_img = self.model.module.use_depth_img
            else:
                self.use_wrist_img = self.model.use_wrist_img
                self.use_depth_img = self.model.use_depth_img
        except:
            self.use_wrist_img = False
            self.use_depth_img = False

        self.model.eval()
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.instruction = ""
        self.stride = stride
        self.data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(), 
                torchvision.transforms.Resize((224,224), antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), 
            ]
        )

        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).text_model

        self.to(self.device)
        self.frame = 0
        
        self.eef_pose = None
        self.empty_instruction = None

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_text_encoder(**inputs)[0].squeeze(0)
        self.instruction = text_embeddings

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        if self.model_input == []:
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = rgb_data.unsqueeze(0) 
            if len(self.model_input) < self.sequence_length:
                self.model_input = self.model_input.repeat(self.sequence_length, 1, 1, 1) 
        else:
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = torch.cat((self.model_input, rgb_data.unsqueeze(0)), dim=0) 
            self.model_input = self.model_input[-self.sequence_length :] 

    def reset_observation(self):
        self.model_input = []  
        self.observation = []  
        self.model_input_wrist = []  
        self.model_input_depth = [] 
        self.frame = 0 
                
    def save_video(self, fpath):
        clip = ImageSequenceClip(self.observation, fps=60 / self.stride) # 60/3=20Hz, same to liebro platform
        clip.write_videofile(fpath, codec="libx264", audio=False, logger=None) 

    def calc_act(self, base_episode, camera_extrinsic_cv, current_frame_idx):
        try:
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][current_frame_idx + self.stride]["prev_ee_pose"]).clone()
        except:
            current_frame_idx = min(current_frame_idx, len(base_episode["step"]) - 1)
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][-1]["prev_ee_pose"]).clone()

        pose1[0] -= 0.615  # base to world
        pose2[0] -= 0.615  # base to world
        action = {}
        action["world_vector"], action["rotation_delta"] = process_traj_v3(
            (camera_extrinsic_cv),
            pose1,
            pose2,
        )

        if base_episode["step"][current_frame_idx]["is_terminal"] == True:
            action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
        else:
            action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
        action["gripper_closedness_action"] = torch.tensor(
            base_episode["step"][current_frame_idx]["action"][-1],
            dtype=torch.float32,
        ).unsqueeze(-1)

        return action

    def get_target_pose(self, delta_pos, delta_rot):
        target_ee_pose_at_camera = sapien.Pose(p=self.eef_pose.p + delta_pos)
        r_prev = quat2mat(self.eef_pose.q)
        r_diff = quat2mat(delta_rot)
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))

        return target_ee_pose_at_camera
 
    def inference(self, extrinsics=None, abs_pose=0, abs_seq_pose=0, horizon=-1, set_pose=False, trajectory_dim=11, reg_prediction_nums=0, pad_diff_nums=0, obs_pose=None, cfg=0, ret_7=False, dim=0):
        obs = {"image": self.model_input[-self.sequence_length :].unsqueeze(0)}
        if self.use_wrist_img:
            obs["wrist_image"] = self.model_input_wrist[-self.sequence_length :].unsqueeze(0)
        if self.use_depth_img:
            obs["depth_image"] = self.model_input_depth[-self.sequence_length :].unsqueeze(0)
        obs["natural_language_embedding"] = self.instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)

        if cfg != 0:
            if self.empty_instruction is None:
                inputs = self.clip_tokenizer(text='', return_tensors="pt", max_length=77, padding="max_length")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                with torch.no_grad():
                    self.empty_instruction = self.clip_text_encoder(**inputs)[0].squeeze(0)
                self.empty_instruction = self.empty_instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)
            obs['natural_language_embedding'] = torch.cat([self.empty_instruction, obs['natural_language_embedding'], ], dim=0)
            obs["image"] = torch.cat([obs["image"], obs["image"]], dim=0)

        if obs_pose is not None:
            obs['poses'] = obs_pose.to(obs["natural_language_embedding"].device)  # B 1 11
        
        with torch.cuda.amp.autocast():
            with torch.no_grad(): 
                if self.use_action_head_diff == 2:
                    if hasattr(self.model, 'module'):
                        model_output = self.model.module.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                    else:
                        model_output = self.model.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                else:
                    if hasattr(self.model, 'module'):
                        model_output = self.model.module.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                    else:
                        model_output = self.model.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)

        if ret_7:
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    model_output["gripper_closedness_action"].cpu(),
                ], dim=-1
            )[0]
            self.frame += self.stride 
            return output[0].cpu().numpy()
        elif trajectory_dim == 7:
            assert set_pose
            rot_delta = model_output["rotation_delta"][0]
            def euler_to_quaternion(eulers):
                import scipy.spatial.transform as st
                quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
                return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])
            quat_list = torch.stack([euler_to_quaternion(rot_delta[i].cpu().numpy()) for i in range(len(rot_delta))])[None,...]
            import numpy as np
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    quat_list.cpu(),
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            self.frame += self.stride
            return output.cpu().numpy()
            pass
        else:
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            pass

        output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1
        output[..., -1] = output[..., -1] > 0.5

        self.frame += self.stride

        return output.cpu().numpy()
