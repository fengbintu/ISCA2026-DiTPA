import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "utils"))
sys.path.append(os.path.join(current_path, "scripts"))
import argparse
import time
import tqdm
import json
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from libero.libero import benchmark
from scripts.action_planner_modeling import RobotTransformerNet
from close_loop_eval import get_task_suite_name, PytorchDiffInference
from data_process.dataset_process import RLDSDataset
from data_process.env_process import (
    get_libero_dummy_action, 
    get_libero_env,
    get_libero_image,
)
from utils.ddp_utils import init_distributed_mode
from utils.excel_utils import export_baseline_log_to_excel, export_ditpa_log_to_excel, save_visulized_task_sample   
from data_process.checkpoint_process import load_checkpoint
import scripts.ditpa_globals as ditpa_gparas

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ditpa_simulation_evaluation(
    test_episodes_num=None,
    model=None,
    args=None,
    stride=1,
    root_folder = None,
    cfg=None, 
    dataset_statistics = None, 
):
    ### Create folder and files ###
    if root_folder != None:  
        log_folder = os.path.join(root_folder, 'logs')
        video_folder = os.path.join(root_folder, 'videos')
        trajectory_folder = os.path.join(root_folder, 'trajectories')
        sw_res_folder = os.path.join(root_folder, 'software_res')
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(video_folder, exist_ok=True)  
        os.makedirs(trajectory_folder, exist_ok=True)  
        os.makedirs(sw_res_folder, exist_ok=True) 
        fd_eval=open(log_folder+f'/{ditpa_gparas.test_label}_eval_record.log', 'w', encoding='utf-8') 

    ### Define inference model ###
    model = PytorchDiffInference(  
        model = model,
        sequence_length=cfg.dataset.traj_length,
        use_wrist_img = cfg.model.use_wrist_img,
        num_pred_action = cfg.num_pred_action,
        stride = stride,  
        use_action_head_diff = cfg.use_action_head_diff, 
    )

    ### Obtain libero task suite ###
    num_trails_per_task = 50
    num_steps_wait = 10 
    benchmark_dict = benchmark.get_benchmark_dict()  
    task_suite_name = get_task_suite_name(cfg.dataname)
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    ### Start evaluation ###
    print(f"============================= Start evaluation =====================================", flush=True)
    print(f"============== It will take several minutes to update, please wait... ==============", flush=True)
    angle_th = 0 if ditpa_gparas.action_skip_flag == False else 2
    if angle_th == 0 or angle_th == 2:
        total_episodes, total_successes, total_infer_times, total_skip_actions, total_latency = 0, 0, 0, 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, None, resolution = 256, cuda_id=args.local_rank)
            model.set_natural_instruction(task_description)  

            task_episodes, task_successes, task_infer_times, task_skip_actions, task_latency = 0, 0, 0, 0, 0
            for episode_idx in range(int(num_trails_per_task)): 
                model.reset_observation()  
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])  
                max_steps = 520

                t = 0
                all_actions = []

                action_pre = np.array(7, dtype=np.float32) # (t-1) action
                cur_abs_rad = np.zeros(3, dtype=np.float32)  # current absolute radian
                pre_abs_deg = np.zeros(3, dtype=np.float32)  # previous absolute degree
                preset_angle = 1*angle_th  # angle threshold 0, 1, 2, 3, ...
                angle_deg = 0  # angle between current and previous absolute degree
                skip_flag = False
                skip_t = 0

                ditpa_gparas.other_latency = 0
                ditpa_gparas.vlm_latency = 0
                ditpa_gparas.dit_latency = 0
                while t < max_steps + num_steps_wait:
                    ### Initialize to drop objects ###
                    if t < num_steps_wait:  # [0,9]
                            obs, reward, done, info = env.step(get_libero_dummy_action(None))
                            t += 1
                            continue
                    ditpa_gparas.dit_steps = t-num_steps_wait # 0, 1, 2, ...
                    
                    ### Set observation ###
                    img = get_libero_image(obs, 224) 
                    model.set_observation(img)

                    ### Action prediction ###
                    if (preset_angle != 0) and (t > num_steps_wait+1) and (angle_deg < preset_angle) and skip_flag:
                        action = action_pre
                        skip_flag = False
                        skip_t += 1
                    else:
                        ditpa_gparas.dit_last_skip_actions = not skip_flag
            
                        dit_start_time = time.time() 
                        model_output = model.inference(
                            trajectory_dim = cfg.trajectory_dim,
                            ret_7 = True,
                        )
                        dit_end_time = time.time() 
                        model_output[...,-1] = np.where(model_output[...,-1] > 0.5, np.ones_like(model_output[...,-1]), np.ones_like(model_output[...,-1])*(-1))
                        normalized_actions = model_output
                        action_norm_stats = dataset_statistics[cfg.dataname]['action']
                        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                        actions = np.where(
                            mask,
                            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                            normalized_actions,
                        )
                        actions[...,-1] = actions[..., -1] * -1.0
                        action = actions
                        skip_flag = True

                        ditpa_gparas.dit_latency = ditpa_gparas.dit_latency + (dit_end_time - dit_start_time)
                    
                    all_actions.append(action.tolist())

                    ### Simulator step ###
                    obs, reward, done, info = env.step(action.tolist())
                    
                    ### Angle calculation ###
                    # 1. save previous action
                    # 2. compute current absolute degree
                    # 3. compute angle between current and previous absolute degree
                    # 4. update previous absolute degree
                    action_pre = action  # [7] save action for reuse

                    cur_abs_rad = action[3:6] + cur_abs_rad  # update current absolute radiance
                    cur_abs_deg = np.degrees(cur_abs_rad)  # update current absolute degree

                    dot = np.sum(cur_abs_deg * pre_abs_deg)
                    norm1 = np.linalg.norm(cur_abs_deg)
                    norm2 = np.linalg.norm(pre_abs_deg)
                    cosine_sim = dot / (norm1 * norm2 + 1e-8)  # compute cosine similarity
                    angle_rad = np.arccos(np.clip(cosine_sim, -1.0, 1.0))  # compute angle in radian
                    angle_deg = angle_rad * 180 / np.pi  # convert to degree

                    pre_abs_deg = cur_abs_deg  # update previous absolute degree

                    if done:
                        task_successes += 1
                        total_successes += 1
                        task_latency += ditpa_gparas.dit_latency - ditpa_gparas.vlm_latency - ditpa_gparas.other_latency
                        total_latency += ditpa_gparas.dit_latency - ditpa_gparas.vlm_latency - ditpa_gparas.other_latency
                        task_infer_times += t-10
                        total_infer_times += t-10
                        task_skip_actions += skip_t
                        total_skip_actions += skip_t
                        break
                    elif t == max_steps + num_steps_wait - 1:
                        task_latency += ditpa_gparas.dit_latency - ditpa_gparas.vlm_latency - ditpa_gparas.other_latency
                        total_latency += ditpa_gparas.dit_latency - ditpa_gparas.vlm_latency - ditpa_gparas.other_latency
                        task_infer_times += (t+1)-10
                        total_infer_times += (t+1)-10
                        task_skip_actions += skip_t
                        total_skip_actions += skip_t
                    t += 1
                
                task_episodes += 1
                total_episodes += 1

                if ditpa_gparas.test_label.lower() == "ditpa" and task_id == 1:  # save sample video for visualization
                    model.save_video(os.path.join(video_folder, f'{ditpa_gparas.test_label}_trail{episode_idx}_{done}_{t-10}.mp4'))

                ### Save results ###
                print(f"Episodes completed so far: {total_episodes}", flush=True)

                fd_eval.write(f"\n======Episodes completed so far: {total_episodes}======")
                fd_eval.write(f"\nSuccess rate: {total_successes / total_episodes * 100:.0f}%")
                if ditpa_gparas.test_label.lower() == "baseline":
                    fd_eval.write(f'\nAction planner latency: {(ditpa_gparas.dit_latency - ditpa_gparas.vlm_latency):.3f} seconds') 
                fd_eval.write(f'\nTotal actions: {t-10}')
                if ditpa_gparas.test_label.lower() == "ditpa":
                    fd_eval.write(f'\nSkip actions: {skip_t}')
                fd_eval.flush()

                all_actions = np.array(all_actions)
                traj_save_path = f"{ditpa_gparas.test_label}_task{task_id+1}_trail{episode_idx}_trajectory.txt"
                np.savetxt(os.path.join(trajectory_folder, traj_save_path), all_actions, fmt="%.6f")

            fd_eval.write(f"\n======Current task {task_id+1} success rate: {(float(task_successes) / float(task_episodes)) * 100:.0f}%======")
            if ditpa_gparas.test_label.lower() == "baseline":
                fd_eval.write(f"\n======Current task {task_id+1} latency: {(float(task_latency) / float(task_episodes)):.3f} seconds======")
            fd_eval.write(f"\n======Current task {task_id+1} actions: {float(task_infer_times) / float(task_episodes)}======")
            if ditpa_gparas.test_label.lower() == "ditpa":
                fd_eval.write(f"\n======Current task {task_id+1} skip actions: {float(task_skip_actions) / float(task_episodes)}======")
            fd_eval.flush()
        
        fd_eval.write(f"\n======Total success rate: {(float(total_successes) / float(total_episodes)) * 100:.0f}%======")
        if ditpa_gparas.test_label.lower() == "baseline":
            fd_eval.write(f"\n======Total latency: {(float(total_latency) / float(total_episodes)):.3f} seconds======")
        fd_eval.write(f"\n======Total actions: {float(total_infer_times) / float(total_episodes)}======")
        if ditpa_gparas.test_label.lower() == "ditpa":
            fd_eval.write(f"\n======Total skip actions: {float(total_skip_actions) / float(total_episodes)}======")
        fd_eval.flush()
    fd_eval.close()

    # Export log to excel and save latest path for hardware evaluation
    log_path = os.path.join(root_folder, "logs", f"{ditpa_gparas.test_label}_eval_record.log")
    software_res_dir = os.path.join(root_folder, "software_res")
    if ditpa_gparas.test_label.lower() == "baseline":
        excel_path = export_baseline_log_to_excel(root_path=root_folder, log_path=log_path, out_dir=software_res_dir)
    elif ditpa_gparas.test_label.lower() == "ditpa":
        excel_path = export_ditpa_log_to_excel(root_path=root_folder, log_path=log_path, out_dir=software_res_dir)

    outputs_root = os.path.join(HydraConfig.get().runtime.cwd, "outputs")
    latest_txt = os.path.join(outputs_root, f"latest_{ditpa_gparas.test_label}_output_path.txt")
    with open(latest_txt, "w", encoding="utf-8") as f:
        f.write(excel_path + "\n")
    
    if ditpa_gparas.test_label.lower() == "ditpa":
        save_visulized_task_sample(res_folder='./results', video_folder=video_folder, sw_folder=sw_res_folder)

    return None


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "..", "config"), config_name="config_diffusion_openx")
def inference(cfg: DictConfig):
    DEVICE = 'cuda'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = argparse.Namespace()
    init_distributed_mode(args, cfg)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    args.local_rank = local_rank
    args.rank = rank   

    ### Define dataset ###
    @dataclass
    class DataAdapterForOpenx:        
        def __call__(self, rlds_batch ):
            loss_weight = torch.logical_not(torch.tensor(rlds_batch['action_past_goal']))
            if 'load_proprio' in cfg:
                dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["proprio"])
                state = torch.tensor(rlds_batch['observation']["state"])
            else:
                dataset_name, action = rlds_batch["dataset_name"], torch.tensor(rlds_batch["action"])
                state = torch.tensor(rlds_batch['action'])
            lang = [item.decode().strip() for item in rlds_batch["task"]["language_instruction"].tolist()]
            dataset_name = [item.decode() for item in rlds_batch["dataset_name"].tolist()]

            pixel_values = torch.tensor(rlds_batch["observation"]["image_primary"])
            pixel_values = (pixel_values / 255. - torch.tensor(IMAGENET_DEFAULT_MEAN)) / torch.tensor(IMAGENET_DEFAULT_STD) # Normalize 
            pixel_values = pixel_values.permute(0, 1, 4, 2, 3)
            del rlds_batch
            return dict(pixel_values=pixel_values, action=action, state=state, dataset_name=dataset_name, language_instruction= lang, loss_weight=loss_weight)

    vla_dataset_openx = RLDSDataset(cfg.dataset.data_path, cfg.dataname, DataAdapterForOpenx(), resize_resolution=(224, 224), shuffle_buffer_size=cfg.shuffle_buffer_size, train=True, image_aug=cfg.image_aug if 'image_aug' in cfg else False,
        window_size= cfg.dataset.traj_length + 1 - cfg.num_pred_action, 
        future_action_window_size= cfg.num_pred_action-1, batch_size=cfg.batch_size, 
        )

    ### Define model ###
    network = RobotTransformerNet(
        output_tensor_spec=None,
        vocab_size=cfg.model.vocab_size,
        trajectory_dim=7,
        time_sequence_length=cfg.dataset.traj_length,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        include_prev_timesteps_actions=cfg.model.include_prev_timesteps_actions,
        freeze_backbone=cfg.model.freeze_backbone,
        use_qformer=cfg.model.use_qformer,
        use_wrist_img=cfg.model.use_wrist_img,
        use_depth_img=cfg.model.use_depth_img,
        prediction_type=cfg.prediction_type,
        dim_align_type=cfg.dim_align_type if 'dim_align_type' in cfg else 0,
        input_size='(224, 224)',
        scheduler_type=cfg.scheduler_type,
        num_inference_steps=cfg.num_inference_steps,
        attn_implementation=cfg.attn_implementation,
        use_action_head_diff=cfg.use_action_head_diff,
    )

    ### Load weight ###
    load_checkpoint(cfg, network)
    network = network.to(DEVICE)

    ### Model evaluation ###
    ditpa_simulation_evaluation(
        model=network,
        test_episodes_num=cfg.close_loop_eval.test_episodes_num,
        args=args,
        stride=cfg.dataset.stride,
        root_folder = os.path.join(HydraConfig.get().runtime.cwd, HydraConfig.get().run.dir, "intermediate_data"),
        cfg = cfg,
        dataset_statistics = vla_dataset_openx.dataset_statistics,
    )


if __name__ == "__main__":
    inference()
