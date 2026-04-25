import os
import re
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def export_baseline_log_to_excel(root_path: str, log_path: str, out_dir: str) -> str:
    task_sr_re = re.compile(r"^======Current task (\d+) success rate: ([0-9.]+)%======$")
    task_lat_re = re.compile(r"^======Current task (\d+) latency: ([0-9.]+) seconds======$")
    task_act_re = re.compile(r"^======Current task (\d+) actions: ([0-9.]+)======$")

    total_sr_re = re.compile(r"^======Total success rate: ([0-9.]+)%======$")
    total_lat_re = re.compile(r"^======Total latency: ([0-9.]+) seconds======$")
    total_act_re = re.compile(r"^======Total actions: ([0-9.]+)======$")

    tasks: dict[int, dict[str, float]] = {}
    total_row: dict[str, float] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = task_sr_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["success_rate"] = float(m.group(2)) / 100.0  # 0~1
                continue

            m = task_lat_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["task_time"] = float(m.group(2))  # seconds
                continue

            m = task_act_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["actions"] = float(m.group(2))
                continue

            m = total_sr_re.match(line)
            if m:
                total_row["success_rate"] = float(m.group(1)) / 100.0
                continue

            m = total_lat_re.match(line)
            if m:
                total_row["task_time"] = float(m.group(1))
                continue

            m = total_act_re.match(line)
            if m:
                total_row["actions"] = float(m.group(1))
                continue

    rows = []
    for tid in sorted(tasks.keys()):
        row = {"task": tid}
        row.update(tasks[tid])
        rows.append(row)

    if total_row:
        total_row_out = {"task": "average"}
        total_row_out.update(total_row)
        rows.append(total_row_out)

    df = pd.DataFrame(rows, columns=["task", "success_rate", "actions", "task_time", "position_instability", "velocity_instability"])

    trajectory_dir = os.path.join(root_path, "trajectories")
    task_stats, overall_stats = compute_avg_pi_vi(trajectory_dir=trajectory_dir, prefix="baseline", num_tasks=10)

    df["position_instability"] = np.nan
    df["velocity_instability"] = np.nan

    for tid in range(1, 11):
        mask = df["task"] == tid
        if mask.any():
            df.loc[mask, "position_instability"] = task_stats[tid]["position_instability"]
            df.loc[mask, "velocity_instability"] = task_stats[tid]["velocity_instability"]

    avg_mask = df["task"] == "average"
    if avg_mask.any():
        df.loc[avg_mask, "position_instability"] = overall_stats["position_instability"]
        df.loc[avg_mask, "velocity_instability"] = overall_stats["velocity_instability"]

    df["position_instability"] = pd.to_numeric(df["position_instability"], errors="coerce").round(3)
    df["velocity_instability"] = pd.to_numeric(df["velocity_instability"], errors="coerce").round(3)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_xlsx = out_dir_p / "baseline_software_results.xlsx"
    df.to_excel(out_xlsx, index=False)

    return str(out_xlsx.resolve())

def export_ditpa_log_to_excel(root_path: str, log_path: str, out_dir: str) -> str:
    task_sr_re = re.compile(r"^======Current task (\d+) success rate: ([0-9.]+)%======$")
    task_act_re = re.compile(r"^======Current task (\d+) actions: ([0-9.]+)======$")
    task_skip_re = re.compile(r"^======Current task (\d+) skip actions: ([0-9.]+)======$")

    total_sr_re = re.compile(r"^======Total success rate: ([0-9.]+)%======$")
    total_act_re = re.compile(r"^======Total actions: ([0-9.]+)======$")
    total_skip_re = re.compile(r"^======Total skip actions: ([0-9.]+)======$")

    tasks: dict[int, dict[str, float]] = {}
    total_row: dict[str, float] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = task_sr_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["success_rate"] = float(m.group(2)) / 100.0  # 0~1
                continue

            m = task_act_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["total_actions"] = float(m.group(2))
                continue

            m = task_skip_re.match(line)
            if m:
                tid = int(m.group(1))
                tasks.setdefault(tid, {})["skip_actions"] = float(m.group(2)) 
                continue

            m = total_sr_re.match(line)
            if m:
                total_row["success_rate"] = float(m.group(1)) / 100.0
                continue

            m = total_act_re.match(line)
            if m:
                total_row["total_actions"] = float(m.group(1))
                continue
            
            m = total_skip_re.match(line)
            if m:
                total_row["skip_actions"] = float(m.group(1))
                continue

    rows = []
    for tid in sorted(tasks.keys()):
        row = {"task": tid}
        row.update(tasks[tid])
        rows.append(row)

    if total_row:
        total_row_out = {"task": "average"}
        total_row_out.update(total_row)
        rows.append(total_row_out)

    df = pd.DataFrame(rows, columns=["task", "success_rate", "total_actions", "skip_actions", "position_instability", "velocity_instability"])

    trajectory_dir = os.path.join(root_path, "trajectories")
    task_stats, overall_stats = compute_avg_pi_vi(trajectory_dir=trajectory_dir, prefix="ditpa", num_tasks=10)

    df["position_instability"] = np.nan
    df["velocity_instability"] = np.nan

    for tid in range(1, 11):
        mask = df["task"] == tid
        if mask.any():
            df.loc[mask, "position_instability"] = task_stats[tid]["position_instability"]
            df.loc[mask, "velocity_instability"] = task_stats[tid]["velocity_instability"]

    avg_mask = df["task"] == "average"
    if avg_mask.any():
        df.loc[avg_mask, "position_instability"] = overall_stats["position_instability"]
        df.loc[avg_mask, "velocity_instability"] = overall_stats["velocity_instability"]

    df["position_instability"] = pd.to_numeric(df["position_instability"], errors="coerce").round(3)
    df["velocity_instability"] = pd.to_numeric(df["velocity_instability"], errors="coerce").round(3)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_xlsx = out_dir_p / "ditpa_software_results.xlsx"
    df.to_excel(out_xlsx, index=False)

    return str(out_xlsx.resolve())

def save_visulized_task_sample(res_folder, video_folder, sw_folder):
    video_dir = video_folder
    out_dir1 = res_folder
    out_dir2 = sw_folder
    prefix = "ditpa"

    pat = re.compile(rf"^{re.escape(prefix)}_trail(\d+)_(True|False)_(\d+)\.mp4$")
    select = None
    for name in os.listdir(video_dir):
        m = pat.match(name)
        if not m:
            continue
        done_str = m.group(2)
        if done_str != "True":
            continue
        trail = int(m.group(1))
        actions = int(m.group(3))
        cand = (actions, trail, name)
        if select is None or cand[0] < select[0]:
            select = cand

    if select is not None:
        actions, trail, name = select
        src = os.path.join(video_dir, name)
        dst_name = f"{prefix}_task_sample.mp4"
        dst1 = os.path.join(out_dir1, dst_name)
        dst2 = os.path.join(out_dir2, dst_name)
        shutil.copy2(src, dst1)
        shutil.copy2(src, dst2)

def _load_trajectory_txt(txt_path: str) -> np.ndarray:
    arr = np.loadtxt(txt_path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]  # single-step trajectory
    return arr


def compute_avg_pi_vi(trajectory_dir: str, prefix: str = "baseline", num_tasks: int = 10):
    pat = re.compile(
        rf"^{re.escape(prefix)}_task(\d+)_trail(\d+)_trajectory(?:_(True|False))?\.txt$"
    )

    task_pi_values = defaultdict(list)
    task_vi_values = defaultdict(list)

    all_pi_values = []
    all_vi_values = []

    if not os.path.isdir(trajectory_dir):
        empty = {tid: {"position_instability": np.nan, "velocity_instability": np.nan, "traj_count": 0}
                 for tid in range(1, num_tasks + 1)}
        overall = {"position_instability": np.nan, "velocity_instability": np.nan, "traj_count": 0}
        return empty, overall

    for name in os.listdir(trajectory_dir):
        m = pat.match(name)
        if not m:
            continue

        task_id = int(m.group(1)) 
        txt_path = os.path.join(trajectory_dir, name)
        poses = _load_trajectory_txt(txt_path)

        if poses.shape[1] < 3:
            continue

        _, pi_all = compute_TCP_position_instability(poses)
        _, vi_all = compute_TCP_velocity_instability(poses)

        if not np.isnan(pi_all):
            task_pi_values[task_id].append(float(pi_all))
            all_pi_values.append(float(pi_all))
        if not np.isnan(vi_all):
            task_vi_values[task_id].append(float(vi_all))
            all_vi_values.append(float(vi_all))

    task_stats = {}
    for tid in range(1, num_tasks + 1):
        pis = task_pi_values.get(tid, [])
        vis = task_vi_values.get(tid, [])
        task_stats[tid] = {
            "position_instability": float(np.mean(pis)) if len(pis) > 0 else np.nan,
            "velocity_instability": float(np.mean(vis)) if len(vis) > 0 else np.nan,
            "traj_count": int(max(len(pis), len(vis))),
        }

    overall = {
        "position_instability": float(np.mean(all_pi_values)) if len(all_pi_values) > 0 else np.nan,
        "velocity_instability": float(np.mean(all_vi_values)) if len(all_vi_values) > 0 else np.nan,
        "traj_count": int(max(len(all_pi_values), len(all_vi_values))),
    }
    return task_stats, overall

def compute_TCP_position_instability(poses):
    poses = np.array(poses)
    pos_array = np.cumsum(poses[:, :3], axis=0)
    T = pos_array.shape[0]
    if T < 2:
        return np.array([np.nan, np.nan, np.nan])
    delta = np.abs(np.diff(pos_array, axis=0))
    instability_per_axis = np.sum(delta, axis=0) / delta.shape[0]
    diffs = np.diff(pos_array, axis=0)
    instability_all_axis = np.mean(np.linalg.norm(diffs, axis=1))

    return instability_per_axis, instability_all_axis

def compute_TCP_velocity_instability(poses):
    poses = np.array(poses)
    pos_array = np.cumsum(poses[:, :3], axis=0)
    T = pos_array.shape[0]
    if T < 3:
        return np.array([np.nan, np.nan, np.nan])
    delta = np.diff(pos_array, axis=0)
    delta2 = np.abs(np.diff(delta, axis=0)) / 2  
    instability_per_axis = np.sum(delta2, axis=0) / delta2.shape[0]
    instability_all_axis = np.mean(np.linalg.norm(delta2, axis=1))
    
    return instability_per_axis, instability_all_axis
