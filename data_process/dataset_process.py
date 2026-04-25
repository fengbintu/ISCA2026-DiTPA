from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Protocol, Tuple, Union
from data_process.action_tokenizer import ActionTokenizer
from data_process.rlds import make_interleaved_dataset, make_single_dataset,apply_frame_transforms
from data_process.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from data_process.rlds.utils.data_utils import NormalizationType

IGNORE_INDEX = -100

class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...

class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the VLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("vla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
        
        if 'DEBUG' in os.environ:
            import ipdb;ipdb.set_trace()

        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size= 1, 
        future_action_window_size= 0,
        batch_size = None,
        batchfy=False,
        subsample_length=None,
        subsample_type=None,
        center_crop=False,
        max_action=None, # 2.5
        max_proprio=None,
        clamp_value=False,
        load_proprio=False,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        load_camera_views=("primary",),
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.buffer_size, self.batchfy = shuffle_buffer_size, batchfy

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=load_proprio,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                # goal_relabeling_strategy="uniform",                 # Goals are currently unused
                subsample_length=subsample_length,
                subsample_type=subsample_type,
                max_action=max_action,
                max_proprio=max_proprio,
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=4,                          # For CPU-intensive ops (decoding, resizing, etc.)
                center_crop=center_crop,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            clamp_value=clamp_value,
        )

        self.train = train
        
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
                random_contrast=[0.9, 1.1],
                random_saturation=[0.9, 1.1],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )})            

        # Initialize RLDS Dataset
        if batch_size is not None:
            rlds_config['batch_size'] = batch_size
        self.batch_size = batch_size
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        print('rlds_config', rlds_config['clamp_value'])
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        if self.batchfy:
            for i in range(0, self.dataset_length, self.buffer_size):
                print('next batch', i)
                for rlds_batch in self.batch_iter(self.buffer_size, i).as_numpy_iterator():
                    yield self.batch_transform(rlds_batch)
        else:
            while True:
                try:
                    for rlds_batch in self.dataset.as_numpy_iterator():
                        yield self.batch_transform(rlds_batch)
                except Exception as e:
                    # exit()
                    import traceback
                    traceback.print_exc()
                    print(e)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
