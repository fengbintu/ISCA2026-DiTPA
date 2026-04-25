import torch
from torch import nn
import datetime
import os
from hydra.core.hydra_config import HydraConfig
import scripts.ditpa_globals as ditpa_gparas

def resume_or_load_checkpoint(cfg, network, optimizer, scheduler, loss_scaler=None):
    run_dir = HydraConfig.get().run.dir
    if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
        run_dir = run_dir.split('/')[0] + '/auto'
    tensorboard_output_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir,)
    checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "checkpoints")
    tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
    log_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "output.log") 
    start_epoch = 0
    total_iter_num = 0
    if "pretrained_path" in cfg and cfg.pretrained_path != "None":
        ckpt_path = cfg.pretrained_path
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, 'cpu')
            if "load_qformer" in cfg and cfg.load_qformer == False:
                state_dict_new = {k:v for k,v in ckpt["parameter"].items() if not "qformer.queries" in k}
                print(network.load_state_dict(state_dict_new, strict = False))
                _, num_quries, _ = ckpt["parameter"]["image_tokenizer.qformer.queries"].shape
                network.image_tokenizer.qformer.queries.data[:, :num_quries].copy_(ckpt["parameter"]["image_tokenizer.qformer.queries"])
            else:
                if 'parameter' in ckpt:
                    print(network.load_state_dict(ckpt["parameter"]))
                else:
                    print(network.load_state_dict(ckpt))
            print("load checkpoint successfully.", flush = True)
    
    if "ckpt_path" in cfg and cfg.ckpt_path != "None" :
        ckpt_path = cfg.ckpt_path
        if cfg.ckpt_path == 'auto' and os.path.exists(checkpoint_path):
            if len(os.listdir(checkpoint_path)) == 0:
                return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir
            ckpt_path = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))[-1])
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, sorted(os.listdir(ckpt_path), key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)))[-1])
        if os.path.exists(ckpt_path):
            print('load ', cfg.ckpt_path)
            ckpt = torch.load(ckpt_path, 'cpu')
            print(network.load_state_dict(ckpt["parameter"]))
            if optimizer is not None and 'optimizer' in ckpt:
                print("load optimizer.", flush = True)
                print(optimizer.load_state_dict(ckpt["optimizer"]))
            if scheduler is not None and 'scheduler' in ckpt:
                print("load scheduler.", flush = True)
                print(scheduler.load_state_dict(ckpt["scheduler"]))

            if loss_scaler is not None and 'loss_scaler' in ckpt:
                print("load loss scaler.", flush = True)
                print(loss_scaler.load_state_dict(ckpt['loss_scaler']))

            start_epoch = ckpt["epoch"]
            total_iter_num = ckpt["total_iter_num"]+1 
            
            run_dir = HydraConfig.get().run.dir
            if 'ckpt_path' in cfg and cfg.ckpt_path =='auto':
                run_dir = 'auto'
            
            if not ckpt_path.__contains__('/2024'): # this means we resume from original directory. Thus we not update the directory ot origin
                pass
            else:
                # use checkpoints run.dir
                run_dir = str(ckpt_path).replace(str(HydraConfig.get().runtime.cwd), '')
                if run_dir.startswith('/'):
                    run_dir = run_dir[1:]
                run_dir = run_dir.split('checkpoints')[0]
                tensorboard_output_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir,)
                checkpoint_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "checkpoints")
                tensorboard_path = os.path.join(tensorboard_output_path, "tensorboard")
                log_path = os.path.join(HydraConfig.get().runtime.cwd, run_dir, "output.log")
        
    print("all resume successfully.", flush = True)
    return start_epoch,total_iter_num,checkpoint_path,tensorboard_path,log_path, run_dir


def _as_scalar_tensor(x, device, dtype):
    if isinstance(x, nn.Parameter):
        x = x.data
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.to(device=device, dtype=dtype)

def _as_tensor(x, device, dtype):
    if isinstance(x, nn.Parameter):
        x = x.data
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.to(device=device, dtype=dtype)

@torch.no_grad()
def apply_dequantation_from_ckpt(cfg, model: nn.Module, ckpt: dict, device="cpu", target_dtype=torch.float32):
    modules = dict(model.named_modules())

    for w_key, w_int in ckpt.items():
        if not (isinstance(w_key, str) and w_key.endswith(".weight")):
            continue
        if not (torch.is_tensor(w_int) and w_int.dtype in (torch.int8, torch.uint8, torch.int16)):
            continue

        module_name = w_key[:-len(".weight")]
        if module_name.startswith("model.model."): 
            module_name_core = module_name[len("model."):]  
        else:
            module_name_core = module_name

        cand = "transformer." + module_name_core 
        mod = modules.get(cand, None)

        if not isinstance(mod, nn.Linear):
            continue

        delta_key = f"{module_name}.weight_quantizer.delta"
        zp_key = f"{module_name}.weight_quantizer.zero_point"
        if delta_key not in ckpt:
            raise KeyError(f"Missing {delta_key} for {module_name}")

        delta_w = ckpt[delta_key]
        zp_w = ckpt.get(zp_key, 0)

        w_int = w_int.to(device=device)
        delta_w = _as_tensor(delta_w, device=device, dtype=torch.float32)
        zp_w = _as_tensor(zp_w, device=device, dtype=torch.float32)  

        out_features = w_int.shape[0] 
        if delta_w.numel() != out_features:
            raise ValueError(f"{delta_key} numel={delta_w.numel()} != out_features={out_features}")
        if zp_w.numel() == 1:
            zp_w = zp_w.expand(out_features, 1)
        elif zp_w.numel() != out_features:
            raise ValueError(f"{zp_key} numel={zp_w.numel()} not in (1, {out_features})")

        delta_w = delta_w.view(out_features, 1)
        zp_w = zp_w.view(out_features, 1)

        w_ref_fp32 = (w_int.to(torch.float32) - zp_w) * delta_w
        w_fp = w_ref_fp32.to(dtype=target_dtype)

        if cfg.model_setting == "default":
            continue
        else:
            mod.weight.data = mod.weight.data.to(device=device, dtype=target_dtype)
            mod.weight.copy_(w_fp)


@torch.no_grad()
def load_quantized_weights_scales(cfg, network):
    raw = torch.load(cfg.quant_action_panner_path, map_location="cpu")
    state_dict = raw.get("parameter", raw.get("model", raw))
    state_dict.pop("_quant_config", None)

    deltas = {k: v for k, v in state_dict.items() if isinstance(k, str) and k.endswith(".delta")}

    ditpa_gparas.dict_scale_param = deltas 
    apply_dequantation_from_ckpt(cfg, network, state_dict, device="cpu", target_dtype=torch.float32)


def load_checkpoint(cfg, network):
    _, _, _, tensorboard_path, log_path, _ = resume_or_load_checkpoint(cfg, network, None, None)
    load_quantized_weights_scales(cfg, network)


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
