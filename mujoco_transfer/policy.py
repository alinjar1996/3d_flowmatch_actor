"""Strict loading and simulator-independent inference for the released policy."""

from collections import deque
import hashlib
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_URL = "https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/3dfa_peract2.pth"
CHECKPOINT_SHA256 = "f92feb0c671e19682bc540514a33b81c8f4a306bf9393ae6cd635ef65f7cf124"
PERACT2_MODEL = dict(
    backbone="clip", embedding_dim=120, num_attn_heads=8,
    num_vis_instr_attn_layers=3, fps_subsampling_factor=4,
    nhist=3, nhand=2, num_shared_attn_layers=4, relative=False,
    rotation_format="quat_xyzw", denoise_timesteps=5,
    denoise_model="rectified_flow", pretrained_backbones=False,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("weight", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a state dict or a checkpoint with 'weight'")
    result = {}
    for key, tensor in state.items():
        # Strip known wrappers, never blindly discard the first seven chars.
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in result:
            raise ValueError(f"Duplicate checkpoint key after unwrapping: {key}")
        result[key] = tensor
    bounds = result.get("workspace_normalizer")
    if bounds is None or bounds.shape != (2, 3):
        raise ValueError("Checkpoint must contain its 2x3 workspace normalizer")
    if not torch.isfinite(bounds).all() or not torch.all(bounds[1] > bounds[0]):
        raise ValueError("Invalid checkpoint workspace bounds")
    return result


def load_actor(path, device="cpu"):
    from modeling.policy.denoise_actor_3d import DenoiseActor
    actor = DenoiseActor(**PERACT2_MODEL)
    state = checkpoint_state(path)
    ignored = migrate_unused_norms(actor, state)
    actor.load_state_dict(state, strict=True)
    actor.legacy_unused_checkpoint_keys = ignored
    return actor.to(device).eval()


def migrate_unused_norms(actor, state):
    """Remove only old norm_kv tensors proven unused by post-norm layers.

    The published file includes these; the released code sets norm_kv=None
    when pre_norm=False and never calls it. All actual model weights still
    load strictly. Unexpected keys outside this migration remain fatal.
    """
    from modeling.utils.layers import AttentionLayer
    ignored = []
    for name, layer in actor.named_modules():
        if isinstance(layer, AttentionLayer) and not layer.pre_norm and layer.norm_kv is None:
            for suffix in ("weight", "bias"):
                key = f"{name}.norm_kv.{suffix}"
                if key in state:
                    if state[key].shape != getattr(layer.norm_q, suffix).shape:
                        raise ValueError(f"Invalid legacy normalization tensor: {key}")
                    del state[key]
                    ignored.append(key)
    return ignored


class PoseHistory:
    """Match the online evaluator: oldest -> newest; repeat initial pose."""

    def __init__(self, length=3):
        self.length = length
        self.states = deque(maxlen=length)

    def reset(self):
        self.states.clear()

    def append(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (2, 8) or not np.isfinite(state).all():
            raise ValueError("Expected [left, right] XYZ/XYZW/open state (2,8)")
        self.states.append(state.copy())
        while len(self.states) < self.length:
            self.states.appendleft(state.copy())
        return np.stack(self.states)


class FlowMatchPolicy:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu or repair the driver")
        self.actor = load_actor(checkpoint, device)
        self.history = PoseHistory()
        self.tokens = None
        self.checkpoint = str(Path(checkpoint).resolve())

    def reset(self, instruction):
        # OpenAI CLIP ships the same BPE vocabulary; remove trailing pads to
        # reproduce HF CLIPTokenizer(padding='longest') for one instruction.
        import clip
        tokens = clip.tokenize([instruction], truncate=False)
        self.tokens = tokens[:, :int(tokens.argmax(dim=-1)[0]) + 1].to(self.device)
        self.history.reset()

    @torch.inference_mode()
    def predict(self, rgb, points, state):
        if self.tokens is None:
            raise RuntimeError("Call reset(instruction) at the start of each episode")
        rgb, points = np.asarray(rgb), np.asarray(points)
        expected = (3, 256, 256, 3)
        if rgb.shape != expected or points.shape != expected:
            raise ValueError(f"Expected RGB/point clouds in camera HWC order {expected}")
        if rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 [0,255]; normalization happens once")
        if not np.isfinite(points).all():
            raise ValueError("Point clouds contain invalid coordinates")
        history = self.history.append(state)
        rgb_t = torch.as_tensor(rgb.copy(), device=self.device).permute(0, 3, 1, 2)[None].float() / 255
        pcd_t = torch.as_tensor(points.copy(), device=self.device).permute(0, 3, 1, 2)[None].float()
        proprio = torch.as_tensor(history, device=self.device)[None]
        mask = torch.zeros((1, 1, 2), dtype=torch.bool, device=self.device)
        action = self.actor(None, mask, rgb_t, None, pcd_t, self.tokens, proprio,
                            run_inference=True)[0, 0].cpu().numpy()
        if action.shape != (2, 8) or not np.isfinite(action).all():
            raise RuntimeError("Policy returned an invalid action")
        return action
