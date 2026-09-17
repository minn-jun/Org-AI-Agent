"""원격 코드 모델(Alibaba-NLP/gte-multilingual-reranker-base, 'new-impl')을 transformers 5.x에서 돌리기 위한 보정.

두 가지가 깨진다 (2026-09-15, transformers 5.16 / 5.17 확인).
1. 회전 위치(rope) cos/sin 표, position_ids 같은 non-persistent 버퍼가 체크포인트에 없어 쓰레기값으로 남는다
   -> IndexError(rope_cos[position_ids]) 또는 CUDA device-side assert.
2. PreTrainedModel.get_extended_attention_mask가 5.x에서 빠졌다 -> AttributeError.

보정 후 확인: 관련 문장 로짓 0.97, 무관 문장 -3.3 / -1.8 (정상 방향). 공식 4.x 실행과의 수치 일치는 확인하지 않았다.
"""

from __future__ import annotations

import types

import torch


def _fix_buffers(model: torch.nn.Module) -> None:
    device = next(model.parameters()).device
    for mod in model.modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "_set_cos_sin_cache"):
            mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2).float().to(device) / mod.dim))
            seq_len = int(mod.max_position_embeddings * getattr(mod, "scaling_factor", 1))
            mod._set_cos_sin_cache(seq_len, device, torch.float32)
        if isinstance(getattr(mod, "position_ids", None), torch.Tensor):
            mod.position_ids = torch.arange(mod.position_ids.size(0), device=device)


def _get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """transformers 4.x PreTrainedModel.get_extended_attention_mask (인코더용)."""
    dtype = dtype or next(self.parameters()).dtype
    ext = attention_mask[:, None, :, :] if attention_mask.dim() == 3 else attention_mask[:, None, None, :]
    ext = ext.to(dtype=dtype)
    return (1.0 - ext) * torch.finfo(dtype).min


def fix_model(model: torch.nn.Module) -> torch.nn.Module:
    """모델을 장치로 옮기고 dtype을 정한 **뒤에** 부른다(버퍼를 그 장치에서 다시 만든다)."""
    _fix_buffers(model)
    for mod in model.modules():
        if not hasattr(mod, "get_extended_attention_mask") and hasattr(mod, "embeddings") and hasattr(mod, "encoder"):
            mod.get_extended_attention_mask = types.MethodType(_get_extended_attention_mask, mod)
    return model
