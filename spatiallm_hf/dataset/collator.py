import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer

from spatiallm_hf.constants import (
    IGNORE_INDEX,
    LAYOUT_PLACEHOLDER,
    POINT_CLOUD_PLACEHOLDER,
)
from spatiallm_hf.dataset.mm_plugin import get_mm_plugin
from spatiallm_hf.dataset.template import TemplateHF

logger = logging.getLogger(__name__)


def prepare_4d_attention_mask(
    attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype"
) -> "torch.Tensor":
    _, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _encode_messages(
    messages: Sequence[Dict[str, str]],
    system: Optional[str],
    template: TemplateHF,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[int], List[int]]:
    input_ids: List[int] = []
    labels: List[int] = []
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system)
    total_length = len(input_ids)

    for source_ids, target_ids in encoded_pairs:
        if total_length >= template.cutoff_len:
            break
        source_len, target_len = infer_seqlen(
            len(source_ids), len(target_ids), template.cutoff_len - total_length
        )
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len
        input_ids += source_ids + target_ids
        labels += [IGNORE_INDEX] * source_len + target_ids
    return input_ids, labels


@dataclass
class SpatialLMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    template: Optional[TemplateHF] = None
    compute_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.template is None:
            raise ValueError(
                "Template is required for SpatialLMDataCollatorForSeq2Seq."
            )
        self.template.fix_special_tokens(self.tokenizer)
        self.mm_plugin = get_mm_plugin(point_token="<|point_pad|>")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        use_aligned = "_prompt" in features[0]
        batch_point_clouds: List[str] = []
        batch_messages: List[List[Dict[str, str]]] = []
        batch_layouts: List[Any] = []

        if use_aligned:
            for f in features:
                prompts = f.pop("_prompt")
                responses = f.pop("_response")
                pcs = f.pop("_point_clouds", None) or []
                batch_point_clouds.extend(pcs)
                batch_messages.append(prompts + responses)
                batch_layouts.append(f.get("_layouts", None))
        else:
            for f in features:
                batch_point_clouds.append(f["pcd_path"])
                batch_layouts.append(f.get("rooms", None))
                content = f.get(
                    "_user_content", f"{POINT_CLOUD_PLACEHOLDER}\n{LAYOUT_PLACEHOLDER}"
                )
                batch_messages.append([{"role": "user", "content": content}])

        mm_inputs = self.mm_plugin.get_mm_inputs(
            batch_point_clouds, batch_messages, layouts=batch_layouts
        )
        batched_messages = mm_inputs.pop("messages")

        for mi, messages in enumerate(batched_messages):
            feature = features[mi]
            input_ids, labels = _encode_messages(
                messages=messages,
                system=feature.get("_system", self.template.default_system),
                template=self.template,
                tokenizer=self.tokenizer,
            )
            feature["input_ids"] = input_ids
            feature["attention_mask"] = [1] * len(input_ids)
            feature["labels"] = labels

        out: Dict[str, torch.Tensor] = super().__call__(features)
        out.update(mm_inputs)

        for key, value in out.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                out[key] = value.to(self.compute_dtype)
        return out


@dataclass
class SFTDataCollatorWith4DAttentionMaskHF(SpatialLMDataCollatorForSeq2Seq):
    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        out = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            out["attention_mask"] = prepare_4d_attention_mask(
                out["attention_mask"], self.compute_dtype
            )
        return out
