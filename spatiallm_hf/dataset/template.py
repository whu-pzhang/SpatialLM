from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizer

from spatiallm_hf.constants import (
    POINT_PAD_TOKEN,
)


@dataclass
class TemplateHF:
    name: str
    default_system: str
    stop_words: List[str]
    cutoff_len: int
    user_format: str
    assistant_format: str
    system_format: str
    prefix_format: str

    def fix_special_tokens(self, tokenizer: PreTrainedTokenizer) -> None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": self.stop_words[0] if self.stop_words else "<|endoftext|>"})

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        special_tokens = [*self.stop_words, POINT_PAD_TOKEN]
        if special_tokens:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [t for t in special_tokens if t not in tokenizer.get_vocab()]}
            )

    def _format(self, messages: Sequence[Dict[str, str]], system: Optional[str]) -> List[str]:
        out: List[str] = []
        sys_text = system or self.default_system
        if sys_text:
            out.append(self.prefix_format + self.system_format.replace("{{content}}", sys_text))
        else:
            out.append(self.prefix_format)

        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            if role == "user":
                out.append(self.user_format.replace("{{content}}", content))
            elif role == "assistant":
                out.append(self.assistant_format.replace("{{content}}", content))
            else:
                raise ValueError(f"Unexpected role: {role}")
        return out

    def encode_multiturn(
        self,
        tokenizer: PreTrainedTokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        encoded = []
        formatted = self._format(messages, system)
        # pairwise user/assistant turns; last assistant is target
        # build source/target token ids incrementally respecting cutoff_len
        ids_per_msg = [
            tokenizer.encode(text, add_special_tokens=False) if len(text) else [] for text in formatted
        ]
        # collapse prefix+system+first user to source; first assistant to target; then subsequent pairs
        # find user/assistant boundaries by role order in messages
        # formatted layout: [prefix(+system), user0, assistant0, user1, assistant1, ...]
        total_len = 0
        for i in range(1, len(ids_per_msg), 2):
            source_ids = [] if i == 1 else ids_per_msg[i - 1]
            target_ids = ids_per_msg[i]
            if i == 1:
                # add prefix/system + user0 to source
                source_ids = ids_per_msg[0] + ids_per_msg[1]
            # truncation within cutoff
            remaining = self.cutoff_len - total_len
            if remaining <= 0:
                break
            s_keep = min(len(source_ids), remaining)
            remaining -= s_keep
            t_keep = min(len(target_ids), remaining)
            source_ids = source_ids[:s_keep]
            target_ids = target_ids[:t_keep]
            total_len += s_keep + t_keep
            encoded.append((source_ids, target_ids))
        return encoded


_TEMPLATES: Dict[str, TemplateHF] = {}


def register_spatiallm_templates(
    cutoff_len: int = 8192,
) -> None:
    # LLaMA 风格
    _TEMPLATES["spatiallm_llama"] = TemplateHF(
        name="spatiallm_llama",
        default_system="",
        stop_words=["<|eot_id|>", "<|eom_id|>"],
        cutoff_len=cutoff_len,
        user_format=(
            "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        assistant_format="{{content}}",
        system_format="<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>",
        prefix_format="",
    )

    # Qwen 风格
    _TEMPLATES["spatiallm_qwen"] = TemplateHF(
        name="spatiallm_qwen",
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        cutoff_len=cutoff_len,
        user_format="<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
        assistant_format="{{content}}<|im_end|>\n",
        system_format="<|im_start|>system\n{{content}}<|im_end|>\n",
        prefix_format="",
    )


def get_template(name: str) -> TemplateHF:
    if name not in _TEMPLATES:
        register_spatiallm_templates()
    if name not in _TEMPLATES:
        raise ValueError(f"Unknown template: {name}")
    return _TEMPLATES[name]

