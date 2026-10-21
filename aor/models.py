"""The six models of the main comparison and their sampling settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20          # 0 disables top-k filtering


#: short name -> (Hugging Face id, sampling settings recommended by the model's developers)
MODELS: Dict[str, Tuple[str, Sampling]] = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", Sampling(0.7, 0.8, 20)),
    "qwen3-4b": ("Qwen/Qwen3-4B", Sampling(0.7, 0.8, 20)),
    "qwen3-8b": ("Qwen/Qwen3-8B", Sampling(0.7, 0.8, 20)),
    "qwen3-14b": ("Qwen/Qwen3-14B", Sampling(0.7, 0.8, 20)),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", Sampling(0.6, 0.9, 0)),
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", Sampling(0.7, 0.9, 0)),
}
#: Hugging Face id -> the model revision of the paper's runs
REVISIONS: Dict[str, str] = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
LABELS = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
          "llama3.1-8b": "Llama-3.1-8B", "mistral-7b": "Mistral-7B"}


def resolve_model(name: str) -> Tuple[str, Sampling]:
    """Hugging Face id and sampling settings for a short name of the paper or any model id."""
    if name in MODELS:
        return MODELS[name]
    for hf_id, sp in MODELS.values():
        if hf_id == name:
            return hf_id, sp
    low = name.lower()
    if "qwen3" in low:
        return name, Sampling(0.7, 0.8, 20)
    if "llama-3" in low:
        return name, Sampling(0.6, 0.9, 0)
    return name, Sampling(0.7, 0.9, 0)


def short_name(name: str) -> str:
    """The short name of a model of the paper, or a file-system friendly form of any other model id."""
    if name in MODELS:
        return name
    for short, (hf_id, _) in MODELS.items():
        if hf_id == name:
            return short
    return name.rstrip("/").split("/")[-1].lower()
