import argparse
import logging
import json
import os
import pandas as pd
import torch._dynamo
from huggingface_hub import login
import numpy as np
from typing import List, Union
from sklearn.metrics import precision_score
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset, load_dataset
import os, json, math, torch, pandas as pd, re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from pathlib import Path

# Suppress torch._dynamo warnings
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Starting inference script...")

# Authenticate with Hugging Face (uses environment variable)

from huggingface_hub import login
# login(token="token_id")

import os, json, math, re, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import os, json, math, re, logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import os, json, math, re, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ----------------------------- Logging -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("persona-uncertainty")



import os
import re
import json
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --------------------------------- logging ---------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------- System prompt for a reasoning model components -----------------------------
CORE = (
    "Your task is to assign a triage category to each patient based on the provided factors.\n"
    "Use the following categories and output exactly one letter.\n"
    "Do not speculate beyond the provided information. Use clinical reasoning based only on what is given.\n"
    "Categories:\n"
    "  A : The patient should stay home\n"
    "  B : The patient should seek routine medical care\n"
    "  C : The patient should seek emergency care\n"
    "Your response must be a single uppercase letter: A, B, or C.\n"
)

# ----------------------------- System prompt for a normal model components -----------------------------

# CORE = (
#     "Your task is to assign a triage category to each patient based on the provided factors.\n"
#     "Use the following categories and output exactly one letter.\n"
#     "Do not speculate beyond the provided information. Use clinical reasoning based only on what is given.\n"
#     "Categories:\n"
#     "  A = The patient should stay home\n"
#     "  B = The patient should seek routine medical care\n"
#     "  C = The patient should seek emergency care\n"
#     "Your response must be a single uppercase letter: A, B, or C.\n"
#     "\n"
#     "Format your reply EXACTLY as:\n"
#     "## Final Response\n"
#     "A\n"
#     "Where the second line is ONE of: A, B, or C. Do not include any other text, punctuation, or quotes before or after the letter.\n"
# )


PERSONA = {
    "lead_physician": "You are an Emergency Department Physician.\n",
    "bold": "You are a Bold Emergency Department Physician.\n",
    "cautious": "You are a Cautious Emergency Department Physician.\n",
    "helpful": "You are a Helpful Assistant.\n",
    "nurse": "You are an Emergency Department Nurse.\n",
    "no_persona":""
}

def make_system_prompt(style: str) -> str:
    # backward-compat: map '' or None to 'helpful'
    style = (style or "helpful").lower()
    persona = PERSONA.get(style, PERSONA["helpful"])
    return persona + CORE


# ----------------------------- Helpers: chat template -----------------------------
def add_chat_template(example, tokenizer, persona_style="helpful"):
    system_prompt = make_system_prompt(persona_style)
    messages = [{"role": "system", "content": system_prompt}]
    if "input_prompt" in example and example["input_prompt"]:
        messages.append({"role": "user", "content": example["input_prompt"]})
    processed_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"input": processed_input, "system_prompt_used": system_prompt}


# ----------------------------- Helpers: scoring utilities -----------------------------
def logsumexp(vals: List[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))

def softmax_from_scores(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    Z = sum(exps.values())
    return {k: exps[k] / Z for k in scores}

def build_label_token_sets(tok, labels: List[str]) -> Dict[str, List[int]]:
    """
    Legacy boundary scorer: collect plausible next-token IDs at the boundary.
    Include plain, leading-space, and leading-newline variants to cover word-start tokens.
    """
    label_sets = {}
    for lab in labels:
        ids = set()
        for form in (lab, " " + lab, "\n" + lab):
            toks = tok(form, add_special_tokens=False).input_ids
            if toks:
                ids.add(toks[-1])
        if not ids:
            logger.warning("No token IDs found for label '%s'. Check tokenizer/labels.", lab)
        label_sets[lab] = sorted(ids)
    return label_sets

def build_label_variants(labels: List[str]) -> Dict[str, List[str]]:
    """
    Variants to handle common chat prefixes robustly (works for multi-token labels too).
    """
    prefixes = ["", " ", "\n"]
    return {L: [p + L for p in prefixes] for L in labels}

def score_label_sequence(
    mdl,
    tok,
    prefix_ids: torch.Tensor,   # 1D tensor of input_ids for the prompt prefix (no pad)
    candidate_texts: List[str],
) -> List[float]:
    """
    Returns log-probabilities for each candidate_text continuation given the encoded prefix.
    Robust for multi-token labels and insensitive to leading spaces/newlines.
    """
    device = mdl.device
    prefix_ids = prefix_ids.to(device)

    # Build batch [num_candidates, seq_len]
    batches = []
    lengths = []
    for cand in candidate_texts:
        cand_ids = tok(cand, add_special_tokens=False).input_ids
        full_ids = torch.tensor(
            (prefix_ids.tolist() + cand_ids),
            dtype=torch.long,
            device=device
        )
        batches.append(full_ids)
        lengths.append(len(cand_ids))

    # Pad
    pad_id = tok.pad_token_id or tok.eos_token_id
    max_len = max(x.size(0) for x in batches)
    input_ids = torch.full((len(batches), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batches):
        input_ids[i, :seq.size(0)] = seq
        attention_mask[i, :seq.size(0)] = 1

    with torch.no_grad():
        out = mdl(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]                 # [B, T-1, V]
        targets = input_ids[:, 1:]                     # [B, T-1]
        logprobs = torch.log_softmax(logits, dim=-1)   # [B, T-1, V]
        token_lp = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    prefix_len = prefix_ids.size(0)
    # first predicted token after prefix is at offset prefix_len-1 in token_lp
    start = prefix_len - 1
    seq_scores = []
    for i, cand_len in enumerate(lengths):
        end = start + cand_len
        seq_scores.append(float(token_lp[i, start:end].sum()))
    return seq_scores


# ----------------------------- Robust label parsing (tail-first, explicit > alias) -----------------------------
def default_aliases_for_abc() -> Dict[str, List[str]]:
    return {
    
        "A": [r"\bstay home\b", r"\bself[- ]?care\b", r"\bminor\b", r"\(A\)"],
        "B": [r"\broutine medical care\b", r"\broutine\b", r"\bnon[- ]?urgent\b", r"\(B\)"],
        "C": [r"\bemergency care\b", r"\burgent\b", r"\bemergency room\b",r"\bemergency\b", r"\(C\)"],
    }

def compile_label_patterns(labels: List[str], label_aliases: Dict[str, List[str]]
) -> Tuple[Dict[str, re.Pattern], Dict[str, List[re.Pattern]]]:
    """
    Returns:
      explicit_pats: dict[label] -> compiled explicit label regex (standalone)
      alias_pats:    dict[label] -> list of compiled alias regexes
    """
    explicit_pats: Dict[str, re.Pattern] = {}
    for L in labels:
        explicit_pats[L] = re.compile(rf"['\"*()]*\b{re.escape(L)}\b['\"*().]*", re.IGNORECASE)

    alias_pats: Dict[str, List[re.Pattern]] = {}
    for L in labels:
        pats = [re.compile(pat, re.IGNORECASE) for pat in label_aliases.get(L, [])]
        alias_pats[L] = pats
    return explicit_pats, alias_pats

def _clean_text_for_match(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<\|.*?\|>", "", text)  # remove control tags
    t = t.lstrip()
    t = re.sub(r"^(assistant|Assistant)\s*[:\-]*\s*", "", t)
    return t

def _last_match_label(segment: str, patterns: Dict[str, re.Pattern] | Dict[str, List[re.Pattern]]) -> Optional[str]:
    best_label, best_end = None, -1
    for L, pats in patterns.items():
        iterable = pats if isinstance(pats, list) else [pats]
        for p in iterable:
            for m in p.finditer(segment):
                if m and m.end() > best_end:
                    best_label, best_end = L, m.end()
    return best_label

# --- 3. Extract the "## Final Response" section ---
FINAL_HDR_RE = re.compile(r"^\s*##\s*Final\s*Response\s*$", re.IGNORECASE | re.MULTILINE)
ANY_HDR_RE   = re.compile(r"^\s*##\s*[^\n\r]+$", re.IGNORECASE | re.MULTILINE)

def extract_final_section(text: str) -> Optional[str]:
    m = FINAL_HDR_RE.search(text)
    if not m:
        return None
    start = m.end()
    nxt = ANY_HDR_RE.search(text, pos=start)
    end = nxt.start() if nxt else len(text)
    return text[start:end].strip()
    
    
def extract_label_from_final_response(
    text: str,
    labels: List[str],
    explicit_pats: Dict[str, re.Pattern],
    alias_pats: Dict[str, List[re.Pattern]],
) -> Optional[str]:
    sec = extract_final_section(text)
    if not sec:
        return None
    y = _last_match_label(sec, explicit_pats)
    if y:
        return y
    return _last_match_label(sec, alias_pats)

def extract_generated_label(
    text: str,
    labels: List[str],
    explicit_pats: Dict[str, re.Pattern],
    alias_pats: Dict[str, List[re.Pattern]],
    tail_chars: int = 500
) -> Optional[str]:
    t = _clean_text_for_match(text)
    if not t:
        return None

    # Extract final section if exists
    sec = extract_final_section(t)
    sec_to_search = sec if sec else t

    # --- Step 1: phrase-first in final section ---
    sec_low = sec_to_search.lower()
    phrase_map = {
        "A": ["stay home", "self-care", "minor"],
        "B": ["routine medical care", "routine", "non-urgent", "primary care"],
        "C": ["emergency care", "urgent", "ed visit"]
    }
    for L, phrases in phrase_map.items():
        for ph in phrases:
            if ph in sec_low:
                return L

    # --- Step 2: explicit letters in tail-first ---
    tail = sec_to_search[-tail_chars:]
    y = _last_match_label(tail, explicit_pats)
    if y: return y
    y = _last_match_label(tail, alias_pats)
    if y: return y

    # --- Step 3: full section fallback ---
    y = _last_match_label(sec_to_search, explicit_pats)
    if y: return y
    return _last_match_label(sec_to_search, alias_pats)
    
    
# ----------------------------- Token-level header matching -----------------------------
def build_header_variants(tok) -> List[List[int]]:
    """
    Tokenize a few common header variants (models often emit leading newlines).
    Returns list of token-id sequences for matching.
    """
    variants = ["## Final Response", "\n## Final Response", "\n\n## Final Response"]
    out = []
    for s in variants:
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids)
    return out

def find_subseq_end(seq: List[int], subseq: List[int], start: int = 0) -> int:
    """
    Return the END index (exclusive) of the first occurrence of subseq in seq[start:], else -1.
    """
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    i = start
    while i + m <= n:
        if seq[i:i+m] == subseq:
            return i + m
        i += 1
    return -1


# ----------------------------- Generation config -----------------------------
@dataclass
class GenConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 512  # keep rationale if the model wants to talk


# ----------------------------- Main -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str,
                        default="/gpfs/data/ceickhof/aabdul/Persona_LLM/data/ada_combined_prompts.csv")
    parser.add_argument("--text_column", type=str, default="input_prompt",
                        help="Column in CSV that contains the user prompt")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/gpfs/data/superlab/models/models--meta-llama-Llama-3.1-70B-Instruct")
    parser.add_argument("--labels", type=str, default="A,B,C",
                        help="Comma-separated labels, e.g. 'A,B,C' or 'home,routine,emergency'")
    parser.add_argument("--label_alias_json", type=str, default="",
                        help="Path to JSON file mapping label -> list of regex aliases. "
                             "If empty and labels=A,B,C, uses default triage aliases.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--out_dir", type=str,
                        default="outputs/inference_outputs")
    
    parser.add_argument(
    "--persona_style",
    type=str,
    default="helpful",
    choices=[
        "helpful", 
        "bold", 
        "cautious", 
        "lead_physician", 
        "nurse",
        "no_persona"
    ],
    help=(
        "Persona style to prepend: 'helpful' (default), 'bold', 'cautious', "
        "'lead_physician'"
        "'nurse', or 'no_persona'"))

    parser.add_argument(
        "--score_mode",
        type=str,
        default="sequence",
        choices=["sequence", "boundary"],
        help="How to score labels when header not found: 'sequence' (robust; default) or 'boundary' (single-step logits)."
    )

    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    assert len(labels) >= 2, "Provide at least 2 labels"

    # Aliases
    if args.label_alias_json:
        with open(args.label_alias_json, "r", encoding="utf-8") as fh:
            label_aliases: Dict[str, List[str]] = json.load(fh)
    else:
        label_aliases = default_aliases_for_abc() if labels == ["A","B","C"] else {L: [] for L in labels}

    # --- data
    df = pd.read_csv(args.annotation_path)
    df = df.tail(5)  # trim for quick runs; remove if you want full dataset
    assert args.text_column in df.columns, f"Missing column '{args.text_column}' in CSV"
    dataset = Dataset.from_pandas(df)

    # --- model/tokenizer
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if args.quantize else None
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(torch.float16 if qcfg is None else None),
        device_map="auto",
        quantization_config=qcfg
    ).eval()

    # Build inputs via chat template
    dataset = dataset.map(
        add_chat_template,
        fn_kwargs={"tokenizer": tok, "persona_style": args.persona_style},
        num_proc=1
    )
    logger.info("Example input:\n%s", dataset[0]["input"])

    prompts = dataset["input"]

    # Label helpers
    label_token_sets = build_label_token_sets(tok, labels)  # kept for 'boundary' fallback
    label_variants = build_label_variants(labels)           # used by sequence scoring
    explicit_pats, alias_pats = compile_label_patterns(labels, label_aliases)

    # generation config
    gcfg = GenConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )

   
    # Extract just the model name (last part after /)
    model_short = args.model_name_or_path.split('/')[-1]  # Gets "HuatuoGPT-o1-8B"
    
    # Use triage_bench as default dataset name
    dataset_name = "triage_bench"
    
    # output path reflects persona
    persona_tag = args.persona_style  # 'lead_physician' (or whatever)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{model_short}_{dataset_name}_{persona_tag}.jsonl"
    )

    header_id_variants = build_header_variants(tok)

    B = args.batch_size
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(prompts), B):
            batch = prompts[i:i+B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(mdl.device)
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()  # per-example input lengths

            # Generate FULL outputs; align to header in token IDs afterwards
            gen = mdl.generate(
                **enc,
                max_new_tokens=gcfg.max_new_tokens,
                do_sample=gcfg.do_sample,
                temperature=(gcfg.temperature if gcfg.do_sample else 1.0),
                top_p=(gcfg.top_p if gcfg.do_sample else 1.0),
                return_dict_in_generate=True,
                output_scores=False,  # set True if you want step-level probes
            )
            seqs = gen.sequences  # [B, padded_input_len + new_tokens]
            # Continuations for display (slice by common padded input length)
            cont = seqs[:, enc.input_ids.size(1):]
            gen_texts = tok.batch_decode(cont, skip_special_tokens=True)

            # Optional: legacy boundary logits for comparison fallback
            ff = mdl(**enc)
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            arange_b = torch.arange(enc.input_ids.size(0), device=ff.logits.device)
            last = ff.logits[arange_b, last_pos, :]

            for b in range(len(batch)):
                # ---- 1) locate the header boundary in TOKEN IDs ----
                full_ids = seqs[b].tolist()
                p_len = int(prompt_lens[b])
                boundary_end = -1
                for hv in header_id_variants:
                    boundary_end = find_subseq_end(full_ids, hv, start=p_len)
                    if boundary_end != -1:
                        break

                # ---- 2) score AFTER the header using sequence scoring (robust) ----
                class_scores: Dict[str, float] = {}
                score_point = "prompt_boundary"
                if boundary_end != -1:
                    prefix_ids = torch.tensor(full_ids[:boundary_end], dtype=torch.long, device=mdl.device)
                    for L in labels:
                        seq_scores = score_label_sequence(mdl, tok, prefix_ids, label_variants[L])
                        class_scores[L] = logsumexp(seq_scores)
                    score_point = "final_response"
                else:
                    # Fallback: chosen mode at prompt boundary
                    if args.score_mode == "sequence":
                        prefix_len = int(enc["attention_mask"][b].sum().item())
                        prefix_ids = enc["input_ids"][b, :prefix_len]
                        for L in labels:
                            seq_scores = score_label_sequence(mdl, tok, prefix_ids, label_variants[L])
                            class_scores[L] = logsumexp(seq_scores)
                    else:
                        for L in labels:
                            tids = label_token_sets[L]
                            vals = [float(last[b, tid]) for tid in tids] if tids else [float("-inf")]
                            class_scores[L] = logsumexp(vals)

                probs = softmax_from_scores(class_scores)
                pred_logit = max(class_scores, key=class_scores.get)

                # ---- 3) parse generated text ----
                gen_text = (gen_texts[b] or "").strip()
                # gen_label = extract_label_from_final_response(gen_text, labels, explicit_pats, alias_pats)
                # if gen_label is None:
                gen_label = extract_generated_label(gen_text, labels, explicit_pats, alias_pats)

                # ---- 4) metrics & write row ----
                pvec = np.array([probs[L] for L in labels], dtype=np.float64)
                pmax = float(pvec.max())
                top2 = float(np.partition(pvec, -2)[-2])
                conf_gen = float(probs.get(gen_label, 0.0)) if gen_label else 0.0
                entropy_bits = float(-(pvec * np.log2(pvec + 1e-12)).sum())
                margin = float(pmax - top2)
                
                row = {
                "input": batch[b],                            # Original input text for this example
                "labels": labels,                             # List of possible labels ['A', 'B', 'C']
                "class_scores": {L: class_scores[L] for L in labels},  # Model logit scores per label
                "class_probs": {L: probs[L] for L in labels},          # Softmax probabilities per label
                "prediction_logit": pred_logit,              # Model's predicted label from class_scores
                "generated_text": gen_text,                  # Decoded text generated by the model
                "generated_label": gen_label,                # Label extracted from generated text
                "conf": pmax,                                # Maximum probability across all labels
                "conf_gen": conf_gen,                         # Probability of the extracted/generated label
                "entropy_bits": entropy_bits,                # Entropy of the probability distribution
                "margin": margin,                            # Difference between top-1 and top-2 probabilities
                "persona_style": args.persona_style,         # Which assistant persona was used
                "score_mode": args.score_mode,               # Scoring mode used: 'sequence' or fallback
                "score_point": score_point,                  # Whether scoring used 'final_response' or 'prompt_boundary'
                "found_final_header": (boundary_end != -1),  # True if a ## Final Response header was detected
                "mismatch": (gen_label is not None and gen_label != pred_logit),  # True if extracted label disagrees with model logit
             
                }


                f.write(json.dumps(row) + "\n")

    print(f"Saved: {out_path}")


# -----------------------------
if __name__ == "__main__":
    main()
