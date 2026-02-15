#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Prompts / Personas
# ------------------------------------------------------------------------------

CORE = (
    "Your task is to assign a triage category to each patient based on the provided factors.\n"
    "Use the following categories and output exactly one letter.\n"
    "Do not speculate beyond the provided information. Use clinical reasoning based only on what is given.\n"
    "Categories:\n"
    "  A = The patient should stay home\n"
    "  B = The patient should seek routine medical care\n"
    "  C = The patient should seek emergency care\n"
    "Your response must be a single uppercase letter: A, B, or C.\n"
    "\n"
    "Format your reply EXACTLY as:\n"
    "## Final Response\n"
    "A\n"
    "Where the second line is ONE of: A, B, or C. Do not include any other text, punctuation, or quotes before or after the letter.\n"
)

PERSONA = {
    "lead_physician": "You are an Emergency Department Physician.\n",
    "bold": "You are a Bold Emergency Department Physician.\n",
    "cautious": "You are a Cautious Emergency Department Physician.\n",
    "helpful": "You are a Helpful Assistant.\n",
    "nurse": "You are an Emergency Department Nurse.\n",
    "no_persona": ""
}

def make_system_prompt(style: str) -> str:
    style = (style or "helpful").lower()
    persona = PERSONA.get(style, PERSONA["helpful"])
    return persona + CORE

# ------------------------------------------------------------------------------
# Chat templating
# ------------------------------------------------------------------------------

def add_chat_template(example, tokenizer, persona_style="helpful"):
    system_prompt = make_system_prompt(persona_style)
    messages = [{"role": "system", "content": system_prompt}]
    if "input_prompt" in example and example["input_prompt"]:
        messages.append({"role": "user", "content": example["input_prompt"]})
    processed_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"input": processed_input, "system_prompt_used": system_prompt}

# ------------------------------------------------------------------------------
# Scoring utilities (numerically robust)
# ------------------------------------------------------------------------------

def logsumexp(vals: List[float]) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))

def softmax_from_scores(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores.keys())
    vals = [scores[k] for k in keys]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        p = 1.0 / len(keys)
        return {k: p for k in keys}
    cleaned = [v if math.isfinite(v) else -1e30 for v in vals]
    m = max(cleaned)
    exps = [math.exp(v - m) for v in cleaned]
    Z = sum(exps)
    return {k: exps[i] / Z for i, k in enumerate(keys)}

def build_label_token_sets(tok, labels: List[str]) -> Dict[str, List[int]]:
    label_sets = {}
    for lab in labels:
        ids = set()
        for form in (lab, " " + lab, "\n" + lab):
            toks = tok(form, add_special_tokens=False).input_ids
            if toks:
                ids.add(toks[-1])
        label_sets[lab] = sorted(ids)
    return label_sets

def build_label_variants(labels: List[str]) -> Dict[str, List[str]]:
    prefixes = ["", " ", "\n"]
    return {L: [p + L for p in prefixes] for L in labels}

def default_aliases_for_abc() -> Dict[str, List[str]]:
    return {
        "A": [r"\bstay home\b", r"\bself[- ]?care\b", r"\bminor\b", r"\(A\)"],
        "B": [r"\broutine medical care\b", r"\broutine\b", r"\bnon[- ]?urgent\b", r"\(B\)"],
        "C": [r"\bemergency care\b", r"\burgent\b", r"\bemergency room\b", r"\bemergency\b", r"\(C\)"],
    }

def compile_label_patterns(labels: List[str], label_aliases: Dict[str, List[str]]
) -> Tuple[Dict[str, re.Pattern], Dict[str, List[re.Pattern]]]:
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
    t = re.sub(r"<\|.*?\|>", "", text)
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

    sec = extract_final_section(t)
    sec_to_search = sec if sec else t

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

    tail = sec_to_search[-tail_chars:]
    y = _last_match_label(tail, explicit_pats)
    if y: return y
    y = _last_match_label(tail, alias_pats)
    if y: return y
    y = _last_match_label(sec_to_search, explicit_pats)
    if y: return y
    return _last_match_label(sec_to_search, alias_pats)

# ------------------------------------------------------------------------------
# Token-level header matching
# ------------------------------------------------------------------------------

def build_header_variants(tok) -> List[List[int]]:
    variants = ["## Final Response", "\n## Final Response", "\n\n## Final Response"]
    out = []
    for s in variants:
        ids = tok(s, add_special_tokens=False).input_ids
        if ids:
            out.append(ids)
    return out

def find_subseq_end(seq: List[int], subseq: List[int], start: int = 0) -> int:
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    i = start
    while i + m <= n:
        if seq[i:i+m] == subseq:
            return i + m
        i += 1
    return -1

# ------------------------------------------------------------------------------
# Generation config
# ------------------------------------------------------------------------------

@dataclass
class GenConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 512

# ------------------------------------------------------------------------------
# Sequence scorer (device-safe for sharded models)
# ------------------------------------------------------------------------------

def score_label_sequence(
    mdl,
    tok,
    prefix_ids: torch.Tensor,
    candidate_texts: List[str],
) -> List[float]:
    device = prefix_ids.device
    prefix_ids = prefix_ids.to(device)

    batches, lengths = [], []
    for cand in candidate_texts:
        cand_ids = tok(cand, add_special_tokens=False).input_ids
        full_ids = torch.tensor(prefix_ids.tolist() + cand_ids, dtype=torch.long, device=device)
        batches.append(full_ids)
        lengths.append(len(cand_ids))

    pad_id = tok.pad_token_id or tok.eos_token_id
    max_len = max(x.size(0) for x in batches)
    input_ids = torch.full((len(batches), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batches):
        input_ids[i, :seq.size(0)] = seq
        attention_mask[i, :seq.size(0)] = 1

    with torch.no_grad():
        out = mdl(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    prefix_len = prefix_ids.size(0)
    start = prefix_len - 1
    seq_scores = []
    for i, cand_len in enumerate(lengths):
        end = start + cand_len
        if cand_len == 0 or start < 0 or end > token_lp.size(1):
            seq_scores.append(float("-inf"))
        else:
            v = token_lp[i, start:end]
            seq_scores.append(float(v.sum().item()) if torch.isfinite(v).all() else float("-inf"))
    return seq_scores

# ------------------------------------------------------------------------------
# Boundary fallback scorer (single-step, great for A/B/C)
# ------------------------------------------------------------------------------

def boundary_class_scores(mdl, tok, enc, b_ix, labels):
    with torch.no_grad():
        out = mdl(**enc)
    last_pos = enc["attention_mask"].sum(dim=1) - 1
    arange_b = torch.arange(enc["input_ids"].size(0), device=out.logits.device)
    last = out.logits[arange_b, last_pos, :]  # [B, V]
    lp = torch.log_softmax(last[b_ix], dim=-1)

    lab2vals = {L: [] for L in labels}
    for L in labels:
        for form in (L, " " + L, "\n" + L):
            ids = tok(form, add_special_tokens=False).input_ids
            if ids:
                lab2vals[L].append(float(lp[ids[-1]].item()))
    return {L: logsumexp(vs) for L, vs in lab2vals.items()}

# ------------------------------------------------------------------------------
# PMI / contextual calibration helpers
# ------------------------------------------------------------------------------

def abc_log_scores_from_boundary_lp(lp: torch.Tensor, tok, labels: List[str]) -> Dict[str, float]:
    """Boundary-style log-scores for A/B/C via log-sum-exp over token variants."""
    def _lse(xs):
        xs = [x for x in xs if math.isfinite(x)]
        if not xs: return float("-inf")
        m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))
    scores = {}
    for L in labels:
        vals = []
        for form in (L, " "+L, "\n"+L):
            ids = tok(form, add_special_tokens=False).input_ids
            if ids:
                vals.append(float(lp[ids[-1]].item()))
        scores[L] = _lse(vals)
    return scores

def softmax_from_log_scores(log_scores: Dict[str, float]) -> Dict[str, float]:
    m = max(log_scores.values())
    exps = {k: math.exp(v - m) for k, v in log_scores.items()}
    Z = sum(exps.values())
    return {k: exps[k] / Z for k in log_scores}

def make_null_prompt(original_prompt: str, pattern: str) -> str:
    """Replace patient-specific content with a neutral stub to estimate label prior."""
    return re.sub(pattern, r"\1 N/A.", original_prompt, flags=re.IGNORECASE)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    import argparse

    # Enable TF32 for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str,
                        default="data/triage_prompts.csv")
    parser.add_argument("--text_column", type=str, default="input_prompt")
    parser.add_argument("--model_name_or_path", type=str,
                        default="google/medgemma-27b-text-it")
    parser.add_argument("--labels", type=str, default="A,B,C")
    parser.add_argument("--label_alias_json", type=str, default="")
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
        help="Persona style to prepend."
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="sequence",
        choices=["sequence", "boundary"],
        help="Fallback when header not found."
    )
    # NEW: optional PMI calibration to reduce 'always C' prior
    parser.add_argument("--calibrate_pmi", action="store_true",
                        help="Subtract per-label null-context scores to remove label prior ('always C').")
    parser.add_argument("--null_replace_regex", type=str,
                        default=r"(Patient information:)[\s\S]*",
                        help="Regex to match patient-specific content; replaced with 'N/A.' for the null prompt.")

    args = parser.parse_args()
    logger.info("Starting inference script...")

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    assert len(labels) >= 2, "Provide at least 2 labels"

    # Aliases
    if args.label_alias_json:
        with open(args.label_alias_json, "r", encoding="utf-8") as fh:
            label_aliases: Dict[str, List[str]] = json.load(fh)
    else:
        label_aliases = default_aliases_for_abc() if labels == ["A", "B", "C"] else {L: [] for L in labels}

    # Data
    df = pd.read_csv(args.annotation_path)
    df = df.head(5)  # trim for quick runs; adjust/remove as needed
    assert args.text_column in df.columns, f"Missing column '{args.text_column}' in CSV"
    dataset = Dataset.from_pandas(df)

    # Tokenizer / Model (bf16 + SDPA; quant optional)
    model_id = args.model_name_or_path
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Stop ids: <eos>, and <end_of_turn> if present
    stop_ids = [tok.eos_token_id]
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot_id not in (None, -1):
        stop_ids.append(eot_id)

    qcfg = None
    if args.quantize:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Prefer bf16 + SDPA; fall back to fp16 if necessary
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.bfloat16 if qcfg is None else None),
            attn_implementation="sdpa",
            device_map="auto",
            quantization_config=qcfg
        ).eval()
    except Exception as e:
        logger.warning("bf16/SDPA load failed (%s). Falling back to fp16.", e)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
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
    label_token_sets = build_label_token_sets(tok, labels)
    label_variants    = build_label_variants(labels)
    explicit_pats, alias_pats = compile_label_patterns(labels, label_aliases)

    # Generation config (we pass via kwargs)
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

            # Encode WITHOUT truncation (silence warning; chat template already bounded)
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False)
            # Let HF dispatch devices automatically; do not force to mdl.device
            prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

            # Generate with proper stop ids; force at least 1 token for visibility
            gen_kwargs = {
                "max_new_tokens": gcfg.max_new_tokens,
                "min_new_tokens": 1,
                "eos_token_id": stop_ids,
                "pad_token_id": tok.pad_token_id,
                "return_dict_in_generate": True,
                "output_scores": False,
            }
            if gcfg.do_sample:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": gcfg.temperature,
                    "top_p": gcfg.top_p,
                })
            else:
                gen_kwargs.update({"do_sample": False})

            gen = mdl.generate(**enc, **gen_kwargs)
            seqs = gen.sequences  # [B, max_input_len + new_tokens]

            # Per-example decode using each sample's prompt length
            gen_texts = []
            for b in range(len(batch)):
                cont_ids = seqs[b, int(prompt_lens[b]):]
                gen_texts.append(tok.decode(cont_ids, skip_special_tokens=True).strip())

            # One forward pass for boundary logits (used for fallback/analysis)
            ff = mdl(**enc)
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            arange_b = torch.arange(enc["input_ids"].size(0), device=ff.logits.device)
            last = ff.logits[arange_b, last_pos, :]

            for b in range(len(batch)):
                # Locate "## Final Response" header in TOKEN IDs (after prompt)
                full_ids = seqs[b].tolist()
                p_len = int(prompt_lens[b])
                boundary_end = -1
                for hv in header_id_variants:
                    boundary_end = find_subseq_end(full_ids, hv, start=p_len)
                    if boundary_end != -1:
                        break

                # Score labels
                class_scores: Dict[str, float] = {}
                score_point = "prompt_boundary"

                if boundary_end != -1:
                    # Robust sequence scoring AFTER header
                    prefix_ids = torch.tensor(full_ids[:boundary_end], dtype=torch.long,
                                              device=enc["input_ids"].device)
                    for L in labels:
                        seq_scores = score_label_sequence(mdl, tok, prefix_ids, label_variants[L])
                        class_scores[L] = logsumexp(seq_scores)
                    score_point = "final_response"
                else:
                    # Fallback per selected mode
                    if args.score_mode == "sequence":
                        prefix_len = int(prompt_lens[b])
                        prefix_ids = enc["input_ids"][b, :prefix_len]
                        for L in labels:
                            seq_scores = score_label_sequence(mdl, tok, prefix_ids, label_variants[L])
                            class_scores[L] = logsumexp(seq_scores)
                    else:
                        # Legacy boundary logits at prompt end
                        lab2vals = {L: [] for L in labels}
                        lp = torch.log_softmax(last[b], dim=-1)
                        for L in labels:
                            tids = label_token_sets[L]
                            vals = [float(lp[tid].item()) for tid in tids] if tids else []
                            lab2vals[L] = vals
                        class_scores = {L: logsumexp(vs) for L, vs in lab2vals.items()}

                # ---- Optional PMI calibration to remove 'always C' prior ----
                raw_class_scores = dict(class_scores)  # keep a copy for logging
                null_class_scores = None
                if args.calibrate_pmi:
                    try:
                        # Build a null prompt for this example (remove patient specifics)
                        orig_prompt_text = batch[b]
                        null_prompt_text = make_null_prompt(orig_prompt_text, args.null_replace_regex)

                        enc_null = tok([null_prompt_text], return_tensors="pt", padding=True, truncation=False)
                        # Move to model device block-wise
                        device = next(iter(mdl.parameters())).device
                        for k in enc_null:
                            enc_null[k] = enc_null[k].to(device)

                        with torch.no_grad():
                            out_null = mdl(**enc_null)
                        last_pos_null = enc_null["attention_mask"].sum(dim=1) - 1
                        lp_null = torch.log_softmax(out_null.logits[0, last_pos_null.item(), :], dim=-1)

                        # Compute boundary-style label scores on the null prompt
                        null_class_scores = abc_log_scores_from_boundary_lp(lp_null, tok, labels)

                        # PMI-style calibration
                        class_scores = {L: class_scores[L] - null_class_scores.get(L, 0.0) for L in labels}
                    except Exception as e:
                        logger.warning("PMI calibration failed for example %d: %s", i + b, e)

                # Convert to probabilities / pick argmax
                probs = softmax_from_scores(class_scores)
                pred_logit = max(class_scores, key=class_scores.get)

                gen_text = (gen_texts[b] or "").strip()
                gen_label = extract_generated_label(gen_text, labels, explicit_pats, alias_pats)

                pvec = np.array([probs[L] for L in labels], dtype=np.float64)
                if np.isfinite(pvec).all():
                    pmax = float(pvec.max())
                    if len(labels) >= 2:
                        top2 = float(np.partition(pvec, -2)[-2])
                        margin = float(pmax - top2)
                    else:
                        margin = float("nan")
                    entropy_bits = float(-(pvec * np.log2(pvec + 1e-12)).sum())
                else:
                    pmax = float("nan")
                    margin = float("nan")
                    entropy_bits = float("nan")

                conf_gen = float(probs.get(gen_label, 0.0)) if gen_label else 0.0

                row = {
                    "input": batch[b],
                    "labels": labels,
                    "class_scores": {L: class_scores[L] for L in labels},     # possibly calibrated
                    "class_probs": {L: probs[L] for L in labels},
                    "prediction_logit": pred_logit,
                    "generated_text": gen_text,
                    "generated_label": gen_label,
                    "conf": pmax,
                    "conf_gen": conf_gen,
                    "entropy_bits": entropy_bits,
                    "margin": margin,
                    "persona_style": args.persona_style,
                    "score_mode": args.score_mode,
                    "score_point": score_point,
                    "found_final_header": (boundary_end != -1),
                    "mismatch": (gen_label is not None and gen_label != pred_logit),
                    "pmi_calibrated": bool(args.calibrate_pmi),
                }
                if args.calibrate_pmi:
                    row["raw_class_scores"] = raw_class_scores
                    row["null_class_scores"] = null_class_scores

                f.write(json.dumps(row) + "\n")

    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
