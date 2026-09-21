#!/usr/bin/env python
"""Retroactively run the flan-t5 'main-2' extractor on the captions of runs that never
had term-extraction (v1, v2), so every run shows LLM terms. Runs on CPU (won't touch the GPU).
Builds per-step cumulative top-8 snapshots (same structure as blip_terms.json) and merges in."""
import json, re
from collections import Counter
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32).to("cpu").eval()
tok = proc.tokenizer

QUANT = r"^(a|an|the|one|two|three|four|five|several|some|many|group of|couple of|pair of)\s+"


def main2(caps):
    joined = " ; ".join(caps)
    prompt = ("Below are descriptions of one image. Name the 2 main things shown "
              "(a person or animal, and an object or shape), as two comma-separated nouns; "
              "always include a person if one is mentioned.\n"
              "Descriptions: a smiling woman in a red hat ; a woman wearing a hat near a round mirror\n"
              "Two main things: woman, hat\n"
              f"Descriptions: {joined}\n"
              "Two main things:")
    with torch.no_grad():
        inp = tok(prompt, return_tensors="pt")
        out = model.language_model.generate(input_ids=inp.input_ids, max_new_tokens=12)
    raw = tok.decode(out[0], skip_special_tokens=True).strip().lower()
    terms = []
    for w in re.split(r"[,/]| and ", raw):
        w = re.sub(QUANT, "", w.strip().strip(".").strip())
        if w and w not in terms:
            terms.append(w)
    return terms[:2]


b = json.load(open("RESULTS/heldout/blip_negatives.json"))
out = json.load(open("RESULTS/heldout/blip_terms.json"))
DOC = ["faces", "bodies", "places", "words", "words2",
       "v2_faces", "v2_faces_crop", "v2_bodies", "v2_bodies_crop", "v2_places", "v2_places_crop",
       "v2_words_1tok_full", "v2_words_1tok_crop", "v2_words_2tok_full", "v2_words_2tok_crop"]
for run in DOC:
    if run in out:
        continue
    steps = b.get(run)
    if not steps:
        continue
    counts, snaps = Counter(), []
    for cs in steps:
        caps = cs if isinstance(cs, list) else [cs]
        for t in main2(caps):
            counts[t] += 1
        snaps.append([w for w, _ in counts.most_common(8)])
    out[run] = snaps
    print(run, "->", snaps[-1], flush=True)
json.dump(out, open("RESULTS/heldout/blip_terms.json", "w"), indent=1)
print("DONE ALL-TERMS", flush=True)
