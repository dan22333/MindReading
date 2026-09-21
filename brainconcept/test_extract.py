#!/usr/bin/env python
"""Show BLIP captions per turn -> new flan-t5 'main 2 things' extractor output, on real turns."""
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
tok = proc.tokenizer


def main2(captions):
    joined = " ; ".join(captions)
    prompt = ("Below are descriptions of one image. Name the 2 main things shown "
              "(a person or animal, and an object or shape), as two comma-separated nouns; "
              "always include a person if one is mentioned.\n"
              "Descriptions: a smiling woman in a red hat ; a woman wearing a hat near a round mirror\n"
              "Two main things: woman, hat\n"
              f"Descriptions: {joined}\n"
              "Two main things:")
    with torch.no_grad():
        inp = tok(prompt, return_tensors="pt").to("cuda")
        out = model.language_model.generate(input_ids=inp.input_ids, max_new_tokens=12)
    return tok.decode(out[0], skip_special_tokens=True).strip().lower()


EXAMPLES = [
    ("step 250", ["man wearing a red and blue dragon mask",
                  "man wearing a red and blue devil mask",
                  "man with a red and blue dragon mask on his head"]),
    ("step 500", ["two women dressed as devils posing for a photo",
                  "two women dressed up as devils",
                  "two women dressed as devils holding bottles of liquor"]),
    ("step 750", ["two women in lab coats standing in front of a blue background",
                  "two women in lab coats standing next to each other",
                  "two women in lab coats standing in front of a blue moon"]),
    ("bodies",   ["a man swinging a baseball bat", "a baseball player at bat",
                  "a man in a uniform hitting a baseball"]),
]
for label, caps in EXAMPLES:
    print(f"=== {label} ===")
    for c in caps:
        print("  BLIP:", c)
    print("  LLM main-2 ->", main2(caps))
    print()
