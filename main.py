"""main.py: experimental process-parallel dispatching."""

import multiprocessing

import torch

from syncode import SyncodeLogitsProcessor, Grammar
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "google/gemma-2-2b-it"
language = "python"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

syncode = SyncodeLogitsProcessor(Grammar(language), tokenizer, parser="lr", mode="grammar_mask")

syncode_here, syncode_there = multiprocessing.Pipe()
model_here, model_there = multiprocessing.Pipe()


def respond_with_accept_mask():
    """Message handler for syncode process."""
    while True:
        input_ids = syncode_there.recv()
        accept_mask = syncode.get_accept_mask(input_ids)
        syncode_there.send(accept_mask)


def respond_with_scores():
    """Message handler for model process."""
    while True:
        input_ids = model_there.recv()
        scores = torch.softmax(model(input_ids).logits[:, -1, :], dim=-1)
        model_there.send(scores)


syncode_process = multiprocessing.Process(target=respond_with_accept_mask, name="syncode")
model_process = multiprocessing.Process(target=respond_with_scores, name="model")


syncode_process.start()
model_process.start()
