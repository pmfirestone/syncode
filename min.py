"""min.py: minimal benchmarking example using cuda tracing."""

# Import requirements.
import tempfile
import re
import json

import torch
from syncode import SyncodeLogitsProcessor, Grammar
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up model and logits_processor.
model_id = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
logits_processor = SyncodeLogitsProcessor(Grammar("python"), tokenizer, mode='grammar_mask', parser='lr')

# Set up prompt and prime for input.
prompt = '''def print_prime(n):
   """
   Print all primes between 1 and n
   """
'''
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
logits_processor.reset(prompt)

def time_callable(func: callable) -> float:
    """Time how long a GPU func takes."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    func()
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    return start_event.elapsed_time(end_event)


syncode_times = [time_callable(lambda: model.generate(
        **inputs, logits_processor=[logits_processor], max_new_tokens=50
)) for _ in range(10)]


vanilla_times = [time_callable(lambda: model.generate(
        **inputs, max_new_tokens=50
)) for _ in range(10)]


# Profile the generation of responding to the prompt.
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_modules=True,
    with_stack=True
) as p:
    outputs = model.generate(
        **inputs, logits_processor=[logits_processor], max_new_tokens=50
    )

# Save the trace to a temporary file. This is a weird kludge, since the names
# of the events in the profile object are not the same as those that end up in
# the trace file. The only work around I've been able to find is sending the
# trace through this export method to the disk and then reading it back in to
# memory for further processing. Reading through the source code implementing
# this method has not clarified my confusion.
_, trace_file = tempfile.mkstemp()
p.export_chrome_trace(trace_file)


def task_durations(task_regex, input_file):
    """Get a list of the durations of the task in the input trace file."""
    with open(input_file) as f:
        trace = json.load(f)
        res = []
        for event in trace["traceEvents"]:
            if task_regex.match(event["name"]):
                res.append(event["dur"])
    return res


# Total time in syncode.
syncode_tot = sum(
    task_durations(
        # This regex is very fragile and depends on how syncode was installed.
        re.compile(r"evaluation/\.\./grammar_decoder\.py\(\d+\): __call__"), trace_file
    )
)

# Total time to generate.
generate_tot = sum(
    task_durations(
        re.compile(r"transformers/generation/utils\.py\(\d+\): generate"), trace_file
    )
)
