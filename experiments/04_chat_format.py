from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

# Qwen3 chat format, built by hand. The empty <think></think> block turns thinking off.
system = (
    "You convert a user request into a function call. "
    "Available functions:\n"
    "- fn_add_numbers(a: number, b: number): Add two numbers.\n"
    "- fn_greet(name: string): Greet a person.\n"
    "- fn_reverse_string(s: string): Reverse a string.\n"
    'Reply with JSON: {"name": ..., "parameters": {...}}'
)

user = "What is the sum of 2 and 3?"

prompt = (
    f"<|im_start|>system\n{system}<|im_end|>\n"
    f"<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

ids = model.encode(prompt)[0].tolist()
print("Prompt length (tokens):", len(ids))
print("Last 8 prompt tokens:", [model.decode([i]) for i in ids[-8:]])

# Special tokens: how are they encoded?
for special in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    enc = model.encode(special)[0].tolist()
    print(f"{special!r} -> ids {enc}")

# Unconstrained greedy generation, to see what the raw model does
print("\n--- Unconstrained output ---")
generated = []
for _ in range(60):
    logits = model.get_logits_from_input_ids(ids)
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    ids.append(next_id)
    generated.append(next_id)
    print(f"id={next_id:<7} token={model.decode([next_id])!r}")
print("\nFull output:\n" + model.decode(generated))
