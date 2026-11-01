from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
prompt = "The capital of France is"

ids = model.encode(prompt)[0].tolist()

print("Prompt:", repr(prompt))
print("\nGenerating 10 tokens (greedy):")

for step in range(10):
    logits = model.get_logits_from_input_ids(ids)
    next_id = max(range(len(logits)), key=lambda i: logits[i])
    ids.append(next_id)
    print(f"  step {step + 1}: id={next_id} token={model.decode([next_id])!r}")

print("\nFull text:", repr(model.decode(ids)))

ids = model.encode(prompt)[0].tolist()
logits = model.get_logits_from_input_ids(ids)
best = max(range(len(logits)), key=lambda i: logits[i])
best_score = logits[best]
logits[best] = float("-inf")  # this token can never be chosen now
second = max(range(len(logits)), key=lambda i: logits[i])

print("\n--- Banning the top token ---")
print(f"Banned : {model.decode([best])!r} (logit {best_score:.2f})")
print(f"New top: {model.decode([second])!r} (logit {logits[second]:.2f})")
