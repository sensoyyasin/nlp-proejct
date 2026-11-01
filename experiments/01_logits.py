from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
ids = model.encode("The capital of Turkey is")[0].tolist()
print("input ids: ", ids)
print("tokens : ", [model.decode([i]) for i in ids])

logits = model.get_logits_from_input_ids(ids)
print("logit count : ", len(logits))

top = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
for t in top:
    print(t, repr(model.decode([t])), round(logits[t], 2))


