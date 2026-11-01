import json

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

vocab_path = model.get_path_to_vocab_file()
print("vocab file:", vocab_path)

with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)  # {token_string: token_id}

print("vocab size (vocab.json):", len(vocab))

print("\nFirst 15 entries:")
for token, token_id in list(vocab.items())[:15]:
    print(f"  id={token_id:<6} raw={token!r}")

print("\nRaw vocab string vs decode():")
for word in [" Paris", " capital", "\n", "{", '"', "true", " 123"]:
    ids = model.encode(word)[0].tolist()
    for i in ids:
        raw = next((t for t, v in vocab.items() if v == i), None)
        print(f"  text={word!r:<12} id={i:<6} raw={raw!r:<12} decoded={model.decode([i])!r}")

print("\nHow numbers/JSON are split into tokens:")
for text in ["12345", "3.14", '{"a": 2.0}', "-42,"]:
    ids = model.encode(text)[0].tolist()
    print(f"  {text!r:<14} -> {[model.decode([i]) for i in ids]}")
