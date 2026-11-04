import json
from llm_sdk import Small_LLM_Model


'''
Vocab is the fixed list of every token the model knows.
     - Every token has a number (ID). The model only works with these numbers.
     - It's fixed. Qwen3's vocab size about 151K.
     - Tokens aren't the same as words.  (Paris-> 1 word), (get_weather) -> maybe get, _, weather -> 3 words.
     - get_logits_from_input_ids returns 151,936 numbers, logits[12095] is the score for token 12095. One score per vocab entry.
     - vocab.json is the vocabulary written to a file. It's a dictionary. {token_string: id}
'''

model = Small_LLM_Model()

# 1 - Find the vocab file
vocab_path = model.get_path_to_vocab_file()
print("Vocab file:", vocab_path)

# 2 - Load it : a dictionary {token_string : token_id}
with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)

print("Number of entries in vocab.json:", len(vocab))
print()

# 3 - Print the first 15 entries.
print("First 15 entries:")
count = 0
for token_string, token_id in vocab.items():
    print(f"{token_id<7} {repr(token_string)}")
    count += 1
    if count == 15:
        break

print()

# 4 - Look up some tokens we saw before
examples = ["The", "Ġcapital", "ĠParis", "ĠAnkara", "Ċ", "{", "\"", "Ġtrue"]
print("Some specific tokens:")

for token_string in examples:
    if token_string in vocab:
        token_id = vocab[token_string]
        decoded = model.decode([token_id])
        print(f"  vocab key={repr(token_string):<12} id={token_id:<7} decoded={repr(decoded)}")
    else:
        print(f"  {repr(token_string)} is not in vocab.json")

