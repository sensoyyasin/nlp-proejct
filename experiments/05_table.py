import time
from llm_sdk import Small_LLM_Model

# Token table: token id -> real text
# The model gives one score (logit) per token id: 151,936 scores. To decide which tokens are allowed, we need the text of every token
# We're building this table once at a startuo : table[token_id] = text

# Special token (id >= 151643) : <|im_end|>, <think>, <tool_call>, ...
FIRST_SPECIAL_ID = 151643

model = Small_LLM_Model()

# Table size = number of logits, so table[i] and logits[i] are the same token
logits = model.get_logits_from_input_ids([0])
vocab_size = len(logits)
print("vocab size:", vocab_size)


# Build the table
start = time.time()
table = []
special_count = 0
broken_count = 0

for token_id in range(vocab_size):
    text = model.decode([token_id])

    if token_id >= FIRST_SPECIAL_ID:
        table.append(None)
        special_count += 1
    elif text == "" or "\ufffd" in text:
        table.append(None)
        broken_count += 1
    else:
        table.append(text)

print(f"built in {time.time() - start:.1f} s")
print("special (id >= 151643):", special_count)
print("broken  (empty / partial):", broken_count)
print("usable:", table.count(None) and vocab_size - table.count(None))
print()


# Look at a few known tokens
for token_id in [70244, 279, 1112, 151645, 151657]:
    print(f"id={token_id:<7} text={repr(table[token_id])}")
print()


# Round-trip test: joining the pieces must give back the original text
for text in ["What is the sum of 2 and 3?", 'Reverse "hello"', "Merhaba dünya"]:
    ids = model.encode(text)[0].tolist()
    pieces = [table[i] for i in ids]

    print(repr(text))
    print("  pieces:", pieces)

    if None in pieces:
        print("  -> contains a None piece")
    else:
        print("  -> rebuilt OK:", "".join(pieces) == text)
