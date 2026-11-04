from llm_sdk import Small_LLM_Model

# 1 - Load the model
model = Small_LLM_Model()

prompt =  "The capital of Turkey is"
print("Prompt : ", prompt)


# 2 - Convert the text into token IDs
# encode() returns a tensor with shape [1, n] (1 row, n tokens).
encoded = model.encode(prompt)

first_row = encoded[0]
input_ids = first_row.tolist()

# 3 - Show which piece of text each ID corresponds to
tokens = []
for token_id in input_ids:
    # decode() expects a list of ids.
    text = model.decode([token_id])
    tokens.append(text)

print("Tokens : ", tokens)

# 4 - Scores (logits) for the next token
logits = model.get_logits_from_input_ids(input_ids)

vocab_size = len(logits)
print("Number of logits (vocab size):", vocab_size)

# 5 - Pair each token ID with its score : (id, score)
pairs = []
for token_id in range(vocab_size):
    score = logits[token_id]
    pairs.append((token_id, score))


# 6 - Sort the pairs by score, from highest to lowest
def get_score(pair):
    token_id, score = pair
    return score

pairs.sort(key=get_score, reverse=True)


# 7 - Print the 5 tokens with the highest score
top_k = 5
print()
print(f"Top {top_k} most likely next tokens:")

for rank in range(top_k):
    token_id, score = pairs[rank]
    text = model.decode([token_id])
    print(f"{rank + 1}. id={token_id:<7} text={repr(text):<15} score={round(score, 2)}")
