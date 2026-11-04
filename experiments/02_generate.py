from llm_sdk import Small_LLM_Model

def argmax(values):
    # return the index of the largest value in a list.
    best_index = 0
    best_value = values[0]

    for i in range(1, len(values)):
        if values[i] > best_value:
            best_value = values[i]
            best_index = i

    return best_index

# 1 - Load the model
model = Small_LLM_Model()

prompt =  "The capital of France is"
print("Prompt : ", prompt)


# 2 - Convert the text into token IDs
# encode() returns a tensor with shape [1, n] (1 row, n tokens).
encoded = model.encode(prompt)

first_row = encoded[0]
input_ids = first_row.tolist()

# 3 - Generate tokens one at a time
num_new_tokens = 10
generated_ids = []

for i in range(num_new_tokens):

    # # Ask the model for the scores of the NEXT token
    logits = model.get_logits_from_input_ids(input_ids)

    # Greedy approach : pick the token with the highest score
    next_id = argmax(logits)
    next_text = model.decode([next_id])

    print(f"Step {i + 1}: id={next_id:<7} text={repr(next_text)}")

    # Append the new token so the model sees it in the next step
    input_ids.append(next_id)
    generated_ids.append(next_id)

print()
print("Generated text:", repr(model.decode(generated_ids)))
print("Full text:     ", repr(prompt + model.decode(generated_ids)))
