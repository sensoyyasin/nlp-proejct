from llm_sdk import Small_LLM_Model

'''
Qwen3 is a chat model. We need to thinking turned off.

The format looks like:
<|im_start|>system
{system message}<|im_end|>
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
<think>

</think>
'''

def argmax(values):
    # Return the index of the largest value in a list.
    best_index = 0
    for index in range(1, len(values)):
        if values[index] > values[best_index]:
            best_index = index
    return best_index

def build_chat_prompt(system_message, user_message):
    prompt = ""
    prompt += "<|im_start|>system\n" + system_message + "<|im_end|>\n"
    prompt += "<|im_start|>user\n" + user_message + "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    prompt += "<think>\n\n</think>\n\n"

    return prompt


model = Small_LLM_Model()

system_message = "You are a helpful assistant. Answer briefly."
user_message = "What is the capital of France?"

prompt = build_chat_prompt(system_message, user_message)
print("Full prompt:")
print(prompt)
print("-" * 40)

# Convert the prompt into token IDs
input_ids = model.encode(prompt)[0].tolist()

# The model ends its answer with  <|im_end|>; find its ID so we can stop there
end_id = model.encode("<|im_end|>")[0].tolist()[0]
print("ID of <|im_end|>:", end_id)

# Generate until the model produces <|im_end|> or we hit the limit
max_new_tokens = 30
generated_ids = []

for step in range(max_new_tokens):
    logits = model.get_logits_from_input_ids(input_ids)
    next_id = argmax(logits)

    # Stop when the model says it is done
    if next_id == end_id:
        print(f"(stopped at step {step + 1}: model produced <|im_end|>)")
        break

    input_ids.append(next_id)
    generated_ids.append(next_id)

#  Show the answer
print("Answer:", repr(model.decode(generated_ids)))
