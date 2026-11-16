# Build the prompt in Qwen's chat format.
# Watch the model stop on its own at <|im_end|>.
# Finding the pick right function that matches the prompt. <|im_end|>, but writes no proper JSON

# Build the prompt in Qwen's chat format.
# Watch step by step which token the model picks and where it stops.


import json
import numpy as np
from llm_sdk import Small_LLM_Model

IM_END_ID = 151645 # <|im_end|>
MAX_NEW_TOKENS = 80  # safety limit
TOP_K = 5


def describe_function(function: dict) -> str:
    parameter_parts: list[str] = []
    for parameter_name, parameter_info in function["parameters"].items():
        parameter_parts.append(f"{parameter_name} : {parameter_info['type']}")

    parameters_text = ", ".join(parameter_parts)
    return f"- {function['name']}({parameters_text}): {function['description']}"


def build_prompt(functions: list[dict], user_prompt: str) -> str:
    function_lines: list[str] = []
    for function in functions:
        function_lines.append(describe_function(function))
    functions_text = "\n".join(function_lines)

    system_text = (
        "You are a function calling assistant.\n"
        "Choose the one function that fits the user's request "
        "and extract its arguments from the request.\n"
        "Available functions:\n"
        f"{functions_text}"
    )

    prompt = (
        "<|im_start|>system\n"
        f"{system_text}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return prompt


def softmax(logits: list[float]) -> np.ndarray:
    scores = np.array(logits, dtype=np.float64)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def token_text(model: Small_LLM_Model, token_id: int) -> str:
    if token_id == IM_END_ID:
        return "<|im_end|>"
    return repr(model.decode([token_id]))


def matching_functions(text_so_far: str, function_names: list[str]) -> list[str]:
    matches: list[str] = []
    for name in function_names:
        if name.startswith(text_so_far):
            matches.append(name)
        elif text_so_far.startswith(name):
            matches.append(name)
    return matches


# Greedy generation, no masking yet. Prints the top candidates at every step.
def generate_until_end(model: Small_LLM_Model, prompt: str,
                       function_names: list[str]) -> str:
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    new_ids: list[int] = []

	# Most important for lLoop.
    for step in range(MAX_NEW_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)#get the logits,scores.
        probs = softmax(logits)

        # sort from highest to lowest probability, keep the first TOP_K
        sorted_ids = np.argsort(probs)[::-1]
        top_ids = sorted_ids[:TOP_K]

        print("")
        print("step", step)

        for rank in range(TOP_K):
            token_id = int(top_ids[rank])
            percent = round(probs[token_id] * 100, 2)
            print("  ", rank + 1, token_text(model, token_id), percent, "%")

        end_percent = round(probs[IM_END_ID] * 100, 4)
        print("  im_end chance:", end_percent, "%")

        next_id = int(top_ids[0])
        if next_id == IM_END_ID:
            print("STOP at step", step, "-> model chose <|im_end|>")
            return model.decode(new_ids)

        input_ids.append(next_id)
        new_ids.append(next_id)

        text_so_far = model.decode(new_ids)
        print("  chosen:", token_text(model, next_id))
        print("  text so far:", repr(text_so_far))
        print("  still possible:", matching_functions(text_so_far, function_names))

    print("STOP -> reached MAX_NEW_TOKENS")
    return model.decode(new_ids)


model = Small_LLM_Model()

with open("data/input/functions_definition.json", encoding="utf-8") as f:
    functions = json.load(f)

function_names: list[str] = []
for function in functions:
    function_names.append(function["name"])

user_prompt = "What is the sum of 2 and 3?"
prompt = build_prompt(functions, user_prompt)

print("user:", user_prompt)
print("allowed functions:", function_names)

answer = generate_until_end(model, prompt, function_names)
print("")
print("final answer:", repr(answer))
