import json
import numpy as np
from llm_sdk import Small_LLM_Model

MAX_NAME_TOKENS = 20


def build_char_to_byte() -> dict[str, int]:
    printable: list[int] = []
    printable += list(range(ord("!"), ord("~") + 1))
    printable += list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))

    char_to_byte: dict[str, int] = {}
    for byte_value in printable:
        char_to_byte[chr(byte_value)] = byte_value

    next_char = 256
    for byte_value in range(256):
        if byte_value not in printable:
            char_to_byte[chr(next_char)] = byte_value
            next_char += 1
    return char_to_byte


def build_id_to_text(vocab: dict[str, int]) -> dict[int, str]:
    char_to_byte = build_char_to_byte()
    id_to_text: dict[int, str] = {}

    for token, token_id in vocab.items():
        raw: list[int] = []
        for char in token:
            raw.append(char_to_byte[char])
        id_to_text[token_id] = bytes(raw).decode("utf-8", errors="replace")
    return id_to_text


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


def is_allowed(new_text: str, candidates: list[str]) -> bool:
    for candidate in candidates:
        if candidate.startswith(new_text):
            return True
    return False


def mask_logits(logits: list[float], text_so_far: str, candidates: list[str],
                id_to_text: dict[int, str]) -> np.ndarray:
    masked = np.full(len(logits), -np.inf)

    for token_id, token in id_to_text.items():
        if token == "":
            continue
        if is_allowed(text_so_far + token, candidates):
            masked[token_id] = logits[token_id]
    return masked


def generate_function_name(model: Small_LLM_Model, prompt: str,
                           function_names: list[str],
                           id_to_text: dict[int, str]) -> str:
    candidates: list[str] = []
    for name in function_names:
        candidates.append(name + '"')

    input_ids: list[int] = model.encode(prompt + '{"name": "')[0].tolist()
    text_so_far = ""

    for step in range(MAX_NAME_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)

        wanted_id = int(np.argmax(logits))
        masked = mask_logits(logits, text_so_far, candidates, id_to_text)
        next_id = int(np.argmax(masked))

        input_ids.append(next_id)
        text_so_far = text_so_far + id_to_text[next_id]

        print("")
        print("step", step)
        print("  model wanted:", repr(id_to_text.get(wanted_id, "<special>")))
        print("  mask gave:   ", repr(id_to_text[next_id]))
        print("  text so far: ", repr(text_so_far))

        if text_so_far in candidates:
            return text_so_far[:-1]

    raise RuntimeError("name was not finished in time")


model = Small_LLM_Model()

with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
    vocab = json.load(f)
id_to_text = build_id_to_text(vocab)

with open("data/input/functions_definition.json", encoding="utf-8") as f:
    functions = json.load(f)

function_names: list[str] = []
for function in functions:
    function_names.append(function["name"])

user_prompt = "What is the sum of 2 and 3?"
prompt = build_prompt(functions, user_prompt)

print("user:", user_prompt)
print("allowed functions:", function_names)

name = generate_function_name(model, prompt, function_names, id_to_text)
print("")
print("chosen function:", name)
