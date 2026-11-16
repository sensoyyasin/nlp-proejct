'''
day5  : Generating string values with constrained decoding

Goal :
	Turn "Greet shrek" into {"prompt" : "Greet shrek", "name" : "fn_greet", "parameters" : {"name" : "shrek"}}

Writing JSON skeleton ourselves. The model only fills in the function name and the string values.

Flow:
    prompt -> we write {"name": "  -> model writes the name (name mask)
           -> we write , "parameters": {"<param>": "
           -> model writes the value (string mask) until it closes the quote

Functions:
    build_char_to_byte    vocab.json chars (e.g. 'Ġ') -> real bytes (e.g. space)
    build_id_to_text      token id -> real text
    describe_function     one function definition -> one readable line
    build_prompt          full prompt in Qwen chat format, thinking off
    write_text            our fixed text -> tokens appended to input_ids
    mask_logits           every score -> -inf, except the allowed token ids
    is_allowed            is this text still the start of a candidate name?
    name_allowed_ids      tokens that keep the name valid (checked every step)
    string_allowed_ids    tokens that are safe inside a JSON string (built once)
    generate_function_name  masked loop until the name + '"' is complete
    generate_string       masked loop until the model closes the quote

Not done yet:
    number / boolean parameters (Day 6), escaped quotes inside strings.
'''


import json
import numpy as np
from llm_sdk import Small_LLM_Model

MAX_NAME_TOKENS = 20
MAX_STRING_TOKENS = 30


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


def write_text(model: Small_LLM_Model, input_ids: list[int], text: str) -> None:
    for token_id in model.encode(text)[0].tolist():
        input_ids.append(token_id)


def mask_logits(logits: list[float], allowed_ids: list[int]) -> np.ndarray:
    masked = np.full(len(logits), -np.inf)
    for token_id in allowed_ids:
        masked[token_id] = logits[token_id]
    return masked


def is_allowed(new_text: str, candidates: list[str]) -> bool:
    for candidate in candidates:
        if candidate.startswith(new_text):
            return True
    return False


def name_allowed_ids(text_so_far: str, candidates: list[str],
                     id_to_text: dict[int, str]) -> list[int]:
    allowed: list[int] = []
    for token_id, token in id_to_text.items():
        if token == "":
            continue
        if is_allowed(text_so_far + token, candidates):
            allowed.append(token_id)
    return allowed


def string_allowed_ids(id_to_text: dict[int, str]) -> list[int]:
    allowed: list[int] = []
    for token_id, token in id_to_text.items():
        if token == "":
            continue
        if not token.isprintable():
            continue
        if "\\" in token or "�" in token:
            continue

        quote_position = token.find('"')
        if quote_position == -1:
            allowed.append(token_id)
        elif quote_position == len(token) - 1:
            allowed.append(token_id)
    return allowed


def generate_function_name(model: Small_LLM_Model, input_ids: list[int],
                           function_names: list[str],
                           id_to_text: dict[int, str]) -> str:
    candidates: list[str] = []
    for name in function_names:
        candidates.append(name + '"')

    text_so_far = ""
    for step in range(MAX_NAME_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)
        allowed = name_allowed_ids(text_so_far, candidates, id_to_text)
        next_id = int(np.argmax(mask_logits(logits, allowed)))

        input_ids.append(next_id)
        text_so_far = text_so_far + id_to_text[next_id]
        print("  name so far:", repr(text_so_far))

        if text_so_far in candidates:
            return text_so_far[:-1]

    raise RuntimeError("name was not finished in time")


def generate_string(model: Small_LLM_Model, input_ids: list[int],
                    string_ids: list[int], id_to_text: dict[int, str]) -> str:
    value = ""
    for step in range(MAX_STRING_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)

        wanted_id = int(np.argmax(logits))
        next_id = int(np.argmax(mask_logits(logits, string_ids)))

        input_ids.append(next_id)
        token = id_to_text[next_id]

        print("")
        print("  step", step)
        print("    model wanted:", repr(id_to_text.get(wanted_id, "<special>")))
        print("    mask gave:   ", repr(token))

        if token.endswith('"'):
            value = value + token[:-1]
            print("    -> string closed")
            return value

        value = value + token
        print("    value so far:", repr(value))

    raise RuntimeError("string was not closed in time")


model = Small_LLM_Model()

with open(model.get_path_to_vocab_file(), encoding="utf-8") as f:
    vocab = json.load(f)
id_to_text = build_id_to_text(vocab)
string_ids = string_allowed_ids(id_to_text)
print("tokens allowed inside a string:", len(string_ids), "/", len(id_to_text))

with open("data/input/functions_definition.json", encoding="utf-8") as f:
    functions = json.load(f)

function_names: list[str] = []
for function in functions:
    function_names.append(function["name"])

user_prompt = "Greet shrek"
print("user:", user_prompt)

input_ids: list[int] = model.encode(build_prompt(functions, user_prompt))[0].tolist()

write_text(model, input_ids, '{"name": "')
name = generate_function_name(model, input_ids, function_names, id_to_text)
print("function:", name)

chosen_function: dict = {}
for function in functions:
    if function["name"] == name:
        chosen_function = function

write_text(model, input_ids, ', "parameters": {')

parameters: dict = {}
first = True
for parameter_name, parameter_info in chosen_function["parameters"].items():
    if parameter_info["type"] != "string":
        raise RuntimeError(f"{parameter_name} is {parameter_info['type']}, not done yet (Day 6)")

    if not first:
        write_text(model, input_ids, ", ")
    first = False

    print("")
    print("parameter:", parameter_name)
    write_text(model, input_ids, f'"{parameter_name}": "')
    parameters[parameter_name] = generate_string(model, input_ids, string_ids, id_to_text)

result = {"prompt": user_prompt, "name": name, "parameters": parameters}
print("")
print("result:", json.dumps(result))
