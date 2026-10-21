# nlp-proejct - Function Calling with Constrained Decoding

## Main goal

The goal of this project is to build a function calling system that turns natural-language requests into structured function calls. Given a prompt like "What is the sum of 2 and 3?", the program does not answer the question. Instead, it uses a small language model (Qwen3-0.6B) to choose the right function and extract its arguments, for example `fn_add_numbers` with `{"a": 2, "b": 3}`. Because small models are unreliable at producing structured output, the project uses constrained decoding: at each generation step, tokens that would break the JSON structure or the function schema are blocked, which guarantees 100% valid, schema-compliant JSON output.

### Example

Imagine a smart assistant that has access to a small set of tools, defined once in `functions_definition.json`:

| Function | Parameters | Purpose |
|----------|------------|---------|
| `fn_get_weather` | `city: string`, `celsius: boolean` | Get the current weather for a city |
| `fn_add_numbers` | `a: number`, `b: number` | Add two numbers |
| `fn_greet` | `name: string` | Greet a person by name |

Users can phrase their requests in many different ways, and the model maps each one to the right tool:

| User prompt | Generated function call |
|-------------|-------------------------|
| "What's the weather like in Paris today, in Celsius?" | `{"name": "fn_get_weather", "parameters": {"city": "Paris", "celsius": true}}` |
| "Is it cold in Tokyo right now?" | `{"name": "fn_get_weather", "parameters": {"city": "Tokyo", "celsius": false}}` |
| "Can you tell me what 1500 plus 27 is?" | `{"name": "fn_add_numbers", "parameters": {"a": 1500.0, "b": 27.0}}` |
| "Say hi to Shrek" | `{"name": "fn_greet", "parameters": {"name": "Shrek"}}` |

Five different sentences, three tools, zero hand-written rules: the LLM understands the language, and constrained decoding guarantees the output structure.

### Why not just let the LLM answer?

An LLM alone cannot check today's weather, cannot be trusted to compute `48273 × 91856` correctly, and cannot actually send an email or write to a database. A function call bridges that gap: the model decides *what* to do, and a real program does it reliably.


## Description

It's a function calling tool that translates natural-language requests into structured, machine-executable function calls using a small language model (**Qwen/Qwen3-0.6B**, ~0.6B parameters).

Given a prompt such as:

```
"What is the sum of 40 and 2?"
```

the program does **not** answer `42`. Instead, it produces:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

Small models are unreliable at producing structured output when simply prompted. This project solves that with **constrained decoding**: at every generation step, the logits of tokens that would break the JSON structure or the function schema are set to negative infinity, so the model can only choose among valid continuations. The result is **100% parseable, schema-compliant JSON**, while the model itself still decides *which* function to call and *what* the argument values are.

### Key features

- Function selection performed by the LLM (no keyword heuristics).
- Token-level constrained decoding driven by the model vocabulary file.
- Schema enforcement for `number`, `integer`, `string` and `boolean` argument types.
- Validation of inputs and outputs with **pydantic**.
- Graceful handling of missing files, invalid JSON and malformed definitions.
- Fully typed code, `flake8` and `mypy` clean.

---

## Instructions

### Requirements

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/)
- The provided `llm_sdk/` package, placed at the repository root (next to `src/`)

### Installation

```bash
make install      # equivalent to: uv sync
```

### Running

```bash
make run
```

or directly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All arguments are optional. By default, inputs are read from `data/input/` and results are written to `data/output/function_calling_results.json`.

### Makefile rules

| Rule | Description |
|------|-------------|
| `make install` | Install dependencies with `uv` |
| `make run` | Run the program with default paths |
| `make debug` | Run the program under `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache` and other caches |
| `make lint` | Run `flake8 .` and `mypy` with the required flags |
| `make lint-strict` | Run `flake8 .` and `mypy . --strict` |

### Project structure

```
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── llm_sdk/                  # provided SDK
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
└── src/
    ├── __main__.py           # CLI entry point
    ├── models.py             # pydantic models (definitions, prompts, results)
    ├── io_utils.py           # safe loading / writing of JSON files
    ├── vocabulary.py         # vocabulary loading and token string decoding
    ├── constraints.py        # schema-driven allowed-token logic
    ├── decoder.py            # constrained generation loop
    └── prompt_builder.py     # prompt construction for the model
```

> Adapt the file names above to match your actual implementation.

---

## Algorithm Explanation

### 1. The generation loop

A language model generates text one token at a time:

```
prompt → token IDs → LLM → logits → next token → append → repeat
```

Constrained decoding inserts a filter between *logits* and *next token*:

1. Get the logits for the current sequence with `get_logits_from_input_ids`.
2. Compute the set of tokens that keep the output valid given the current state.
3. Set the logits of every other token to `-inf`.
4. Pick the highest remaining logit (greedy decoding).
5. Append the token and update the decoder state.

Because invalid tokens can never be selected, the output is valid **by construction**, not by luck.

### 2. Mapping tokens to strings

The vocabulary file (from `get_path_to_vocab_file()`) maps token IDs to their string form. Qwen uses a byte-level BPE tokenizer, so token strings contain special characters:

- `Ġ` represents a leading space
- `Ċ` represents a newline

At startup, every token is decoded once into its real text and stored in a lookup table (`token_id → text`). This table is what makes it possible to answer "would this token keep the JSON valid?" for the whole vocabulary at each step.

### 3. Two-phase decoding

The output object is generated in two phases, each with its own constraint.

#### Phase A — function name selection

The set of valid function names is known in advance (from `functions_definition.json`). The decoder keeps the text generated so far for the name and, at each step, only allows tokens whose text extends the current prefix toward **at least one** valid name:

```
allowed(token) ⇔ ∃ name ∈ names : (current + token_text) is a prefix of name
```

When the generated text exactly matches a full name and no longer name shares that prefix, the phase ends. The model therefore chooses the function, but can only "spell" names that exist.

#### Phase B — arguments, driven by the schema

Once the function is chosen, its parameter schema is known. The JSON skeleton (`{`, keys, `:`, `,`, `}`) is fully determined by the schema, so it is **inserted directly** without asking the model. The model is only consulted for the **values**, each with a type-specific constraint:

| Type | Allowed tokens | End condition |
|------|----------------|---------------|
| `number` | Tokens composed of digits, at most one `.`, an optional leading `-`, and the result must remain a valid numeric prefix | A token that cannot extend the number becomes most likely, or a terminator (`,` / `}`) is chosen |
| `integer` | Same as `number`, without `.` | Same as above |
| `string` | Any token that does not contain an unescaped `"` or a control character | The closing `"` token is chosen |
| `boolean` | Tokens that extend toward `true` or `false` | Full literal reached |

Each value is then validated (e.g. `float(...)` for numbers) and converted to the correct Python type before being stored.

### 4. Assembling the result

The chosen name and the decoded arguments are wrapped in a pydantic model, validated against the function definition (all required parameters present, correct types, no extra keys), and written to the output file with `json.dump`. Because both phases are constrained, parsing never fails.

---

## Design Decisions

- **Schema skeleton is injected, not generated.** Keys, braces and separators are deterministic, so letting the model generate them only wastes forward passes and adds risk. Injecting them reduces the number of LLM calls to roughly the number of *value* tokens.
- **Greedy decoding.** For function calling, the goal is the most likely correct answer, not diversity. Greedy selection is deterministic, which makes debugging and evaluation reproducible.
- **Precomputed token table.** Decoding all vocabulary strings once at startup avoids repeated string processing inside the generation loop, which runs once per generated token.
- **Prefix filtering for names instead of scoring each name.** Prefix filtering lets the model choose freely among valid names with a single generation pass, rather than computing a full likelihood for every candidate function.
- **Prompt with function descriptions.** The prompt lists the available functions with their descriptions and parameters so the model has the context to make the right choice. The prompt helps *accuracy*; constrained decoding guarantees *validity*. The project never relies on the prompt alone for structure.
- **pydantic everywhere.** Input files, function definitions and results are all pydantic models, so malformed data is rejected early with clear error messages instead of failing deep inside the decoder.
- **Fail per prompt, not per run.** If one prompt fails (e.g. an unexpected model error), the error is reported and processing continues with the next prompt.

---

## Performance Analysis

> Replace the placeholders below with your own measurements.

| Metric | Result |
|--------|--------|
| Valid JSON output | 100% (guaranteed by construction) |
| Schema compliance | 100% (guaranteed by construction) |
| Correct function selection | `<X>%` on `<N>` test prompts |
| Correct argument extraction | `<Y>%` on `<N>` test prompts |
| Total processing time | `<T>` seconds for `<N>` prompts on `<hardware>` |

**Observations**

- Validity is never the bottleneck: errors that remain are *semantic* (wrong function or wrong value), not structural.
- Most of the runtime is spent in model forward passes. Injecting the schema skeleton and stopping value generation as soon as a terminator is reached significantly reduces the number of calls.
- Accuracy depends mainly on prompt quality and on how clearly the function descriptions differ from each other.

---

## Challenges Faced

- **Byte-level token strings.** Tokens do not map one-to-one to readable text (`Ġ`, `Ċ`, partial UTF-8 sequences). A dedicated decoding step was required to build a reliable `token → text` table.
- **Tokens spanning several JSON parts.** A single token can contain, for example, a digit followed by a comma (`"2,"`). The constraint logic must check the *whole* token text against the current state, not only its first character, and handle the case where a token both ends a value and starts the next structure.
- **Knowing when a number ends.** Unlike strings, numbers have no closing delimiter. The decoder must allow a terminator to compete with digit tokens so the model itself decides when the number is complete.
- **Strings containing quotes or special characters.** Prompts such as `Reverse the string 'hello "world"'` require correct escaping of the generated value before it is written as JSON.
- **Speed.** Filtering ~150k tokens at every step is expensive in pure Python. Precomputing token properties (e.g. "is numeric", "contains a quote") once at startup made the per-step filtering fast enough to stay well under the 5-minute limit.

---

## Testing Strategy

- **Unit tests** (pytest, not submitted) for:
  - vocabulary decoding (`Ġ`, `Ċ`, special tokens),
  - allowed-token computation for each type (number, integer, string, boolean),
  - function name prefix filtering,
  - input loading with missing files, empty files and invalid JSON.
- **End-to-end tests** on the provided example files and on custom files with:
  - empty strings and strings with quotes, unicode and special characters,
  - large and negative numbers, floats,
  - functions with several parameters of different types,
  - ambiguous prompts and functions with similar descriptions.
- **Output validation**: every produced file is reloaded with `json.load` and checked against the function definitions (names, required keys, types, no extra keys).
- **Static checks**: `make lint` (flake8 + mypy) on every change.

---

## Example Usage

Default paths:

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Example input (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Example output (`function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

Error handling example:

```bash
$ uv run python -m src --input missing.json
Error: input file 'missing.json' was not found.
```

---

## Resources

### References

- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Qwen3 model card — Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte-Pair Encoding tokenization — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023) — the theory behind regex/grammar-constrained decoding
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [mypy documentation](https://mypy.readthedocs.io/)
