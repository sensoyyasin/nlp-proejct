# Function Calling with Constrained Decoding

execute : python experiments/04_chat_format.py

Turn a natural-language request into a structured function call using a small language model (**Qwen3-0.6B**).

```
"What is the sum of 2 and 3?"
        ↓
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The program does **not** answer the question (`5`). It decides **which function** to call and **with which arguments**.

Small models are unreliable at writing JSON: they wrap it in ```` ```json ```` fences, invent function names, put numbers in quotes or forget closing braces. **Constrained decoding** fixes this: at every generation step, tokens that would break the JSON or the schema get a logit of `-inf`, so they can never be chosen. The model still picks the function and the values; it just can't produce invalid output.

```
Prompt → Tokenization → Input IDs → LLM → Logits → [mask invalid tokens] → Next token
```

---

## Progress

### Days 1–2 · Together
- [X] Read `llm_sdk/`
- [X] Set up the project: `uv add numpy pydantic` and `llm_sdk`
- [X] Generate 10 tokens from *"The capital of France is"* without constraints
- [X] Print a few entries of `vocab.json` to see how tokens look
- [X] Wrap prompts in the Qwen3 chat format, with thinking turned off

### Days 3–7 · Person A — Decoder
- [X] Day 3 · Table: token ID → real text
- [ ] Day 4 · Force the function name to be one from the list
- [ ] Day 5 · Generate string values
- [ ] Day 6 · Generate number and boolean values; write `{`, keys and `}` ourselves
- [ ] Day 7 · `generate(prompt)` returns `{name, parameters}`

### Days 3–7 · Person B — Shell
- [ ] Day 3 · pydantic models (function definition, prompt, result)
- [ ] Day 4 · Read/write JSON files without crashing on bad input
- [ ] Day 5 · Command-line arguments and Makefile
- [ ] Days 6–7 · ~20 hard test prompts and a script that scores them

**🏁 Milestone (day 7):** `make run` works on the example files.

### Days 8–11 · Make it good
- [ ] Full run finishes in under 5 minutes
- [ ] Fix failing test prompts by improving the prompt
- [ ] Numbers stop correctly
- [ ] Quotes inside strings are escaped
- [ ] `make lint` passes

### Days 12–14 · Wrap up
- [ ] Fill the performance table with real numbers
- [ ] Each person explains and changes the other's code once
- [ ] Fresh clone: `uv sync` and `make run` work

---

## `llm_sdk` cheat sheet

| Method | Returns | Note |
|--------|---------|------|
| `Small_LLM_Model()` | model | Loads `Qwen/Qwen3-0.6B` (mps → cuda → cpu) |
| `encode(text)` | 2-D `torch.Tensor` | Use `encode(text)[0].tolist()` to get `list[int]` |
| `get_logits_from_input_ids(ids)` | `list[float]` | One score per vocab token (~151k). No KV cache: reprocesses the whole sequence every call |
| `decode(ids)` | `str` | Skips special tokens |
| `get_path_to_vocab_file()` | `str` | Path to `vocab.json` |

The SDK has no chat-template helper: the Qwen chat format (`<|im_start|>user … <|im_end|>`) must be built by hand.

---

## Instructions

### Requirements
- Python **3.10+**
- [uv](https://docs.astral.sh/uv/)
- The provided `llm_sdk/` package at the repository root (next to `src/`)

### Installation
```bash
make install      # uv sync
```

### Running
```bash
make run
```
or:
```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```
All arguments are optional. Defaults read from `data/input/` and write to `data/output/function_calling_results.json`.

### Makefile rules

| Rule | Description |
|------|-------------|
| `make install` | Install dependencies with uv |
| `make run` | Run with default paths |
| `make debug` | Run under `pdb` |
| `make clean` | Remove `__pycache__`, `.mypy_cache` and other caches |
| `make lint` | `flake8 .` and mypy with the required flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |

### Project structure
```
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── llm_sdk/                  # provided SDK
├── data/input/
│   ├── functions_definition.json
│   └── function_calling_tests.json
└── src/
    ├── __main__.py           # CLI entry point
    ├── models.py             # pydantic models
    ├── io_utils.py           # safe JSON loading / writing
    ├── vocabulary.py         # vocab loading and token decoding
    ├── constraints.py        # schema-driven allowed-token logic
    ├── decoder.py            # constrained generation loop
    └── prompt_builder.py     # prompt construction
```

---

## Algorithm

### 1. Generation loop
1. Get logits for the current sequence with `get_logits_from_input_ids`.
2. Compute which tokens keep the output valid in the current state.
3. Set every other logit to `-inf`.
4. Pick the highest remaining logit (greedy).
5. Append the token, update the state, repeat.

Invalid tokens can never be selected, so the output is valid **by construction**.

### 2. Token → text table
Qwen uses byte-level BPE, so vocab strings contain special characters (`Ġ` = leading space, `Ċ` = newline). Every token is decoded once at startup into a `token_id → text` table, which is what lets each step check the whole vocabulary quickly.

### 3. Two-phase decoding

**Phase A — function name.** Only tokens that keep the generated text a prefix of at least one valid name are allowed:
```
allowed(token) ⇔ ∃ name ∈ names : (current + token_text) is a prefix of name
```
The phase ends when the text matches a full name and no longer name shares the prefix.

**Phase B — arguments.** The JSON skeleton (`{`, keys, `:`, `,`, `}`) is determined by the schema, so it is inserted directly. The model only generates **values**:

| Type | Allowed tokens | End condition |
|------|----------------|---------------|
| `number` | Digits, at most one `.`, optional leading `-`, must stay a valid numeric prefix | A terminator (`,` / `}`) wins over digits |
| `integer` | Same as number, without `.` | Same |
| `string` | Any token without an unescaped `"` or control character | Closing `"` chosen |
| `boolean` | Tokens extending toward `true` / `false` | Full literal reached |

Each value is validated and converted to the correct Python type.

### 4. Result
Name and arguments are wrapped in a pydantic model, validated against the definition (required keys, types, no extras) and written with `json.dump`.

---

## Design Decisions
- **Skeleton injected, not generated** — fewer forward passes, less risk.
- **Greedy decoding** — deterministic, reproducible, picks the most likely answer.
- **Precomputed token table** — no string processing inside the hot loop.
- **Prefix filtering for names** — one generation pass instead of scoring every function.
- **Function descriptions in the prompt** — the prompt helps *accuracy*; constrained decoding guarantees *validity*.
- **pydantic everywhere** — malformed data is rejected early with clear errors.
- **Fail per prompt, not per run** — one failing prompt doesn't stop the others.

---

## Performance Analysis

| Metric | Result |
|--------|--------|
| Valid JSON output | 100% (by construction) |
| Schema compliance | 100% (by construction) |
| Correct function selection | `<X>%` on `<N>` prompts |
| Correct argument extraction | `<Y>%` on `<N>` prompts |
| Total processing time | `<T>` s for `<N>` prompts on `<hardware>` |

**Baseline without constrained decoding** (Qwen3-0.6B via Ollama, same 11 example prompts, temperature 0): **0/11** outputs could be parsed with `json.loads` (all wrapped in ```` ```json ```` fences, one with unbalanced braces and invented keys). With schema-constrained decoding: **11/11** valid. Remaining errors are semantic, e.g. `regex: "NUMBERS"` instead of `\d+`.

---

## Challenges Faced
- **Byte-level tokens** — `Ġ`, `Ċ` and partial UTF-8 sequences need a dedicated decoding step.
- **Tokens spanning JSON parts** — a token like `2,` ends a value and starts a separator; the whole token text must be checked.
- **Numbers have no closing delimiter** — terminators must compete with digit tokens so the model decides when to stop.
- **Quotes inside strings** — e.g. `Reverse the string 'hello "world"'` requires correct escaping.
- **Speed** — filtering ~150k tokens per step in pure Python is slow; precomputed numpy masks fix it.

---

## Testing Strategy
- **Unit tests** (pytest, not submitted): vocab decoding, allowed tokens per type, name prefix filtering, loading missing/empty/invalid files.
- **End-to-end tests**: quotes, unicode, empty strings, negative and large numbers, floats, multi-parameter functions, ambiguous prompts.
- **Output validation**: every result file is reloaded and checked against the definitions.
- **Static checks**: `make lint` on every change.

---

## Example

Input (`function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Output (`function_calling_results.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "shrek"}},
  {"prompt": "Reverse the string 'hello'", "name": "fn_reverse_string", "parameters": {"s": "hello"}}
]
```

Error handling:
```bash
$ uv run python -m src --input missing.json
Error: input file 'missing.json' was not found.
```

---

## Resources

- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Byte-Pair Encoding — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)
- [Constrained decoding explained — Aidan Cooper](https://www.aidancooper.co.uk/constrained-decoding/)
- [pydantic](https://docs.pydantic.dev/) · [uv](https://docs.astral.sh/uv/) · [mypy](https://mypy.readthedocs.io/)


---

## Decoder plan (Person A · Days 4–7)

### Day 4 · Force the function name
**File:** `experiments/06_function_name.py`

1. Build the prompt in chat format (like `04`). Put the function list in the system message, and end the prompt with `{"name": "` so the model starts writing the name right away.
2. Loop (like `02_generate.py`), but at each step:
   - If `generated + table[i]` is not the start of a valid name, or `table[i]` is `None`, set its logit to `-inf`.
   - Pick the argmax of the remaining tokens.
3. Stop when the generated text is a full name and no longer name shares that prefix.
4. Test: *"What is the sum of 2 and 3?"* → `fn_add_numbers`

> **Why:** the model can never invent a name that isn't in the list.

### Day 5 · Generate string values
**File:** `experiments/07_string_value.py`

1. End the prompt with `... "s": "`.
2. Allowed tokens: those without `"` or a newline in them.
3. Stop when the model picks a token containing a quote. Watch out for tokens like `` ` "` `` (space + quote): the quote can be **inside** a longer token.
4. Test: *"Reverse the string 'hello'"* → `hello`

### Day 6 · Numbers and booleans; we write the skeleton
**File:** `experiments/08_number_bool.py`

1. **Number:** only digits, one `.`, and a leading `-`, plus `,` or `}` as the "stop" option. Digits are tokenized one by one, so at each step the model decides: one more digit, or stop?
2. **Boolean:** only tokens that are a prefix of `true` or `false`.
3. **Skeleton:** the model does not generate `{`, `"a": `, `,` or `}`. We insert them into the text ourselves, and the model only fills in the values.

> **Why:** fewer steps, faster, and broken JSON is impossible.

### Day 7 · Put it all together
**File:** `src/decoder.py`

1. `generate(prompt)` → `{"name": ..., "parameters": {...}}`
2. Flow: name (Day 4) → look up that function's parameters → for each parameter, generate a string / number / boolean based on its type (Days 5–6).
3. Move the Day 3 table code into `src/vocabulary.py`.
4. Connect with Person B's pydantic models.

**🏁 Goal:** `make run` works on the example files.

### Side tasks
- [ ] Fix `{token_id:<7}` in `03_vocab.py`
- [ ] Create `.gitignore`; commit 01–05, `pyproject.toml` and `uv.lock`
- [ ] `uv python pin 3.13`
- [ ] README: use `uv run python experiments/...`, and don't clone into Desktop/Documents (iCloud)
- [ ] Decide what to do with `test_project_data.py`
