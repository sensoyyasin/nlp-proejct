# Qwen Project — Plan

## Project summary

**Goal:** We build a program that turns an English sentence into a function call. For example, "What is the sum of 2 and 3?" becomes `fn_add_numbers` with `{"a": 2, "b": 3}`. The program doesn't answer the question. It only decides **which function** to call and **with which arguments**.

**Inputs and output:**

| File | Role |
|---|---|
| `functions_definition.json` | The list of available functions, with their parameters and types |
| `function_calling_tests.json` | The sentences to process (any number) |
| `function_calling_results.json` | Output: one function call per sentence |

**How it works:**

1. Load both files and the Qwen3-0.6B model once.
2. For each sentence, build a text that lists the available functions and the sentence.
3. Qwen generates the answer one token at a time. At each step, we block every token that would break the format (constrained decoding):
   - for the function name, Qwen can only spell names that exist
   - for each value, Qwen can only pick tokens of the right type (digits for numbers, no quotes inside strings)
4. We write the JSON structure ourselves; Qwen only fills in the name and the values.
5. Save all results to the output file.

**Why:** Small models often produce broken JSON. With constraints, the output is always valid. Qwen's understanding of the language decides the content, and our code guarantees the structure.

**The only tokenizer work:** Qwen's vocabulary stores tokens in a byte-level form (`Ġ` = space, `Ċ` = newline). We convert them back to real text once at startup. We don't write a BPE tokenizer.

**Team split (2 people, about 2–3 weeks):**

| Person | Responsibilities |
|---|---|
| **A — model side** | Vocabulary decoding, the generation loop, constraints for names and values, speed |
| **B — everything around it** | Loading and validating files, building the prompt, the command-line program, error handling, Makefile, testing, README |

**Milestones:**

| When | Milestone |
|---|---|
| Day 1 | Read the SDK together and agree on interfaces |
| End of week 1 | The pipeline runs end to end with correct function names |
| Week 2 | Parameter values for all types, edge cases, speed |
| Last days | Testing, lint (flake8/mypy), measure accuracy and time |

---

## How to start, in order

| # | Step | Owner |
|---|---|---|
| 1 | Read the `llm_sdk/` folder. Find out how to load the model, how to turn text into token IDs, how to get logits, and where the vocab file is. | Together |
| 2 | Make Qwen generate freely, with no constraints. Give it "The capital of France is" and generate 10 tokens by always taking the highest logit. If this works, you understand the core loop. | A |
| 3 | Open the vocab file and look at the tokens (`Ġ`, `Ċ`, and so on). Write the function that converts them to real text. | A |
| 4 | Load the two JSON files and print them. This is independent of the model, so it can be done at the same time. | B |
| 5 | Add the first constraint: function names only. This is the first real piece of the project. | A |
| 6 | Add parameter values, one type at a time: numbers first, then strings. | A |

Person B also handles the output file, error handling, and the Makefile.

---

## Example: one prompt through every step

We follow `"What is the sum of 2 and 3?"`. The token splits and scores below are made up for illustration, but the process is exactly what happens.

### Steps 1–3: done once at the start

**Step 1.** After reading `functions_definition.json`, the program knows:

| Name | Parameters |
|---|---|
| `fn_add_numbers` | a: number, b: number |
| `fn_greet` | name: string |
| `fn_reverse_string` | s: string |
| `fn_get_square_root` | a: number |
| `fn_substitute_string_with_regex` | source_string, regex, replacement: string |

**Step 2.** After reading `function_calling_tests.json`, the program has a list of sentences (11 in the example file, but it can be any number).

**Step 3.** Qwen is loaded, along with the vocabulary table (about 150,000 rows):

| Token ID | Stored as | Real text |
|---|---|---|
| 1 | `"` | `"` |
| 17 | `2` | `2` |
| 18 | `3` | `3` |
| 11 | `,` | `,` |
| 8822 | `fn` | `fn` |
| 62 | `_` | `_` |
| 718 | `add` | `add` |
| 23811 | `Ġhello` | ` hello` |
| … | … | … |

### Step 4: build the text for this prompt

```
You are a function calling assistant.

Available functions:
- fn_add_numbers(a: number, b: number): Add two numbers together...
- fn_greet(name: string): Generate a greeting message...
- fn_reverse_string(s: string): Reverse a string...
- fn_get_square_root(a: number): Calculate the square root...
- fn_substitute_string_with_regex(...): Replace all occurrences...

Request: What is the sum of 2 and 3?
Function call: {"name": "
```

This text is turned into token IDs, for example `[2610, 525, 264, ...]`, and given to Qwen. The text ends with `{"name": "`, so the most natural next thing for Qwen to write is a function name.

### Step 5a: choose the function name

Each time you call Qwen, it returns a score for every token. You keep only the allowed ones and take the best.

**Round 1** — generated so far: `""`

| Token | Qwen's score | Allowed? |
|---|---|---|
| `The` | 9.1 | ❌ no name starts with "The" |
| `fn` | 8.7 | ✅ every name starts with "fn" |
| `add` | 7.2 | ❌ no name starts with "add" |
| `{` | 6.0 | ❌ |

`The` had the highest score, but it isn't allowed, so the winner is **`fn`**.

**Round 2** — generated so far: `"fn"`

| Token | Score | Allowed? |
|---|---|---|
| `_add` | 9.4 | ✅ "fn_add" is the start of fn_add_numbers |
| `_g` | 5.1 | ✅ (fn_greet, fn_get_square_root) |
| `(` | 4.0 | ❌ |

The winner is **`_add`**. Generated so far: `"fn_add"`. Only one name still fits.

**Round 3** — the winner is **`_numbers`**. Generated so far: `"fn_add_numbers"`. This is a complete name, and no other name starts with it, so Phase A is done.

> Qwen chose the function because it understood "sum". Our code only stopped it from writing something invalid like `The answer is 5`.

### Step 5b: the parameter values

From step 1, we know that `fn_add_numbers` takes `a: number` and `b: number`.

We write the structure ourselves. Qwen doesn't need to generate it:

```
{"name": "fn_add_numbers", "parameters": {"a": 
```

Now Qwen is asked for the value of `a`, and only number tokens are allowed:

| Token | Score | Allowed? |
|---|---|---|
| `two` | 8.0 | ❌ not a digit |
| `2` | 7.9 | ✅ |
| `5` | 6.1 | ✅ |
| `"` | 5.0 | ❌ |

The winner is **`2`**. Next round:

| Token | Score | Allowed? |
|---|---|---|
| `,` (meaning "the number is finished") | 9.5 | ✅ |
| `0` | 3.2 | ✅ |

The winner is **`,`**, so the number is complete: **a = 2.0**.

We write `, "b": ` and the same thing happens. The result is **b = 3.0**.

> Qwen wanted to write `two` and later might have computed `5`. The constraints stopped it from writing `two`, and the prompt plus the context steered it to copy `2` from the sentence.

### Step 6: save the result

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Then go back to step 4 with the next sentence, `"What is the sum of 265 and 345?"`, and repeat the same process.

### Step 7: after all prompts

Write all the saved results into `data/output/function_calling_results.json` as one JSON list.

---

## Functions to write

Person A handles the model side and Person B handles everything around it. Function names are suggestions.

| File | Function | What it does | Owner | Difficulty |
|---|---|---|---|---|
| `models.py` | `FunctionDefinition` (pydantic class) | Shape of one tool: name, description, parameters | B | Easy |
| | `PromptItem` (pydantic class) | Shape of one input prompt | B | Easy |
| | `FunctionCallResult` (pydantic class) | Shape of one output: prompt, name, parameters | B | Easy |
| `io_utils.py` | `load_definitions(path)` | Read and validate `functions_definition.json` | B | Easy |
| | `load_prompts(path)` | Read and validate `function_calling_tests.json` | B | Easy |
| | `write_results(path, results)` | Write the output JSON file | B | Easy |
| `prompt_builder.py` | `build_prompt(user_prompt, definitions)` | Build the text in step 4 | B | Easy, then tuning |
| `vocabulary.py` | `decode_token(token_str)` | Turn `Ġhello` into `" hello"` | A | Medium |
| | `load_vocabulary(path)` | Build the token ID → real text table, once | A | Medium |
| `constraints.py` | `is_allowed_name(current, token_text, names)` | Does this token lead toward a valid name? | A | Medium |
| | `is_allowed_number(current, token_text)` | Is this still a valid number? | A | Hard |
| | `is_allowed_string(token_text)` | No quote or newline inside? | A | Medium |
| | `is_allowed_boolean(current, token_text)` | Does it lead toward `true` or `false`? | A | Easy |
| `decoder.py` | `pick_best_allowed(ids, rule)` | Call Qwen once, return the best allowed token | A | Medium |
| | `generate_function_name(ids, names)` | Step 5a | A | Medium |
| | `generate_value(ids, type)` | Step 5b, for one parameter | A | Hard |
| | `generate_arguments(ids, definition)` | Write the skeleton, then call `generate_value` for each parameter | A | Medium |
| | `process_prompt(prompt, definitions)` | Steps 4 to 6 for one prompt | A + B | Easy |
| `__main__.py` | `main()` | Command-line arguments, loading, loop, error per prompt, writing | B | Easy |
| `Makefile` | install, run, debug, clean, lint | Required rules | B | Easy |
| (testing) | Validation script | Reload the output and check it against the definitions | B | Easy |

Person A has fewer functions but the harder ones. Person B has more functions, and they are mostly simple, so B also takes on testing, prompt tuning, and the README.

---

## Workload by week

| When | Person A (model side) | Person B (everything around it) |
|---|---|---|
| **Day 1** | **Together:** read `llm_sdk/`, agree on function names and what each one takes and returns | (same) |
| **Days 2–3** | Free generation with no constraints, `decode_token`, `load_vocabulary` | pydantic models, `load_definitions`, `load_prompts`, `write_results`, error messages |
| **Days 4–5** | `pick_best_allowed`, `generate_function_name` | `build_prompt`, `main()`, Makefile, run a pipeline with a fake decoder |
| **End of week 1** | **Merge:** the real function names appear in the output file | (same) |
| **Days 6–8** | Numbers, then strings, then booleans in `generate_value` | Write your own test files with quotes, negative numbers, and many parameters; validation script |
| **Days 9–10** | Speed: precompute token flags, fewer model calls | Prompt tuning for the hard regex prompts, flake8 and mypy cleanup |
| **Last days** | Fix bugs found by B's tests | Measure accuracy and time, fill in the README table |

---

## Rules for working together

| Rule | Why |
|---|---|
| Agree on function inputs and outputs on day 1 | Then you can each work alone without breaking each other's code |
| Separate Git branches, merge every 2–3 days | Small merges cause fewer conflicts |
| B tests A's code, and A reviews B's code | Each person sees the other's half of the project |
| Both run `make lint` before merging | Keeps flake8 and mypy clean along the way, not only at the end |
