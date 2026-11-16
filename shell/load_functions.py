import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from .models import FunctionDefinition


DEFAULT_FUNCTIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "input"
    / "functions_definition.json"
)


def load_function_definitions(
    path: str | Path = DEFAULT_FUNCTIONS_PATH,
) -> list[FunctionDefinition]:
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    return TypeAdapter(list[FunctionDefinition]).validate_python(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate function definitions from a JSON file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the definitions JSON file.",
    )
    args = parser.parse_args()

    definitions = load_function_definitions(args.path)
    print(f"Loaded {len(definitions)} function definitions:")
    for definition in definitions:
        parameters = ", ".join(
            f"{name}: {parameter.type}"
            for name, parameter in definition.parameters.items()
        )
        print(f"- {definition.name}({parameters}) -> {definition.returns.type}")


if __name__ == "__main__":
    main()
