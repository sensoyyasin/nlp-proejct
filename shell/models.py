from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParameterDefinition(StrictModel):
    type: Literal["number", "integer", "string", "boolean"]


class FunctionDefinition(StrictModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class Prompt(StrictModel):
    prompt: str


class Result(StrictModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
