"""Box abstract base + global registry.

A Box is a pure, single-shot transformation: `run(inputs, params) -> output`.
Each box declares its `inputs_schema`, `params_schema`, and `outputs_schema` as
Pydantic models so the M9 web UI can render forms and validate connections.
"""

from __future__ import annotations

from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
ParamsT = TypeVar("ParamsT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Box(Generic[InputT, ParamsT, OutputT]):
    """Base class for all boxes.

    Subclasses set the four ClassVars and implement `run`. The decorator
    `@register_box` records the subclass in `BOX_REGISTRY` keyed by `name`.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    inputs_schema: ClassVar[type[BaseModel]]
    params_schema: ClassVar[type[BaseModel]]
    outputs_schema: ClassVar[type[BaseModel]]

    def run(self, inputs: InputT, params: ParamsT) -> OutputT:  # pragma: no cover - abstract
        raise NotImplementedError


BOX_REGISTRY: dict[str, type[Box]] = {}


def register_box(cls: type[Box]) -> type[Box]:
    """Class decorator: register a Box subclass in `BOX_REGISTRY`."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"Box {cls.__name__} must define a non-empty `name` ClassVar")
    for attr in ("inputs_schema", "params_schema", "outputs_schema", "description"):
        if not hasattr(cls, attr):
            raise ValueError(f"Box {cls.__name__} missing ClassVar `{attr}`")
    if name in BOX_REGISTRY and BOX_REGISTRY[name] is not cls:
        raise ValueError(f"Duplicate box name '{name}' in registry")
    BOX_REGISTRY[name] = cls
    return cls


__all__ = ["Box", "BOX_REGISTRY", "register_box"]
