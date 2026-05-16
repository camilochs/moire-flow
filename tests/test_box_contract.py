"""Box ABC contract: registry semantics + missing-attr enforcement."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from moire_flow.boxes import BOX_REGISTRY, Box, register_box


def test_registry_contains_structure_loader():
    assert "structure_loader" in BOX_REGISTRY


def test_register_box_requires_name():
    class NoName(Box):  # type: ignore[type-arg]
        description = "x"
        inputs_schema = BaseModel
        params_schema = BaseModel
        outputs_schema = BaseModel

    with pytest.raises(ValueError, match="name"):
        register_box(NoName)


def test_register_box_requires_schemas():
    class Partial(Box):  # type: ignore[type-arg]
        name = "partial_box_test"
        description = "x"

    with pytest.raises(ValueError, match="inputs_schema"):
        register_box(Partial)


def test_registry_rejects_duplicate_names():
    class A(Box):  # type: ignore[type-arg]
        name = "dup_box_test"
        description = "a"
        inputs_schema = BaseModel
        params_schema = BaseModel
        outputs_schema = BaseModel

    class B(Box):  # type: ignore[type-arg]
        name = "dup_box_test"
        description = "b"
        inputs_schema = BaseModel
        params_schema = BaseModel
        outputs_schema = BaseModel

    register_box(A)
    try:
        with pytest.raises(ValueError, match="Duplicate"):
            register_box(B)
    finally:
        BOX_REGISTRY.pop("dup_box_test", None)
