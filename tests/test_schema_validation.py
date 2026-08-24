"""
§18 "Schema validation": ImageTags accepts valid payloads, rejects
malformed ones (missing fields, wrong types, out-of-range confidence).
Pure pytest, no network calls, no database.
"""

import pytest
from pydantic import ValidationError

from app.schemas.image_tags import ImageTags


def test_accepts_valid_payload():
    tags = ImageTags(
        subject="red fox",
        category="animal",
        attributes=["orange fur", "wild", "forest"],
        caption="A red fox standing in a snowy field",
        confidence=0.92,
    )
    assert tags.subject == "red fox"
    assert tags.confidence == 0.92
    assert len(tags.attributes) == 3


def test_accepts_valid_json_string():
    """The actual real-world path: VisionService parses raw model
    text via model_validate_json, not model construction directly."""
    raw = (
        '{"subject": "gray wolf", "category": "animal", '
        '"attributes": ["gray fur", "wild"], "caption": "A wolf in the snow", '
        '"confidence": 0.87}'
    )
    tags = ImageTags.model_validate_json(raw)
    assert tags.subject == "gray wolf"


@pytest.mark.parametrize(
    "missing_field",
    ["subject", "category", "attributes", "caption", "confidence"],
)
def test_rejects_missing_required_field(missing_field):
    payload = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["orange fur"],
        "caption": "A fox",
        "confidence": 0.9,
    }
    del payload[missing_field]
    with pytest.raises(ValidationError):
        ImageTags(**payload)


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2.0, -5.0])
def test_rejects_out_of_range_confidence(bad_confidence):
    with pytest.raises(ValidationError):
        ImageTags(
            subject="red fox",
            category="animal",
            attributes=["orange fur"],
            caption="A fox",
            confidence=bad_confidence,
        )


def test_rejects_wrong_type_for_confidence():
    with pytest.raises(ValidationError):
        ImageTags(
            subject="red fox",
            category="animal",
            attributes=["orange fur"],
            caption="A fox",
            confidence="very confident",  # str instead of float
        )


def test_rejects_wrong_type_for_attributes():
    with pytest.raises(ValidationError):
        ImageTags(
            subject="red fox",
            category="animal",
            attributes="orange fur, wild",  # str instead of list
            caption="A fox",
            confidence=0.9,
        )


def test_rejects_empty_attributes_list():
    """attributes has min_length=1 per the schema — an empty tag list
    isn't useful output and should fail validation, not silently pass."""
    with pytest.raises(ValidationError):
        ImageTags(
            subject="red fox",
            category="animal",
            attributes=[],
            caption="A fox",
            confidence=0.9,
        )


def test_rejects_empty_subject_string():
    with pytest.raises(ValidationError):
        ImageTags(
            subject="",
            category="animal",
            attributes=["orange fur"],
            caption="A fox",
            confidence=0.9,
        )


def test_rejects_malformed_json():
    with pytest.raises(ValidationError):
        ImageTags.model_validate_json("{not valid json")


def test_rejects_json_missing_fields():
    with pytest.raises(ValidationError):
        ImageTags.model_validate_json('{"subject": "fox"}')