"""
Image Tag Schema — the contract the vision model's output must satisfy.

Spec §11.2. This is deliberately the *only* shape a vision model
response is ever trusted in. Per §11.3 / FR-1.3: every raw model
response is validated against this schema before being persisted;
on failure, retry up to MAX_RETRIES, and if still invalid, persist
image_metadata.status = "failed" with the raw response kept for
debugging — never invent a fallback tag to force a "success".
"""

from pydantic import BaseModel, Field


class ImageTags(BaseModel):
    """
    Structured output the vision model must produce for a single image.

    Field-by-field meaning (§11.2):
    - subject: the concrete thing depicted, e.g. "red fox"
    - category: a broader class it belongs to, e.g. "animal"
    - attributes: descriptive tags, e.g. ["orange fur", "wild", "forest"]
    - caption: a natural-language description — this is what gets
      embedded for semantic matching (§12.1), not the tags array,
      because captions carry richer meaning than a keyword list.
    - confidence: the model's own self-reported certainty, 0.0-1.0.
      This is NOT the same as similarity_score computed later during
      matching — confidence is about "did I classify this image
      correctly", similarity is about "does this image match this post".
    """

    subject: str = Field(..., min_length=1, description='e.g. "red fox"')
    category: str = Field(..., min_length=1, description='e.g. "animal"')
    attributes: list[str] = Field(
        ..., min_length=1, description='e.g. ["orange fur", "wild", "forest"]'
    )
    caption: str = Field(..., min_length=1, description="Natural-language description")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's own estimate, 0.0-1.0"
    )