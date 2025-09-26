"""Structured configuration handling for Coord2Region pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence
import numbers

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    confloat,
    conint,
    field_validator,
    model_validator,
)

CoordinateTriple = List[float | int]


class Coord2RegionConfig(BaseModel):
    """Pydantic model capturing all CLI-facing configuration options."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    input_type: Literal["coords", "region_names"] = "coords"
    inputs: Optional[List[Any]] = None
    coordinates: Optional[List[CoordinateTriple]] = Field(
        default=None,
        description=(
            "Input coordinates provided inline as triples "
            "(List[Tuple[float, float, float]]) or compatible sequences."
        ),
    )
    coordinates_file: Optional[str] = Field(
        default=None,
        alias="coords_file",
        description=(
            "Local filesystem path to tabular coordinates (e.g. CSV/TSV/XLSX). "
            "Remote URLs are not supported."
        ),
    )
    region_names: Optional[List[str]] = None
    # direct study inputs are not supported
    legacy_config: Optional[Dict[str, Any]] = Field(default=None, alias="config")

    outputs: List[
        Literal[
            "region_labels", "summaries", "images", "raw_studies", "mni_coordinates"
        ]
    ] = Field(default_factory=list)
    output_format: Optional[Literal["json", "pickle", "csv", "pdf", "directory"]] = None
    output_name: Optional[str] = None
    image_backend: Literal["ai", "nilearn", "both"] = "ai"
    batch_size: conint(ge=0) = 0

    working_directory: Optional[str] = None
    email_for_abstracts: Optional[str] = None
    use_cached_dataset: bool = True
    # unified sources control for dataset preparation and study search
    sources: Optional[List[str]] = None
    study_search_radius: confloat(ge=0) = 0.0
    region_search_radius: Optional[confloat(ge=0)] = None

    atlas_names: Optional[List[str]] = None
    atlas_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    max_atlases: Optional[conint(gt=0)] = None

    image_model: Optional[str] = None

    # Image generation prompt customization
    image_prompt_type: Optional[str] = Field(
        default=None,
        description=(
            "Template to use for AI image generation prompts. One of: "
            "'anatomical', 'functional', 'schematic', 'artistic', or 'custom'."
        ),
    )
    image_custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Custom image prompt template used when image_prompt_type is 'custom'. "
            "Supports {coordinate}, {first_paragraph}, "
            "and {atlas_context} placeholders."
        ),
    )

    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    providers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    summary_models: Optional[List[str]] = None
    prompt_type: Optional[str] = None
    custom_prompt: Optional[str] = None
    summary_max_tokens: Optional[conint(gt=0)] = None

    @field_validator("outputs", mode="before")
    @classmethod
    def _normalize_outputs(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("outputs must be provided as a list of strings")
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            normalized.append(str(item).lower())
        return list(dict.fromkeys(normalized))

    @field_validator("coordinates", mode="before")
    @classmethod
    def _normalize_coordinates(cls, value: Any) -> Optional[List[CoordinateTriple]]:
        if value is None:
            return None
        if isinstance(value, list):
            return [cls._coerce_coordinate(item) for item in value]
        raise TypeError("coordinates must be provided as a list")

    @field_validator("atlas_names", "sources", "region_names", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise TypeError("Field must be provided as a string or list of strings")
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return cleaned or None

    @field_validator("providers", mode="before")
    @classmethod
    def _normalize_providers(cls, value: Any) -> Dict[str, Dict[str, Any]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("providers must be a mapping of provider names to config")
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, cfg in value.items():
            if not isinstance(cfg, dict):
                raise TypeError("each provider configuration must be a dictionary")
            normalized[str(key)] = cfg
        return normalized

    @field_validator("summary_models", mode="before")
    @classmethod
    def _normalize_summary_models(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            raise TypeError("summary_models must be provided as a string or list")

        normalized: List[str] = []
        seen = set()
        for item in candidates:
            if item is None:
                continue
            name = str(item).strip()
            if name and name not in seen:
                normalized.append(name)
                seen.add(name)
        return normalized or None

    @field_validator("atlas_configs", mode="before")
    @classmethod
    def _normalize_atlas_configs(cls, value: Any) -> Dict[str, Dict[str, Any]]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("atlas_configs must be a mapping of atlas names to options")
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, cfg in value.items():
            if not isinstance(cfg, dict):
                raise TypeError("Each atlas configuration must be a dictionary")
            normalized[str(key)] = dict(cfg)
        return normalized

    def _legacy_list(self, key: str) -> Optional[List[str]]:
        if not self.legacy_config or key not in self.legacy_config:
            return None
        return type(self)._normalize_string_list(self.legacy_config[key])

    @model_validator(mode="after")
    def _validate_model(self) -> "Coord2RegionConfig":
        if not self.outputs:
            raise ValueError("At least one output must be specified")

        if "mni_coordinates" in self.outputs and self.input_type != "region_names":
            raise ValueError(
                "'mni_coordinates' output requires input_type='region_names'"
            )

        if self.output_format and not self.output_name:
            raise ValueError("output_name must be provided when output_format is set")

        legacy_atlases = self._legacy_list("atlas_names") or []
        active_atlases = self.atlas_names or legacy_atlases
        if (
            self.max_atlases
            and active_atlases
            and len(active_atlases) > self.max_atlases
        ):
            raise ValueError("Number of atlas names exceeds the configured max_atlases")

        if active_atlases:
            existing = dict(self.atlas_configs)
            for name in active_atlases:
                if name not in existing:
                    derived = self._derive_atlas_config(name)
                    if derived:
                        existing[name] = derived
            object.__setattr__(self, "atlas_configs", existing)

        self._validate_inputs_section()

        return self

    def _validate_inputs_section(self) -> None:
        if self.input_type == "coords":
            has_inline = bool(self.coordinates)
            has_legacy = bool(self.inputs)
            has_file = bool(self.coordinates_file)

            if sum(bool(x) for x in (has_inline, has_legacy, has_file)) == 0:
                raise ValueError(
                    "Coordinate inputs require 'coordinates', legacy 'inputs', or "
                    "'coordinates_file'"
                )

            if sum(bool(x) for x in (has_inline, has_legacy, has_file)) > 1:
                raise ValueError(
                    "Specify coordinates either inline, via 'inputs', or with "
                    "'coordinates_file', but not multiple sources"
                )

            if has_legacy:
                coords = [self._coerce_coordinate(item) for item in self.inputs or []]
                object.__setattr__(self, "coordinates", coords)
                object.__setattr__(self, "inputs", None)

            if self.region_names:
                raise ValueError(
                    "Field 'region_names' is not valid when " "input_type='coords'"
                )

        elif self.input_type == "region_names":
            names = self.region_names or (
                [str(item) for item in self.inputs] if self.inputs else []
            )
            if not names:
                raise ValueError(
                    "Region name inputs require 'region_names' or the legacy "
                    "'inputs' field"
                )
            object.__setattr__(self, "region_names", names)
            object.__setattr__(self, "inputs", None)

            if self.coordinates or self.coordinates_file:
                raise ValueError(
                    "Coordinate fields are not valid when " "input_type='region_names'"
                )
        else:
            raise ValueError("input_type must be 'coords' or 'region_names'")

    @staticmethod
    def _coerce_coordinate(item: Any) -> CoordinateTriple:
        if isinstance(item, str):
            parts = item.replace(",", " ").split()
        else:
            try:
                parts = list(item)
            except TypeError as exc:  # pragma: no cover - defensive
                raise ValueError("Coordinate entries must be iterable") from exc

        if len(parts) != 3:
            raise ValueError("Each coordinate must contain exactly three values")

        try:
            return [Coord2RegionConfig._cast_numeric(part) for part in parts]
        except (TypeError, ValueError) as exc:
            raise ValueError("Coordinate values must be numeric") from exc

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        if not value:
            return False
        if value.startswith(("~", "./", "../")):
            return True
        if os.path.isabs(value):
            return True
        if os.sep in value:
            return True
        if os.altsep and os.altsep in value:
            return True
        if len(value) > 2 and value[1] == ":" and value[0].isalpha():
            return True
        return False

    @staticmethod
    def _derive_atlas_config(name: str) -> Optional[Dict[str, Any]]:
        text = str(name).strip()
        if not text:
            return None
        lower = text.lower()
        if lower.startswith(("http://", "https://")):
            return {"atlas_url": text}
        if Coord2RegionConfig._looks_like_path(text):
            return {"atlas_file": text}
        return None

    @staticmethod
    def _cast_numeric(value: Any) -> float | int:
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            as_float = float(value)
            if as_float.is_integer():
                return int(as_float)
            return as_float
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        return as_float

    def _has_llm_credentials(self) -> bool:
        if self.providers or any(
            getattr(self, key)
            for key in (
                "gemini_api_key",
                "openrouter_api_key",
                "openai_api_key",
                "anthropic_api_key",
                "huggingface_api_key",
            )
        ):
            return True
        if self.legacy_config:
            for key in (
                "providers",
                "gemini_api_key",
                "openrouter_api_key",
                "openai_api_key",
                "anthropic_api_key",
                "huggingface_api_key",
            ):
                if key in self.legacy_config:
                    return True
        return False

    def collect_inputs(
        self, *, load_coords_file: Callable[[str], Sequence[Sequence[float]]]
    ) -> List[Any]:
        """Resolve configured inputs into data consumable by the pipeline."""
        if self.input_type == "coords":
            if self.coordinates_file:
                path = Path(self.coordinates_file).expanduser()
                coords = load_coords_file(str(path))
                return [
                    [self._cast_numeric(val) for val in triple] for triple in coords
                ]

            coords = self.coordinates or []
            return [[self._cast_numeric(val) for val in triple] for triple in coords]

        if self.input_type == "region_names":
            return [str(item) for item in self.region_names or []]
        # no other input types supported
        raise ValueError("input_type must be 'coords' or 'region_names'")

    def build_pipeline_config(self) -> Dict[str, Any]:
        """Construct the keyword arguments passed to ``run_pipeline``'s config."""
        config: Dict[str, Any] = dict(self.legacy_config or {})
        fields_set = self.model_fields_set

        def override(
            field: str, *, key: Optional[str] = None, transform=lambda x: x
        ) -> None:
            if field in fields_set:
                config[key or field] = transform(getattr(self, field))

        override("use_cached_dataset")
        override("study_search_radius", transform=lambda v: float(v))
        override(
            "region_search_radius",
            transform=lambda v: float(v) if v is not None else v,
        )
        override("working_directory")
        override("email_for_abstracts")
        override("sources")
        override("atlas_names")
        override("image_model")
        override("image_prompt_type")
        override("image_custom_prompt")
        override("providers")

        for key in (
            "gemini_api_key",
            "openrouter_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "huggingface_api_key",
        ):
            if key in fields_set and getattr(self, key):
                config[key] = getattr(self, key)

        override("summary_models", transform=lambda v: list(v))
        override("prompt_type")
        override("custom_prompt")
        if "summary_max_tokens" in fields_set:
            config["summary_max_tokens"] = (
                int(self.summary_max_tokens)
                if self.summary_max_tokens is not None
                else None
            )

        if self.atlas_configs:
            config["atlas_configs"] = dict(self.atlas_configs)

        return config

    def to_pipeline_runtime(self, inputs: Sequence[Any]) -> Dict[str, Any]:
        """Return arguments expected by :func:`coord2region.pipeline.run_pipeline`."""
        runtime: Dict[str, Any] = {
            "inputs": inputs,
            "input_type": self.input_type,
            "outputs": self.outputs,
            "output_format": self.output_format,
            "output_name": self.output_name,
            "image_backend": self.image_backend,
        }

        pipeline_config = self.build_pipeline_config()
        if pipeline_config:
            runtime["config"] = pipeline_config

        return runtime


__all__ = ["Coord2RegionConfig", "ValidationError"]
