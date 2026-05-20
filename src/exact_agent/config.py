"""Application configuration.

Configs live in `configs/*.yaml`; environment variables (prefixed `EXACT_`)
override individual leaves. The whole settings object is exposed through
`get_settings()` and cached for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Resolve the repo root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sub-models mirroring configs/app.yaml structure
# ---------------------------------------------------------------------------


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_s: int = 30
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    # Renamed away from `json` to avoid shadowing BaseModel.json (Pydantic warns).
    json_format: bool = Field(default=True, alias="json")
    include_request_id: bool = True

    model_config = {"populate_by_name": True}


class RouterConfig(BaseModel):
    premises_keys: list[str] = Field(
        default_factory=lambda: ["premises-NL", "premises_NL", "premises"]
    )


class LogicPipelineConfig(BaseModel):
    use_z3_fallback: bool = True
    z3_timeout_ms: int = 8000
    forward_chain_max_iters: int = 50
    top_k_premises: int = 8


class PhysicsPipelineConfig(BaseModel):
    numeric_tolerance_rel: float = 0.01
    use_solver_first: bool = True
    fallback_to_llm: bool = True
    # E6: number of independent LLM extractions to majority-vote per
    # symbol. 1 = legacy single-shot (default — backwards compatible);
    # 3 = self-consistency vote (Slide 28 official practical tip).
    llm_self_consistency_n: int = 1


class PipelinesConfig(BaseModel):
    logic: LogicPipelineConfig = Field(default_factory=LogicPipelineConfig)
    physics: PhysicsPipelineConfig = Field(default_factory=PhysicsPipelineConfig)


class SelfCorrectionConfig(BaseModel):
    enabled: bool = False
    max_rounds: int = 2


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Aggregated application settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXACT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_project_root)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    pipelines: PipelinesConfig = Field(default_factory=PipelinesConfig)
    self_correction: SelfCorrectionConfig = Field(default_factory=SelfCorrectionConfig)

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build Settings by layering app.yaml under env-var overrides."""
    root = _project_root()
    yaml_payload = _load_yaml(root / "configs" / "app.yaml")
    # pydantic-settings merges env vars on top of constructor kwargs.
    return Settings(**yaml_payload)
