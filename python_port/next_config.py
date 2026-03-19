from dataclasses import dataclass, field
from typing import Any


@dataclass
class NextConfig:
    options: dict[str, Any] = field(default_factory=dict)


next_config = NextConfig()
