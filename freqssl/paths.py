"""Machine-independent path placeholders for YAML configurations.

String values may contain ``${NAME}`` or ``${NAME:-default}``. ``NAME`` is read
from the environment; ``REPO_ROOT`` is always the checkout that holds this
package unless the environment sets it explicitly. A placeholder without a
default whose variable is unset is an error, so a missing path never silently
becomes an empty string. Text without ``${`` is returned unchanged, which keeps
old configurations and checkpoints byte-for-byte compatible.
"""

import os
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _lookup(name, default):
    value = os.environ.get(name)
    if value:
        return value
    if name == "REPO_ROOT":
        return str(REPO_ROOT)
    if default is not None:
        return expand(default)
    raise KeyError(f"Configuration needs environment variable {name}")


def expand(value):
    """Expand placeholders in strings, recursing through lists and mappings."""
    if isinstance(value, str):
        return _PATTERN.sub(lambda m: _lookup(m.group(1), m.group(2)), value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value
