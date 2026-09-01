from __future__ import annotations

from . import smoke
from .phase_only import (
    add_phase_only_arguments,
    load_phase_only_settings,
    load_phase_only_settings_from_args,
    phase_only_gradient_diagnostic,
    phase_only_runtime,
)


def main() -> None:
    originals = {
        "add_settings_arguments": smoke.add_settings_arguments,
        "load_settings": smoke.load_settings,
        "load_settings_from_args": smoke.load_settings_from_args,
        "gradient_diagnostic": smoke.gradient_diagnostic,
    }
    try:
        smoke.add_settings_arguments = add_phase_only_arguments
        smoke.load_settings = load_phase_only_settings
        smoke.load_settings_from_args = load_phase_only_settings_from_args
        smoke.gradient_diagnostic = phase_only_gradient_diagnostic
        with phase_only_runtime():
            smoke.main()
    finally:
        for name, value in originals.items():
            setattr(smoke, name, value)


if __name__ == "__main__":
    main()
