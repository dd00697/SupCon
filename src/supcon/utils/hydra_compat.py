from __future__ import annotations

import argparse

def patch_hydra_argparse() -> None:
    original = argparse._ActionsContainer._check_help
    if getattr(original, "_supcon_hydra_compat", False):
        return

    def _check_help_with_fallback(self, action):
        try:
            return original(self, action)
        except ValueError:
            original_help = action.help
            if original_help is not None and not isinstance(original_help, str):
                action.help = str(original_help)
                try:
                    return original(self, action)
                finally:
                    action.help = original_help
            raise

    _check_help_with_fallback._supcon_hydra_compat = True
    argparse._ActionsContainer._check_help = _check_help_with_fallback
