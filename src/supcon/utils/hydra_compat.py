from __future__ import annotations

import argparse


def patch_hydra_argparse() -> None:
    # Hydra 1.3 can pass a non-string help object for shell-completion.
    # Python 3.14 argparse validates help strings more strictly and fails.
    #
    # On some Python builds (e.g. Colab), argparse internals differ and the
    # private _check_help hook may not exist; in that case we no-op.
    target_cls = None
    original = None
    for cls in (getattr(argparse, "_ActionsContainer", None), getattr(argparse, "ArgumentParser", None)):
        if cls is None:
            continue
        method = getattr(cls, "_check_help", None)
        if method is not None:
            target_cls = cls
            original = method
            break

    if original is None or target_cls is None:
        return

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
    setattr(target_cls, "_check_help", _check_help_with_fallback)
