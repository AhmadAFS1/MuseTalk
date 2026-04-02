import torch


def apply_mmlab_compat_patches() -> None:
    """
    Torch 2.5 adds ``torch.optim.Adafactor``. Older MMEngine releases also try
    to register Transformers' ``Adafactor`` under the same registry key during
    import, which raises a duplicate-registration KeyError before MMPose can
    initialize. Avatar-prep paths do not rely on Adafactor, so hide the Torch
    copy before importing the MMLab stack.
    """
    if hasattr(torch.optim, "Adafactor"):
        try:
            delattr(torch.optim, "Adafactor")
        except Exception:
            pass
