from typing import Callable

import torch


def apply_to_modules(
    model: torch.nn.Module,
    fn: Callable[[torch.nn.Module, str], torch.nn.Module],
    target_selector_fn: Callable[[torch.nn.Module, str], bool],
    prefix: str = "",
) -> None:
    """Replace all modules whose type is in target_classes and not in the forbidden classes with
    result of given function.

    Args:
        model: Model
        fn: Function to be applied to targeted modules.
        target_selector_fn: Decide whether the module is supposed to be replaced.
        prefix: Prefix used to track nested name of module. Useful for logging.
    """

    for module_name, module in model.named_children():
        new_prefix = f"{prefix}/{module_name}"

        # nested module ==> go inside it
        apply_to_modules(module, fn, target_selector_fn, prefix=new_prefix)

        if target_selector_fn(module, new_prefix) or target_selector_fn(
            module, new_prefix.replace("/", ".")
        ):
            # Replace module with result of function
            setattr(model, module_name, fn(module, new_prefix))
