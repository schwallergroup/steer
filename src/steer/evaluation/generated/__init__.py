"""
Generated benchmark evaluation system.

This module provides automated benchmark generation and evaluation for
synthetic chemistry route analysis.
"""

__version__ = "1.0.0"

# Don't auto-register on import to avoid side effects
# Users should explicitly call register_with_steer() if they want CLI integration

def register_with_steer():
    """
    Register generated evaluation classes with the steer evaluation system.

    This allows generated benchmarks to work with the existing CLI:

        >>> from steer.evaluation.generated import register_with_steer
        >>> register_with_steer()
        >>> # Now you can use: python -m steer.cli synth --bench_spec <generated_benchmark> bench

    Returns:
        dict: Dictionary of registered classes
    """
    from .register_eval_classes import update_tasks_module
    return update_tasks_module()


__all__ = [
    "register_with_steer",
]
