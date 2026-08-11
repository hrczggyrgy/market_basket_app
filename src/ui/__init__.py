"""UI package for Market Basket Intelligence."""

from .registry import ModeSpec, dispatch, get_mode, get_modes, register_mode, render_sidebar

__all__ = [
    "register_mode",
    "get_modes",
    "get_mode",
    "render_sidebar",
    "dispatch",
    "ModeSpec",
]
