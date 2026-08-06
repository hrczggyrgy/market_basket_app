"""UI package for Market Basket Intelligence."""

from .registry import register_mode, get_modes, get_mode, render_sidebar, dispatch, ModeSpec

__all__ = [
    "register_mode",
    "get_modes",
    "get_mode",
    "render_sidebar",
    "dispatch",
    "ModeSpec",
]