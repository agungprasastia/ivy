"""Color themes and branding presets inspired by Hermes Agent."""

from __future__ import annotations

from dataclasses import dataclass

IVY_ASCII_LOGO = """[bold {title}]██╗██╗   ██╗██╗   ██╗[/]
[bold {accent}]██║██║   ██║╚██╗ ██╔╝[/]
[{accent}]██║██║   ██║ ╚████╔╝ [/]
[{border}]██║╚██╗ ██╔╝  ╚██╔╝  [/]
[{border}]██║ ╚████╔╝    ██║   [/]
[{dim}]╚═╝  ╚═══╝     ╚═╝   [/]"""

IVY_HERO_ART = """[{dim}]⠀⠀⠀⠀⠀⢀⣀⣤⣴⣶⣶⣶⣦⣤⣀⡀⠀⠀⠀⠀⠀[/]
[{border}]⠀⠀⠀⣠⣾⣿⣿⣿⣿⡿⠛⠛⢿⣿⣿⣿⣿⣷⣄⠀⠀⠀[/]
[{accent}]⠀⠀⣼⣿⣿⣿⣿⡿⠋⠀⣠⣄⠀⠙⢿⣿⣿⣿⣿⣧⠀⠀[/]
[{title}]⠀⢸⣿⣿⣿⣿⡿⠁⠀⣰⣿⣿⣆⠀⠈⢿⣿⣿⣿⣿⡇⠀[/]
[{accent}]⠀⠀⢻⣿⣿⣿⣿⣆⠀⠙⢿⡿⠋⠀⣰⣿⣿⣿⣿⡟⠀⠀[/]
[{border}]⠀⠀⠀⠙⢿⣿⣿⣿⣿⣶⣤⣤⣶⣿⣿⣿⣿⡿⠋⠀⠀⠀[/]
[{dim}]⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠿⠿⠟⠛⠉⠀⠀⠀⠀⠀[/]"""


@dataclass(frozen=True)
class ThemeConfig:
    """Visual palette and branding tokens for the Ivy TUI."""

    name: str
    description: str
    glyph: str
    prompt_symbol: str
    banner_border: str
    banner_title: str
    banner_accent: str
    banner_dim: str
    banner_text: str
    ui_accent: str
    ui_label: str
    ui_ok: str
    ui_error: str
    ui_warn: str
    status_bar_bg: str
    status_bar_text: str
    status_bar_strong: str
    status_bar_dim: str
    completion_bg: str
    completion_current_bg: str


THEMES: dict[str, ThemeConfig] = {
    "ivy": ThemeConfig(
        name="ivy",
        description="Emerald & jade — default Ivy theme",
        glyph="✦",
        prompt_symbol="❯",
        banner_border="#059669",
        banner_title="#34D399",
        banner_accent="#10B981",
        banner_dim="#6EE7B7",
        banner_text="#ECFDF5",
        ui_accent="#10B981",
        ui_label="#34D399",
        ui_ok="#4ADE80",
        ui_error="#F87171",
        ui_warn="#FBBF24",
        status_bar_bg="#0F172A",
        status_bar_text="#CBD5E1",
        status_bar_strong="#34D399",
        status_bar_dim="#64748B",
        completion_bg="#0F172A",
        completion_current_bg="#064E3B",
    ),
    "hermes": ThemeConfig(
        name="hermes",
        description="Classic Hermes — gold and bronze",
        glyph="☤",
        prompt_symbol="❯",
        banner_border="#CD7F32",
        banner_title="#FFD700",
        banner_accent="#FFBF00",
        banner_dim="#B8860B",
        banner_text="#FFF8DC",
        ui_accent="#FFBF00",
        ui_label="#DAA520",
        ui_ok="#4CAF50",
        ui_error="#EF5350",
        ui_warn="#FFA726",
        status_bar_bg="#1A1A2E",
        status_bar_text="#C0C0C0",
        status_bar_strong="#FFD700",
        status_bar_dim="#8A7A4A",
        completion_bg="#1A1A2E",
        completion_current_bg="#333355",
    ),
    "ocean": ThemeConfig(
        name="ocean",
        description="Cyan & indigo — cool slate",
        glyph="◈",
        prompt_symbol="❯",
        banner_border="#0284C7",
        banner_title="#38BDF8",
        banner_accent="#0EA5E9",
        banner_dim="#7DD3FC",
        banner_text="#F0F9FF",
        ui_accent="#38BDF8",
        ui_label="#818CF8",
        ui_ok="#34D399",
        ui_error="#FB7185",
        ui_warn="#FBBF24",
        status_bar_bg="#0B1120",
        status_bar_text="#CBD5E1",
        status_bar_strong="#38BDF8",
        status_bar_dim="#64748B",
        completion_bg="#0B1120",
        completion_current_bg="#1E3A5F",
    ),
}


def get_theme(name: str | None = None) -> ThemeConfig:
    """Resolve a built-in theme by name, defaulting to 'ivy'."""
    key = (name or "ivy").strip().lower()
    return THEMES.get(key, THEMES["ivy"])


def list_themes() -> list[ThemeConfig]:
    """Return all built-in themes."""
    return list(THEMES.values())
