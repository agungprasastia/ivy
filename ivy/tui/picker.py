"""Inline arrow-key interactive selector powered by prompt_toolkit."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Literal

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from ivy.tui.theme import ThemeConfig, get_theme


@dataclass(frozen=True)
class PickerOption:
    """Single selectable row in an interactive picker."""

    value: str
    label: str
    meta: str = ""
    description: str = ""


@dataclass(frozen=True)
class PickerResult:
    """Outcome from an interactive picker interaction."""

    action: Literal["select", "new", "delete", "cancel"]
    value: str | None = None
    query: str = ""


@dataclass
class PickerState:
    """Pure state model for filtering, scrolling, and navigating picker items."""

    options: list[PickerOption]
    current_value: str | None = None
    allow_custom: bool = False
    max_visible: int = 8
    query: str = ""
    cursor: int = 0
    scroll_offset: int = 0
    _filtered: list[PickerOption] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.apply_filter("")
        if self.current_value:
            for idx, opt in enumerate(self._filtered):
                if opt.value == self.current_value:
                    self.cursor = idx
                    self._clamp_scroll()
                    break

    @property
    def items(self) -> list[PickerOption]:
        return self._filtered

    @property
    def selected(self) -> PickerOption | None:
        if 0 <= self.cursor < len(self._filtered):
            return self._filtered[self.cursor]
        return None

    def apply_filter(self, new_query: str) -> None:
        self.query = new_query
        q = new_query.strip().lower()
        if not q:
            self._filtered = list(self.options)
        else:
            self._filtered = [
                o
                for o in self.options
                if q in o.label.lower()
                or q in o.value.lower()
                or q in o.meta.lower()
                or q in o.description.lower()
            ]
        if self.allow_custom and q and not any(o.value.lower() == q for o in self._filtered):
            self._filtered.append(
                PickerOption(
                    value=new_query.strip(),
                    label=f"Use custom: {new_query.strip()}",
                    meta="custom",
                )
            )
        self.cursor = 0 if self._filtered else -1
        self.scroll_offset = 0

    def move(self, delta: int) -> None:
        if not self._filtered:
            self.cursor = -1
            return
        self.cursor = (self.cursor + delta) % len(self._filtered)
        self._clamp_scroll()

    def _clamp_scroll(self) -> None:
        if self.cursor < 0:
            self.scroll_offset = 0
            return
        if self.cursor < self.scroll_offset:
            self.scroll_offset = self.cursor
        elif self.cursor >= self.scroll_offset + self.max_visible:
            self.scroll_offset = self.cursor - self.max_visible + 1


def render_picker_fragments(
    title: str,
    state: PickerState,
    theme: ThemeConfig | None = None,
    width: int | None = None,
    allow_new: bool = False,
    allow_delete: bool = False,
) -> FormattedText:
    """Build FormattedText fragments for the rounded inline picker box."""
    t = theme or get_theme()
    cols = shutil.get_terminal_size((96, 24)).columns
    box_w = max(54, min(width or (cols - 4), 92))
    inner_w = box_w - 4

    def _pad_row(content: list[tuple[str, str]], used_len: int) -> list[tuple[str, str]]:
        rem = max(0, inner_w - used_len)
        return [
            ("class:picker.border", "│ "),
            *content,
            ("class:picker.border", (" " * rem) + " │\n"),
        ]

    header_txt = f" {t.glyph} {title} "
    top_fill = max(2, box_w - 3 - len(header_txt))
    frags: list[tuple[str, str]] = [
        ("class:picker.border", "╭─"),
        ("class:picker.title", header_txt),
        ("class:picker.border", ("─" * top_fill) + "╮\n"),
    ]

    # Search/filter row
    if state.query:
        q_label = f"Filter: {state.query}"
        frags.extend(_pad_row([("class:picker.filter", q_label[:inner_w])], min(inner_w, len(q_label))))
    else:
        hint_q = "Type to filter..."
        frags.extend(_pad_row([("class:picker.dim", hint_q)], len(hint_q)))

    frags.append(("class:picker.border", f"├{'─' * (box_w - 2)}┤\n"))

    visible = state.items[state.scroll_offset : state.scroll_offset + state.max_visible]
    if not visible:
        empty_txt = "No matching items"
        frags.extend(_pad_row([("class:picker.dim", empty_txt)], len(empty_txt)))
    else:
        for idx, opt in enumerate(visible, start=state.scroll_offset):
            is_sel = idx == state.cursor
            is_cur = state.current_value is not None and opt.value == state.current_value
            arrow = "❯ " if is_sel else "  "
            dot = "● " if is_cur else "  "
            meta_part = f"  {opt.meta}" if opt.meta else ""
            max_lbl = max(12, inner_w - 4 - len(meta_part))
            lbl = opt.label if len(opt.label) <= max_lbl else opt.label[: max_lbl - 1] + "…"

            row_style = "class:picker.selected" if is_sel else "class:picker.text"
            dot_style = "class:picker.active" if is_cur else row_style
            meta_style = "class:picker.selected.meta" if is_sel else "class:picker.dim"
            row_frags: list[tuple[str, str]] = [
                ("class:picker.accent" if is_sel else "class:picker.dim", arrow),
                (dot_style, dot),
                (row_style, lbl),
            ]
            if meta_part:
                row_frags.append((meta_style, meta_part))
            used = 4 + len(lbl) + len(meta_part)
            frags.extend(_pad_row(row_frags, used))

        if len(state.items) > state.max_visible:
            more = f"  ({state.scroll_offset + 1}-{state.scroll_offset + len(visible)} of {len(state.items)})"
            frags.extend(_pad_row([("class:picker.dim", more)], len(more)))

    frags.append(("class:picker.border", f"├{'─' * (box_w - 2)}┤\n"))

    keys = ["↑↓ move", "Enter select"]
    if allow_new:
        keys.append("Ctrl+N new")
    if allow_delete:
        keys.append("Del delete")
    keys.append("Esc cancel")
    footer_txt = " · ".join(keys)[:inner_w]
    frags.extend(_pad_row([("class:picker.dim", footer_txt)], len(footer_txt)))
    frags.append(("class:picker.border", f"╰{'─' * (box_w - 2)}╯"))
    return FormattedText(frags)


def _build_picker_style(theme: ThemeConfig) -> Style:
    return Style.from_dict(
        {
            "picker.border": theme.banner_border,
            "picker.title": f"bold {theme.banner_title}",
            "picker.accent": f"bold {theme.ui_accent}",
            "picker.active": f"bold {theme.ui_ok}",
            "picker.text": theme.status_bar_text,
            "picker.dim": theme.status_bar_dim,
            "picker.filter": f"bold {theme.ui_label}",
            "picker.selected": f"bold {theme.status_bar_strong} bg:{theme.completion_current_bg}",
            "picker.selected.meta": f"{theme.status_bar_text} bg:{theme.completion_current_bg}",
        }
    )


async def run_interactive_picker(
    title: str,
    options: list[PickerOption],
    current_value: str | None = None,
    theme: ThemeConfig | None = None,
    allow_new: bool = False,
    allow_delete: bool = False,
    allow_custom: bool = False,
) -> PickerResult:
    """Run an inline keyboard-driven selector and return the chosen PickerResult."""
    t = theme or get_theme()
    state = PickerState(
        options=options,
        current_value=current_value,
        allow_custom=allow_custom,
    )
    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        state.move(-1)

    @kb.add("down")
    def _down(event) -> None:
        state.move(1)

    @kb.add("pageup")
    def _pgup(event) -> None:
        state.move(-state.max_visible)

    @kb.add("pagedown")
    def _pgdn(event) -> None:
        state.move(state.max_visible)

    @kb.add("enter")
    def _enter(event) -> None:
        sel = state.selected
        if sel is not None:
            event.app.exit(result=PickerResult(action="select", value=sel.value, query=state.query))
        elif allow_custom and state.query.strip():
            event.app.exit(result=PickerResult(action="select", value=state.query.strip(), query=state.query))
        else:
            event.app.exit(result=PickerResult(action="cancel", query=state.query))

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=PickerResult(action="cancel", query=state.query))

    if allow_new:
        @kb.add("c-n")
        def _new(event) -> None:
            event.app.exit(result=PickerResult(action="new", query=state.query))

    if allow_delete:
        @kb.add("delete")
        @kb.add("c-d")
        def _delete(event) -> None:
            sel = state.selected
            if sel is not None:
                event.app.exit(result=PickerResult(action="delete", value=sel.value, query=state.query))

    @kb.add("backspace")
    def _backspace(event) -> None:
        if state.query:
            state.apply_filter(state.query[:-1])

    @kb.add("<any>")
    def _char(event) -> None:
        ch = event.data
        if ch and len(ch) == 1 and ch.isprintable():
            state.apply_filter(state.query + ch)

    control = FormattedTextControl(
        lambda: render_picker_fragments(
            title=title,
            state=state,
            theme=t,
            allow_new=allow_new,
            allow_delete=allow_delete,
        ),
        focusable=True,
    )
    app: Application[PickerResult] = Application(
        layout=Layout(HSplit([Window(content=control, dont_extend_height=True)])),
        key_bindings=kb,
        style=_build_picker_style(t),
        full_screen=False,
        erase_when_done=True,
    )
    return await app.run_async()
