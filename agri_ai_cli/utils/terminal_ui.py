"""
Pure Python ANSI Terminal UI - No external dependencies.
Provides colored output, tables, progress bars, spinners, and panels.
"""

import sys
import time
import os

# ANSI color codes
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"


def supports_color():
    """Check if the terminal supports ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = supports_color()


def colorize(text, color_code):
    """Apply color to text if terminal supports it."""
    if USE_COLOR:
        return f"{color_code}{text}{Color.RESET}"
    return text


def bold(text):
    return colorize(text, Color.BOLD)


def success(text):
    return colorize(text, Color.GREEN)


def error(text):
    return colorize(text, Color.RED)


def warning(text):
    return colorize(text, Color.YELLOW)


def info(text):
    return colorize(text, Color.CYAN)


def dim(text):
    return colorize(text, Color.DIM)


def header(text):
    return colorize(text, Color.BOLD + Color.BRIGHT_CYAN)


# ── Box Drawing ────────────────────────────────────────────────────

BOX_CHARS = {
    "tl": "+", "tr": "+", "bl": "+", "br": "+",
    "h": "-", "v": "|", "lj": "+", "rj": "+",
    "tj": "+", "bj": "+", "cross": "+",
}

UNICODE_BOX = {
    "tl": "\u250c", "tr": "\u2510", "bl": "\u2514", "br": "\u2518",
    "h": "\u2500", "v": "\u2502", "lj": "\u251c", "rj": "\u2524",
    "tj": "\u252c", "bj": "\u2534", "cross": "\u253c",
}

DOUBLE_BOX = {
    "tl": "\u2554", "tr": "\u2557", "bl": "\u255a", "br": "\u255d",
    "h": "\u2550", "v": "\u2551", "lj": "\u2560", "rj": "\u2563",
    "tj": "\u2566", "bj": "\u2569", "cross": "\u256c",
}


def get_box(style="unicode"):
    if style == "double":
        return DOUBLE_BOX
    if style == "unicode":
        return UNICODE_BOX
    return BOX_CHARS


def strip_ansi(text):
    """Remove ANSI escape codes for length calculation."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', str(text))


def visible_len(text):
    """Get visible length of text (excluding ANSI codes)."""
    return len(strip_ansi(str(text)))


# ── Panel / Box ────────────────────────────────────────────────────

def print_panel(content, title="", width=70, style="unicode", color=Color.CYAN):
    """Print a bordered panel with optional title."""
    box = get_box(style)
    inner = width - 2

    # Top border
    if title:
        title_str = f" {title} "
        pad = inner - len(title_str)
        left_pad = pad // 2
        right_pad = pad - left_pad
        top = f"{box['tl']}{box['h'] * left_pad}{title_str}{box['h'] * right_pad}{box['tr']}"
    else:
        top = f"{box['tl']}{box['h'] * inner}{box['tr']}"

    print(colorize(top, color))

    # Content lines
    lines = content.split("\n") if isinstance(content, str) else content
    for line in lines:
        vlen = visible_len(line)
        padding = inner - vlen - 2
        if padding < 0:
            padding = 0
        print(f"{colorize(box['v'], color)} {line}{' ' * padding} {colorize(box['v'], color)}")

    # Bottom border
    bottom = f"{box['bl']}{box['h'] * inner}{box['br']}"
    print(colorize(bottom, color))


# ── Table ──────────────────────────────────────────────────────────

def print_table(headers, rows, col_widths=None, title=None, color=Color.CYAN):
    """Print a formatted table with borders."""
    if not col_widths:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = visible_len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, visible_len(row[i]))
            col_widths.append(min(max_w + 2, 40))

    box = get_box("unicode")
    total_w = sum(col_widths) + len(col_widths) + 1

    if title:
        print(colorize(f"\n  {title}", Color.BOLD + color))

    # Top border
    parts = [box["tl"]]
    for i, w in enumerate(col_widths):
        parts.append(box["h"] * w)
        parts.append(box["tj"] if i < len(col_widths) - 1 else box["tr"])
    print(colorize("".join(parts), color))

    # Header row
    row_parts = [box["v"]]
    for i, h in enumerate(headers):
        cell = f" {h}"
        cell += " " * (col_widths[i] - visible_len(cell))
        row_parts.append(colorize(cell, Color.BOLD))
        row_parts.append(colorize(box["v"], color))
    print("".join(row_parts))

    # Header separator
    parts = [box["lj"]]
    for i, w in enumerate(col_widths):
        parts.append(box["h"] * w)
        parts.append(box["cross"] if i < len(col_widths) - 1 else box["rj"])
    print(colorize("".join(parts), color))

    # Data rows
    for row in rows:
        row_parts = [colorize(box["v"], color)]
        for i, w in enumerate(col_widths):
            val = str(row[i]) if i < len(row) else ""
            cell = f" {val}"
            pad = w - visible_len(cell)
            if pad < 0:
                pad = 0
            cell += " " * pad
            row_parts.append(cell)
            row_parts.append(colorize(box["v"], color))
        print("".join(row_parts))

    # Bottom border
    parts = [box["bl"]]
    for i, w in enumerate(col_widths):
        parts.append(box["h"] * w)
        parts.append(box["bj"] if i < len(col_widths) - 1 else box["br"])
    print(colorize("".join(parts), color))


# ── Progress Bar ───────────────────────────────────────────────────

def progress_bar(current, total, width=40, prefix="", suffix=""):
    """Print an inline progress bar."""
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = f"{ratio * 100:.0f}%"
    line = f"\r  {prefix} {colorize(bar, Color.GREEN)} {pct} {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        print()


# ── Spinner ────────────────────────────────────────────────────────

SPINNER_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]


def spinner_task(label, duration=1.0, steps=10):
    """Show a spinner for a simulated task."""
    for i in range(steps):
        frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
        sys.stdout.write(f"\r  {colorize(frame, Color.CYAN)} {label}...")
        sys.stdout.flush()
        time.sleep(duration / steps)
    sys.stdout.write(f"\r  {success('*')} {label}... done\n")
    sys.stdout.flush()


# ── Confidence / Bar Charts ───────────────────────────────────────

def confidence_bar(value, max_val=1.0, width=20, label=""):
    """Print a horizontal bar showing confidence/score."""
    ratio = min(value / max(max_val, 0.001), 1.0)
    filled = int(width * ratio)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    if ratio >= 0.7:
        bar_colored = colorize(bar, Color.GREEN)
    elif ratio >= 0.4:
        bar_colored = colorize(bar, Color.YELLOW)
    else:
        bar_colored = colorize(bar, Color.RED)
    pct = f"{ratio * 100:.1f}%"
    if label:
        print(f"  {label:20s} {bar_colored} {pct}")
    else:
        print(f"  {bar_colored} {pct}")


# ── Section Headers ───────────────────────────────────────────────

def print_section(title, icon=">>"):
    """Print a section header."""
    line = f"\n  {colorize(icon, Color.CYAN)} {colorize(title, Color.BOLD + Color.WHITE)}"
    print(line)
    print(f"  {colorize('-' * (len(title) + 4), Color.DIM)}")


def print_banner(title, subtitle="", width=70):
    """Print a large banner."""
    print()
    print(colorize("=" * width, Color.CYAN))
    padding = (width - len(title)) // 2
    print(colorize(" " * padding + title, Color.BOLD + Color.BRIGHT_GREEN))
    if subtitle:
        padding2 = (width - len(subtitle)) // 2
        print(colorize(" " * padding2 + subtitle, Color.DIM))
    print(colorize("=" * width, Color.CYAN))
    print()


def print_kv(key, value, indent=4):
    """Print a key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{colorize(key + ':', Color.BOLD)} {value}")


def print_list(items, indent=4, bullet="*"):
    """Print a bulleted list."""
    spaces = " " * indent
    for item in items:
        print(f"{spaces}{colorize(bullet, Color.GREEN)} {item}")


def print_status(label, status, indent=4):
    """Print a status indicator."""
    spaces = " " * indent
    if status.upper() in ("OK", "PASS", "GOOD", "LOW", "HEALTHY"):
        colored = success(status)
    elif status.upper() in ("WARN", "MEDIUM", "MODERATE", "FAIR"):
        colored = warning(status)
    else:
        colored = error(status)
    print(f"{spaces}{label}: [{colored}]")


def divider(char="-", width=60):
    """Print a divider line."""
    print(colorize("  " + char * width, Color.DIM))
