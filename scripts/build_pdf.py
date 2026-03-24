#!/usr/bin/env python3
"""
build_pdf.py — Convert MyST markdown source to PDF via pandoc + typst.

Reads book/myst.yml for TOC order, preprocesses MyST directives into
pandoc-compatible markdown, and runs pandoc with Typst to produce
exports/ai4stats.pdf.

This script is a READ-ONLY consumer of book/*.md. It never modifies source files.
"""

import os
import re
import sys
import shutil
import subprocess
import yaml
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_DIR = os.path.join(REPO_ROOT, "book")
EXPORTS_DIR = os.path.join(REPO_ROOT, "exports")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

# Cover image for PDF (high-res version) — RELATIVE to REPO_ROOT
COVER_IMAGE_REL = "figures/AI-for-Official-Stats-Cover-1200x1600-300dpi.png"
COVER_IMAGE_ABS = os.path.join(REPO_ROOT, COVER_IMAGE_REL)

# Admonition class -> emoji + label
ADMONITION_LABELS = {
    "tip": "💡 Tip",
    "note": "📝 Note",
    "warning": "⚠️ Warning",
    "dropdown": "📋 Details",
}

# Front/back matter files that get unnumbered chapter headers
UNNUMBERED_FILES = {
    "foreword.md",
    "version.md",
    "intro.md",
    "glossary.md",
    "bibliography.md",
    "book_index.md",
}


def read_toc(myst_yml_path):
    with open(myst_yml_path, "r") as f:
        config = yaml.safe_load(f)
    toc = config.get("project", {}).get("toc", [])
    return [entry["file"] for entry in toc if "file" in entry]


def strip_jupytext_frontmatter(text):
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end == -1:
        return text
    frontmatter = text[3:end]
    if "jupytext:" in frontmatter or "kernelspec:" in frontmatter:
        return text[end + 3:].lstrip("\n")
    return text


def strip_chapter_number_prefix(text):
    """Convert '# Chapter N - Title' to '# Title' for pandoc numbering."""
    return re.sub(
        r'^# Chapter \d+\s*[-–—]\s*(.*)$',
        r'# \1',
        text,
        flags=re.MULTILINE
    )


def convert_myst_image(match):
    full_block = match.group(0)
    path_match = re.search(r'\{image\}\s+(.+)', full_block)
    if not path_match:
        return full_block
    path = path_match.group(1).strip()

    alt = ""
    width = ""
    alt_match = re.search(r':alt:\s*(.+)', full_block)
    if alt_match:
        alt = alt_match.group(1).strip()
    width_match = re.search(r':width:\s*(.+)', full_block)
    if width_match:
        w = width_match.group(1).strip()
        if w.endswith("%"):
            width = f"{{ width={w} }}"
        elif w.endswith("px"):
            width = f"{{ width={w} }}"
        else:
            width = f"{{ width={w} }}"

    return f"![{alt}]({path}){width}"


def convert_myst_figure(match):
    full_block = match.group(0)
    path_match = re.search(r'\{figure\}\s+(.+)', full_block)
    if not path_match:
        return full_block
    path = path_match.group(1).strip()

    width = ""
    width_match = re.search(r':width:\s*(.+)', full_block)
    if width_match:
        w = width_match.group(1).strip()
        if w.endswith("%"):
            width = f"{{ width={w} }}"
        else:
            width = f"{{ width={w} }}"

    lines = full_block.split("\n")
    caption_lines = []
    past_options = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(":") and not past_options:
            continue
        if stripped == "" and not past_options:
            past_options = True
            continue
        if stripped == "```":
            break
        if past_options:
            caption_lines.append(line)

    caption = " ".join(caption_lines).strip() if caption_lines else ""
    return f"![{caption}]({path}){width}"


def convert_admonition(match):
    full_block = match.group(0)

    title_match = re.search(r'\{admonition\}\s+(.+)', full_block)
    title = title_match.group(1).strip() if title_match else "Note"

    class_match = re.search(r':class:\s*(\w+)', full_block)
    adm_class = class_match.group(1).strip() if class_match else "note"

    label = ADMONITION_LABELS.get(adm_class, "📝 Note")

    lines = full_block.split("\n")
    content_lines = []
    past_header = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(":") and not past_header:
            continue
        if stripped == "```":
            break
        past_header = True
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    quoted_content = "\n".join(
        f"> {line}" if line.strip() else ">" for line in content.split("\n")
    )

    return f"> **{label}: {title}**\n>\n{quoted_content}\n"


def convert_dropdown(match):
    full_block = match.group(0)

    title_match = re.search(r'\{dropdown\}\s+(.+)', full_block)
    title = title_match.group(1).strip() if title_match else "Details"

    lines = full_block.split("\n")
    content_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(":"):
            continue
        if stripped == "```":
            break
        content_lines.append(line)

    content = "\n".join(content_lines).strip()
    quoted_content = "\n".join(
        f"> {line}" if line.strip() else ">" for line in content.split("\n")
    )

    return f"> **📋 {title}**\n>\n{quoted_content}\n"


def convert_code_block(match):
    full_block = match.group(0)
    lang_match = re.search(r'\{code-block\}\s+(\w+)', full_block)
    lang = lang_match.group(1) if lang_match else ""

    lines = full_block.split("\n")
    content_lines = []
    past_header = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith(":") and not past_header:
            continue
        if stripped == "```":
            break
        past_header = True
        content_lines.append(line)

    content = "\n".join(content_lines)
    return f"```{lang}\n{content}\n```"


def convert_glossary(match):
    full_block = match.group(0)
    lines = full_block.split("\n")
    content_lines = []
    for line in lines[1:]:
        if line.strip() == "```":
            break
        content_lines.append(line)
    return "\n".join(content_lines)


def process_directives(text):
    text = re.sub(r'```\{contents\}.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```\{tableofcontents\}.*?```', '', text, flags=re.DOTALL)

    text = re.sub(
        r'```\{code-block\}\s+\w+\n.*?```',
        convert_code_block, text, flags=re.DOTALL
    )
    text = re.sub(
        r'```\{glossary\}\n.*?```',
        convert_glossary, text, flags=re.DOTALL
    )
    text = re.sub(
        r'```\{image\}\s+.+?\n.*?```',
        convert_myst_image, text, flags=re.DOTALL
    )
    text = re.sub(
        r'```\{figure\}\s+.+?\n.*?```',
        convert_myst_figure, text, flags=re.DOTALL
    )
    text = re.sub(
        r'```\{admonition\}\s+.+?\n.*?```',
        convert_admonition, text, flags=re.DOTALL
    )
    text = re.sub(
        r'```\{dropdown\}\s+.+?\n.*?```',
        convert_dropdown, text, flags=re.DOTALL
    )

    return text


def strip_cover_image_from_intro(text):
    """Remove the cover image and 'Brock Webb' byline from intro.md.
    
    The intro has the cover image embedded for the HTML site. In the PDF,
    the cover is a separate full-bleed page, so we strip the duplicate.
    """
    # Remove the markdown image that references the cover
    text = re.sub(
        r'!\[.*?\]\(images/cover-web\.png\)\{[^}]*\}\s*',
        '',
        text
    )
    # Also remove a plain image reference without attributes
    text = re.sub(
        r'!\[.*?\]\(images/cover-web\.png\)\s*',
        '',
        text
    )
    # Remove standalone "**Brock Webb**" byline (it's on the cover page)
    text = re.sub(r'^\*\*Brock Webb\*\*\s*$', '', text, flags=re.MULTILINE)
    # Remove the --- separator right after the removed content
    # (only if it's at the start after cleanup)
    text = re.sub(r'^\s*---\s*$', '', text, count=1, flags=re.MULTILINE)
    return text.lstrip("\n")


def strip_web_only_from_intro(text):
    """Remove web-only sections from intro.md for PDF output."""
    # Remove ## Contents heading and surrounding separators
    # The {tableofcontents} directive is already stripped, but the heading remains
    text = re.sub(r'---\s*\n+## Contents\s*\n+', '', text)

    # Remove the Download tip admonition block (converted to blockquote by process_directives)
    text = re.sub(
        r'> \*\*💡 Tip: Download\*\*\n>\n(> .*\n)*',
        '',
        text
    )

    # Clean up any double --- separators left behind
    text = re.sub(r'\n---\s*\n---\s*\n', '\n---\n', text)

    return text


def process_file(filepath, filename):
    with open(filepath, "r") as f:
        text = f.read()

    text = strip_jupytext_frontmatter(text)
    text = process_directives(text)

    if filename == "cover.md":
        return None

    if filename == "intro.md":
        text = strip_cover_image_from_intro(text)
        text = strip_web_only_from_intro(text)

    text = number_figures(text, filename)

    # Keep original headings as-is — "# Chapter N - Title" is explicit and clear.
    # No number-sections, no stripping, no {.unnumbered} markers needed.

    return text


def detect_font(preferred, fallbacks):
    result = subprocess.run(
        ["typst", "fonts"], capture_output=True, text=True
    )
    available = result.stdout
    for font in [preferred] + fallbacks:
        if font in available:
            return font
    return preferred


def write_typst_cover():
    """Write a Typst include file for the full-bleed cover page."""
    template_content = f"""// Full-bleed cover page for ai4stats PDF
#page(margin: 0pt)[
  #image("{COVER_IMAGE_REL}", width: 100%, height: 100%, fit: "stretch")
]
#pagebreak()
"""
    path = os.path.join(SCRIPTS_DIR, "typst_cover.typ")
    with open(path, "w") as f:
        f.write(template_content)
    return path


def main():
    print("=" * 60)
    print("AI for Official Statistics -- PDF Build (pandoc + typst)")
    print("=" * 60)

    if not shutil.which("pandoc"):
        print("ERROR: pandoc not found. Install with: brew install pandoc")
        sys.exit(1)

    pandoc_ver = subprocess.run(
        ["pandoc", "--version"], capture_output=True, text=True
    ).stdout.split("\n")[0]
    print(f"  {pandoc_ver}")

    if not shutil.which("typst"):
        print("ERROR: typst not found. Install with: brew install typst")
        sys.exit(1)

    typst_ver = subprocess.run(
        ["typst", "--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"  typst {typst_ver}")

    if not os.path.exists(COVER_IMAGE_ABS):
        print(f"ERROR: Cover image not found: {COVER_IMAGE_ABS}")
        sys.exit(1)

    # Detect best available fonts
    print("\nDetecting fonts...")
    body_font = detect_font(
        "Libertinus Serif",
        ["New Computer Modern", "STIX Two Text", "Source Serif Pro",
         "Noto Serif", "Georgia", "Palatino"]
    )
    mono_font = detect_font(
        "Fira Code",
        ["JetBrains Mono", "Source Code Pro", "Inconsolata",
         "DejaVu Sans Mono", "Menlo", "Courier New"]
    )
    print(f"  Body: {body_font}")
    print(f"  Mono: {mono_font}")

    # Read TOC
    myst_yml = os.path.join(BOOK_DIR, "myst.yml")
    toc_files = read_toc(myst_yml)
    print(f"\nTOC: {len(toc_files)} files")

    cover_typ_path = write_typst_cover()
    print(f"Cover template: {cover_typ_path}")

    # Process each file
    combined_md = []
    for filename in toc_files:
        filepath = os.path.join(BOOK_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP (not found): {filename}")
            continue

        result = process_file(filepath, filename)
        if result is None:
            print(f"  SKIP (cover): {filename}")
            continue

        combined_md.append(result)
        print(f"  OK: {filename}")

    # Write combined markdown
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    combined_path = os.path.join(EXPORTS_DIR, "_combined.md")

    # Pandoc YAML metadata — toc disabled here; we inject cover + #outline() manually
    # so the order is: cover → TOC → chapters (not TOC → cover → chapters)
    metadata = f"""---
top-level-division: chapter
mainfont: "{body_font}"
monofont: "{mono_font}"
fontsize: 11pt
papersize: us-letter
margin-left: 1in
margin-right: 1in
margin-top: 1in
margin-bottom: 1in
---

"""

    # Cover + TOC injected as raw Typst at the top of the body.
    # Typst generates #outline() after rendering all headings, so it sees every chapter.
    cover_raw = f"""```{{=typst}}
#page(margin: 0pt)[
  #image("{COVER_IMAGE_REL}", width: 100%, height: 100%, fit: "stretch")
]
#outline(depth: 2, indent: 1em)
#pagebreak()
```

"""

    # Join sections with pagebreak between each file
    separator = "\n\n```{=typst}\n#pagebreak()\n```\n\n"

    combined_text = separator.join(combined_md)

    # Load table numbering and build per-header queues for sequential consumption
    table_numbers = load_table_numbers()
    numbering_queues = build_numbering_queues(table_numbers) if table_numbers else {}
    if table_numbers:
        print(f"\nTable numbering: {len(table_numbers)} entries loaded")

    # Apply custom column widths from table_map.yaml (pops captions from queues)
    table_map = load_table_map()
    if table_map:
        print(f"\nApplying table map ({len(table_map)} entries)...")
        combined_text = apply_table_map(combined_text, table_map, numbering_queues)

    # Convert remaining numbered markdown tables to Typst (pops from same queues)
    combined_text = inject_table_numbers(combined_text, numbering_queues)

    # Escape # in code blocks so Typst doesn't interpret them as function calls
    combined_text = escape_hash_in_code_blocks(combined_text)

    with open(combined_path, "w") as f:
        f.write(metadata)
        f.write(cover_raw)
        f.write(combined_text)

    print(f"\nCombined markdown: {combined_path}")
    print(f"  Size: {os.path.getsize(combined_path) / 1024:.0f} KB")

    # Run pandoc with typst backend
    output_pdf = os.path.join(EXPORTS_DIR, "ai4stats.pdf")
    cmd = [
        "pandoc",
        combined_path,
        "-o", output_pdf,
        "--pdf-engine=typst",
        f"--include-in-header={os.path.join(SCRIPTS_DIR, 'typst_header.typ')}",
        f"--resource-path={BOOK_DIR}:{os.path.join(BOOK_DIR, 'images')}:{FIGURES_DIR}:{REPO_ROOT}",
    ]

    print(f"\nRunning pandoc + typst...")
    print(f"  Output: {output_pdf}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        print(f"\nPandoc FAILED (exit {result.returncode})")
        print("STDERR:")
        for line in result.stderr.strip().split("\n")[-80:]:
            print(f"  {line}")
        log_path = os.path.join(EXPORTS_DIR, "pandoc_build.log")
        with open(log_path, "w") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        print(f"\nFull log: {log_path}")

        # Also keep combined.md for debugging
        debug_path = os.path.join(EXPORTS_DIR, "_combined_DEBUG.md")
        shutil.copy2(combined_path, debug_path)
        print(f"Debug markdown: {debug_path}")
        sys.exit(1)
    else:
        size_mb = os.path.getsize(output_pdf) / (1024 * 1024)
        print(f"\nSUCCESS: {output_pdf}")
        print(f"  Size: {size_mb:.1f} MB")

        if result.stderr:
            warnings = [l for l in result.stderr.split("\n") if l.strip()]
            if warnings:
                print(f"\n  Warnings ({len(warnings)}):")
                for w in warnings[:10]:
                    print(f"    {w}")
                if len(warnings) > 10:
                    print(f"    ... and {len(warnings) - 10} more")

    if os.path.exists(combined_path):
        os.remove(combined_path)
        print(f"  Cleaned up: {combined_path}")

    strip_blank_first_page(output_pdf)

    # Copy to datestamped version so committed ai4stats.pdf is not overwritten
    today = date.today().strftime("%Y-%m-%d")
    dated_name = f"ai4stats_{today}.pdf"
    dated_path = os.path.join(EXPORTS_DIR, dated_name)
    shutil.copy2(output_pdf, dated_path)
    print(f"  Datestamped copy: {dated_path}")


def escape_typst(text):
    """Escape Typst special characters in user-facing content text."""
    text = text.replace('\\', '\\\\')
    text = text.replace('#', '\\#')
    text = text.replace('$', '\\$')
    text = text.replace('~', '\\~')
    text = text.replace('<', '\\<')
    text = text.replace('>', '\\>')
    text = text.replace('@', '\\@')
    return text


def escape_hash_in_code_blocks(text):
    """Escape # inside fenced code blocks so Typst doesn't interpret them.

    Uses a line-by-line state machine to avoid cross-block regex matching.
    Skips ```{=typst} blocks — those are raw Typst passthrough.
    """
    lines = text.split('\n')
    result = []
    in_code_block = False
    is_typst_block = False

    for line in lines:
        stripped = line.strip()

        if not in_code_block and stripped.startswith('```'):
            in_code_block = True
            is_typst_block = '{=typst}' in stripped
            result.append(line)
            continue

        if in_code_block and stripped == '```':
            in_code_block = False
            is_typst_block = False
            result.append(line)
            continue

        if in_code_block and not is_typst_block:
            result.append(line.replace('#', '\\#'))
        else:
            result.append(line)

    return '\n'.join(result)


def load_table_numbers():
    """Load table inventory and build chapter-sequential numbering.

    Returns ordered list of (file, header_key, label) tuples.
    Preserves order so duplicate headers are assigned distinct sequential numbers.
    """
    import json
    from collections import defaultdict

    inv_path = os.path.join(SCRIPTS_DIR, "table_inventory.json")
    if not os.path.exists(inv_path):
        return []

    with open(inv_path) as f:
        tables = json.load(f)

    chapter_counters = defaultdict(int)
    result = []

    for t in tables:
        fname = t["file"]
        if "chapter-" not in fname:
            continue
        ch_num = int(fname.replace("chapter-", "").replace(".md", ""))
        chapter_counters[ch_num] += 1
        seq = chapter_counters[ch_num]
        header_key = " | ".join(c.strip() for c in t["header_match"].split("|"))
        result.append((fname, header_key, f"Table {ch_num}.{seq}"))

    return result


def build_numbering_queues(table_numbers):
    """Convert ordered list to per-header-key queues for sequential consumption.

    Returns dict: header_key -> deque of labels.
    First pop for a given key gets the first occurrence, etc.
    """
    from collections import defaultdict, deque
    queues = defaultdict(deque)
    for (fname, header_key, label) in table_numbers:
        queues[header_key].append(label)
    return queues


def inject_table_numbers(text, numbering_queues):
    """Convert numbered markdown tables to raw Typst blocks with captions.

    Handles standalone tables (| col |) and blockquoted tables (> | col |).
    Tables already replaced by apply_table_map() are gone from markdown by the
    time this runs, so no skip-set is needed.
    """
    if not numbering_queues:
        return text

    lines = text.split("\n")
    result = []
    i = 0
    numbered = 0

    while i < len(lines):
        line = lines[i]

        # Standalone markdown table
        if line.startswith("|") and "|" in line[1:]:
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1

            if len(table_lines) < 2:
                result.extend(table_lines)
                continue

            header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
            header_key = " | ".join(header_cells)

            if header_key in numbering_queues and numbering_queues[header_key]:
                caption = numbering_queues[header_key].popleft()
                sep_idx = 1
                data_rows = [r for r in table_lines[sep_idx + 1:]
                             if not re.match(r'^\|[\s\-|:]+\|$', r)]
                n_cols = len(header_cells)
                columns = ["1fr"] * n_cols
                result.append(md_table_to_typst(header_cells, data_rows, columns, caption=caption))
                numbered += 1
            else:
                result.extend(table_lines)
            continue

        # Blockquoted markdown table (inside admonition)
        if line.startswith("> |") and "|" in line[3:]:
            table_lines = []
            while i < len(lines) and lines[i].startswith("> |"):
                table_lines.append(lines[i])
                i += 1

            stripped_lines = [l[2:] for l in table_lines]

            if len(stripped_lines) < 2:
                result.extend(table_lines)
                continue

            header_cells = [c.strip() for c in stripped_lines[0].strip("|").split("|")]
            header_key = " | ".join(header_cells)

            if header_key in numbering_queues and numbering_queues[header_key]:
                caption = numbering_queues[header_key].popleft()
                sep_idx = 1
                data_rows = [r for r in stripped_lines[sep_idx + 1:]
                             if not re.match(r'^\|[\s\-|:]+\|$', r)]
                n_cols = len(header_cells)
                columns = ["1fr"] * n_cols
                result.append(md_table_to_typst(header_cells, data_rows, columns, caption=caption))
                numbered += 1
            else:
                result.extend(table_lines)
            continue

        result.append(line)
        i += 1

    if numbered:
        print(f"  Table numbering: converted {numbered} table(s) to Typst with captions")
    return "\n".join(result)


def number_figures(text, filename):
    """Add figure numbers. Currently only Chapter 10 has a figure."""
    if filename == "chapter-10.md":
        text = text.replace(
            "![SDL Risk Classification Decision Tree",
            "![Figure 10.1: SDL Risk Classification Decision Tree"
        )
    return text


def load_table_map():
    """Load custom column proportions from scripts/table_map.yaml. Returns list or []."""
    map_path = os.path.join(SCRIPTS_DIR, "table_map.yaml")
    if not os.path.exists(map_path):
        return []
    with open(map_path) as f:
        data = yaml.safe_load(f)
    return data.get("tables", []) if data else []


def md_table_to_typst(header_cells, data_rows, columns, caption=None):
    """Convert a markdown table to a raw Typst table block with custom column widths.

    Wraps caption + table in #block(breakable: false) to prevent page splits.
    """
    lines = ["```{=typst}", "#block(breakable: false)["]
    if caption:
        lines.append(f"#align(left)[#text(weight: \"bold\")[{escape_typst(caption)}]]")
        lines.append("#v(0.3em)")
    col_spec = ", ".join(columns)
    lines += [
        f"#table(",
        f"  columns: ({col_spec}),",
        f"  inset: 8pt,",
        f"  stroke: 0.5pt + luma(180),",
        f"  table.header(",
    ]
    header_typst = ", ".join(f"[*{escape_typst(c.strip())}*]" for c in header_cells)
    lines.append(f"    {header_typst},")
    lines.append("  ),")
    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        row_typst = ", ".join(f"[{escape_typst(c)}]" for c in cells)
        lines.append(f"  {row_typst},")
    lines.append(")")
    lines.append("]")  # close #block
    lines.append("```")
    return "\n".join(lines)


def apply_table_map(text, table_map, numbering_queues=None):
    """Replace mapped markdown tables with raw Typst blocks."""
    if not table_map:
        return text

    # Build lookup: header_match (normalized) -> entry
    lookup = {}
    for entry in table_map:
        key = " | ".join(c.strip() for c in entry["header_match"].split("|"))
        lookup[key] = entry

    lines = text.split("\n")
    result = []
    i = 0
    replaced = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("|") and "|" in line[1:]:
            # Collect full table block
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1

            if len(table_lines) < 2:
                result.extend(table_lines)
                continue

            # Normalize header for lookup
            header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
            header_key = " | ".join(header_cells)

            if header_key in lookup:
                entry = lookup[header_key]
                caption = None
                if numbering_queues and header_key in numbering_queues and numbering_queues[header_key]:
                    caption = numbering_queues[header_key].popleft()
                sep_idx = 1  # skip separator row
                data_rows = [r for r in table_lines[sep_idx + 1:]
                             if not re.match(r'^\|[\s\-|:]+\|$', r)]
                typst_block = md_table_to_typst(header_cells, data_rows, entry["columns"], caption=caption)
                result.append(typst_block)
                replaced += 1
            else:
                result.extend(table_lines)
            continue

        result.append(line)
        i += 1

    if replaced:
        print(f"  Table map: replaced {replaced} table(s) with custom Typst blocks")
    return "\n".join(result)


def strip_blank_first_page(pdf_path):
    """Remove page 1 (blank pandoc/typst title page) from the output PDF."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("\nNOTE: pypdf not installed — blank first page not removed.")
        print("  Install with: pip install pypdf")
        return

    if not os.path.exists(pdf_path):
        return

    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    if total < 2:
        print(f"\nPDF has {total} page(s), skipping first-page strip.")
        return

    tmp_path = pdf_path + ".tmp"
    writer = PdfWriter()
    for page in reader.pages[1:]:
        writer.add_page(page)

    with open(tmp_path, "wb") as f:
        writer.write(f)

    os.replace(tmp_path, pdf_path)
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"\n  Stripped blank title page: {total} -> {total - 1} pages ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
