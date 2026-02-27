#!/usr/bin/env python3
"""
Compile a resorch paper manuscript from Markdown to LaTeX PDF.

Usage:
    python scripts/compile_paper.py <workspace_path>

Supported Markdown:
  - # Title, ## Section, ### Subsection (with optional numbering)
  - ## Abstract, ## References (special handling)
  - **bold**, *italic*, `code`, $math$, $$display math$$
  - Markdown pipe tables (with alignment)
  - [text](url) links, ![alt](path) images
  - Fenced code blocks (```...```)
  - Ordered and unordered lists (single-level)
  - Blockquotes (> prefix)
  - Horizontal rules (--- or ***)
Not supported:
  - Nested lists, multi-paragraph list items
  - Footnotes, definition lists
  - HTML blocks, inline HTML
  - Task lists, strikethrough
"""

import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import argparse

# Set in main() so inline markdown image paths can be rebased for .tex output location.
_WORKSPACE_ROOT = None
_OUTPUT_DIR = None
_INLINE_FIGURE_PATHS = set()  # Populated during format_inline/parse_manuscript


def escape_latex(text):
    """Escape LaTeX special characters in plain text segments."""
    backslash_token = "@@BACKSLASH@@"
    out = text.replace("\\", backslash_token)
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircle{}"),
    ]
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out.replace(backslash_token, r"\textbackslash{}")


def escape_url_for_latex(url):
    """Escape LaTeX-special chars in URL for use inside \\href{...}."""
    replacements = [("%", r"\%"), ("#", r"\#"), ("&", r"\&"), ("_", r"\_"), ("~", r"\~{}")]
    out = url
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def strip_heading_number(text):
    """Strip leading heading numbering like '1. Intro' or '2.1 Dataset'."""
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", text).strip()


def is_numeric_like(text):
    value = text.strip()
    if not value:
        return False
    value = value.replace(",", "")
    if value.endswith("%"):
        value = value[:-1].strip()
    value = value.replace("±", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?-[+-]?\d+(?:\.\d+)?", value):
        return True
    return False


def parse_table_row(line):
    content = line.strip()
    if content.startswith("|"):
        content = content[1:]
    if content.endswith("|"):
        content = content[:-1]
    pipe_placeholder = "@@PIPE@@"
    content = content.replace(r"\|", pipe_placeholder)
    return [cell.strip().replace(pipe_placeholder, "|") for cell in content.split("|")]


def parse_separator_row(row):
    aligns = []
    for cell in row:
        token = cell.strip()
        if token.startswith(":") and token.endswith(":"):
            aligns.append("c")
        elif token.endswith(":"):
            aligns.append("r")
        elif token.startswith(":"):
            aligns.append("l")
        else:
            aligns.append(None)
    return aligns


def format_inline(text):
    """
    Convert inline Markdown to LaTeX using placeholder-protect strategy:
    1) Extract markdown/url/math markers into placeholders
    2) Escape remaining plain text
    3) Restore placeholders as LaTeX commands
    """
    placeholders = {}
    counter = [0]

    def stash(value):
        key = "@@PH%03d@@" % counter[0]
        counter[0] += 1
        placeholders[key] = value
        return key

    def apply_plain_rewrites(text_value):
        work = re.sub(
            r"\bDOI:\s*([^\s]+)",
            lambda m: "https://doi.org/" + m.group(1).rstrip(".,;"),
            text_value,
            flags=re.IGNORECASE,
        )

        # Markdown images: ![alt](path) -> \includegraphics
        def md_image_repl(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            # Rebase relative to output_dir if workspace context is available.
            if _WORKSPACE_ROOT is not None and _OUTPUT_DIR is not None:
                # Try resolving relative to workspace root, then relative to manuscript dir.
                for base in (_WORKSPACE_ROOT, _WORKSPACE_ROOT / "paper"):
                    candidate = base / match.group(2)
                    if candidate.exists():
                        _INLINE_FIGURE_PATHS.add(str(candidate.resolve()))
                        img_path = os.path.relpath(str(candidate), str(_OUTPUT_DIR)).replace("\\", "/")
                        break
            clean_alt = re.sub(r"^(?:Figure|Fig\.?)\s*\d+[.:]?\s*", "", alt_text).strip()
            caption = escape_latex(clean_alt) if clean_alt else escape_latex(img_path.split("/")[-1].replace("_", " "))
            return stash(
                r"\begin{figure}[htbp]" + "\n"
                r"\centering" + "\n"
                r"\includegraphics[width=\textwidth]{" + img_path + "}" + "\n"
                r"\caption{" + caption + "}" + "\n"
                r"\end{figure}"
            )

        work = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", md_image_repl, work)

        # Markdown links: [text](url) -> \href{url}{text}
        def md_link_repl(match):
            link_text = match.group(1)
            url = match.group(2)
            return stash(r"\href{" + escape_url_for_latex(url) + "}{" + escape_latex(link_text) + "}")

        work = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", md_link_repl, work)

        def url_repl(match):
            raw = match.group(0)
            trailing = ""
            while raw and raw[-1] in ".,);!?":
                trailing = raw[-1] + trailing
                raw = raw[:-1]
            return stash(r"\url{" + raw + "}") + trailing

        work = re.sub(r"https?://[^\s]+", url_repl, work)
        work = re.sub(r"\bosf\.io/[A-Za-z0-9]+\b", lambda m: stash(r"\url{https://" + m.group(0) + "}"), work)
        work = re.sub(r"\|i-j\|", lambda _m: stash(r"$|i-j|$"), work)
        work = re.sub(r"\b(R)\^2\b", lambda m: stash(m.group(1) + r"\textsuperscript{2}"), work)
        work = re.sub(r"\b(chi)\^2\b", lambda m: stash(m.group(1) + r"\textsuperscript{2}"), work, flags=re.IGNORECASE)
        return work

    work_text = text

    display_math_pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
    while True:
        m = display_math_pattern.search(work_text)
        if not m:
            break
        token = stash(r"\[" + m.group(1) + r"\]")
        work_text = work_text[: m.start()] + token + work_text[m.end() :]

    inline_math_pattern = re.compile(r"(?<!\d)\$(?!\s)(.+?)(?<!\s)\$(?!\d)")
    while True:
        m = inline_math_pattern.search(work_text)
        if not m:
            break
        token = stash("$" + m.group(1) + "$")
        work_text = work_text[: m.start()] + token + work_text[m.end() :]

    code_pattern = re.compile(r"`([^`]+)`")
    while True:
        m = code_pattern.search(work_text)
        if not m:
            break
        code_text = escape_latex(m.group(1))
        token = stash(r"\texttt{" + code_text + "}")
        work_text = work_text[: m.start()] + token + work_text[m.end() :]

    work_text = apply_plain_rewrites(work_text)

    bold_pattern = re.compile(r"\*\*([^*]+?)\*\*")
    while True:
        m = bold_pattern.search(work_text)
        if not m:
            break
        inner = apply_plain_rewrites(m.group(1))
        inner = escape_latex(inner)
        token = stash(r"\textbf{" + inner + "}")
        work_text = work_text[: m.start()] + token + work_text[m.end() :]

    italic_pattern = re.compile(r"(?<!\*)\*([^*]+?)\*(?!\*)")
    while True:
        m = italic_pattern.search(work_text)
        if not m:
            break
        inner = apply_plain_rewrites(m.group(1))
        inner = escape_latex(inner)
        token = stash(r"\textit{" + inner + "}")
        work_text = work_text[: m.start()] + token + work_text[m.end() :]

    escaped = escape_latex(work_text)
    for key in sorted(placeholders.keys(), reverse=True):
        escaped = escaped.replace(key, placeholders[key])
    return escaped


def choose_column_specs(headers, rows, aligns):
    col_count = len(headers)
    specs = []
    for idx in range(col_count):
        align = aligns[idx] if idx < len(aligns) else None
        col_cells = []
        if idx < len(headers):
            col_cells.append(headers[idx])
        for row in rows:
            if idx < len(row):
                col_cells.append(row[idx])
            else:
                col_cells.append("")
        numeric_count = 0
        non_empty = 0
        max_len = 0
        for cell in col_cells[1:]:
            val = cell.strip()
            if not val:
                continue
            non_empty += 1
            max_len = max(max_len, len(val))
            if is_numeric_like(val):
                numeric_count += 1
        numeric_ratio = float(numeric_count) / float(non_empty) if non_empty else 0.0

        if align == "c":
            spec = "c"
        elif align == "r":
            spec = "r"
        elif align == "l":
            spec = "l"
        else:
            if numeric_ratio >= 0.6:
                spec = "r"
            elif max_len > 48:
                spec = "p{4.0cm}"
            else:
                spec = "l"
        specs.append(spec)
    return specs


def normalize_row_length(row, col_count):
    if len(row) < col_count:
        return row + [""] * (col_count - len(row))
    if len(row) > col_count:
        return row[:col_count]
    return row


def table_to_latex(table_lines, caption):
    if len(table_lines) < 2:
        return ""
    rows = [parse_table_row(line) for line in table_lines]
    header = rows[0]
    sep = rows[1]
    data_rows = rows[2:] if len(rows) > 2 else []

    col_count = len(header)
    header = normalize_row_length(header, col_count)
    sep = normalize_row_length(sep, col_count)
    data_rows = [normalize_row_length(r, col_count) for r in data_rows]

    aligns = parse_separator_row(sep)
    specs = choose_column_specs(header, data_rows, aligns)
    colspec = "".join(specs)

    header_tex = " & ".join(format_inline(cell) for cell in header) + r" \\"
    rows_tex = [" & ".join(format_inline(cell) for cell in row) + r" \\" for row in data_rows]

    is_wide = col_count >= 8
    is_long = len(data_rows) > 12
    caption_tex = format_inline(caption) if caption else ""
    pieces = []

    if is_long:
        pieces.append(r"\begin{center}")
        if is_wide:
            pieces.append(r"\footnotesize")
        pieces.append(r"\begin{longtable}{" + colspec + "}")
        if caption_tex:
            pieces.append(r"\caption{" + caption_tex + r"}\\")
        pieces.append(r"\toprule")
        pieces.append(header_tex)
        pieces.append(r"\midrule")
        pieces.append(r"\endfirsthead")
        pieces.append(r"\toprule")
        pieces.append(header_tex)
        pieces.append(r"\midrule")
        pieces.append(r"\endhead")
        pieces.extend(rows_tex)
        pieces.append(r"\bottomrule")
        pieces.append(r"\end{longtable}")
        pieces.append(r"\end{center}")
    else:
        env = "table*" if is_wide else "table"
        pieces.append(r"\begin{" + env + r"}[htbp]")
        pieces.append(r"\centering")
        if col_count >= 7:
            pieces.append(r"\footnotesize")
        if caption_tex:
            pieces.append(r"\caption{" + caption_tex + r"}")
        pieces.append(r"\begin{adjustbox}{max width=\textwidth}")
        pieces.append(r"\begin{tabular}{" + colspec + "}")
        pieces.append(r"\toprule")
        pieces.append(header_tex)
        pieces.append(r"\midrule")
        pieces.extend(rows_tex)
        pieces.append(r"\bottomrule")
        pieces.append(r"\end{tabular}")
        pieces.append(r"\end{adjustbox}")
        pieces.append(r"\end{" + env + r"}")
    return "\n".join(pieces) + "\n"


def convert_reference_line(line):
    entry = re.sub(r"^\s*\d+\.\s*", "", line).strip()
    if not entry:
        return ""
    entry = re.sub(
        r"\bDOI:\s*([^\s]+)",
        lambda m: "https://doi.org/" + m.group(1).rstrip(".,;"),
        entry,
        flags=re.IGNORECASE,
    )
    return format_inline(entry)


def collect_figures(workspace_path, output_dir):
    fig_dir = workspace_path / "results" / "fig"
    if not fig_dir.exists():
        return []
    pngs = sorted(glob.glob(str(fig_dir / "*.png")))
    pdfs = sorted(glob.glob(str(fig_dir / "*.pdf")))
    figures = pngs + pdfs
    blocks = []
    for fig in figures:
        fig_path = pathlib.Path(fig)
        if str(fig_path.resolve()) in _INLINE_FIGURE_PATHS:
            continue  # Already included inline in manuscript
        rel = os.path.relpath(str(fig_path), str(output_dir))
        rel = rel.replace("\\", "/")
        caption = format_inline(fig_path.stem.replace("_", " "))
        block = "\n".join(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\includegraphics[width=\textwidth]{" + rel + "}",
                r"\caption{" + caption + "}",
                r"\end{figure}",
                "",
            ]
        )
        blocks.append(block)
    return blocks


def parse_manuscript(manuscript_path):
    title = ""
    state = "preamble"
    abstract_parts = []
    body_parts = []
    references = []
    pending_table_caption = None
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph = " ".join(part.strip() for part in paragraph_lines if part.strip())
        paragraph_lines = []
        if not paragraph:
            return
        converted = format_inline(paragraph)
        if state == "abstract":
            abstract_parts.append(converted + "\n")
        else:
            body_parts.append(converted + "\n")

    with open(manuscript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    saw_title_marker = False
    idx = 0
    while idx < len(lines):
        raw = lines[idx].rstrip("\n")
        stripped = raw.strip()

        if stripped.startswith("_Created:"):
            idx += 1
            continue

        if raw.startswith("# "):
            flush_paragraph()
            candidate = raw[2:].strip()
            if candidate.lower() == "title":
                saw_title_marker = True
                title = ""
            else:
                title = candidate
                saw_title_marker = False
            idx += 1
            continue

        if saw_title_marker and not title and stripped and not stripped.startswith("#"):
            title = stripped
            saw_title_marker = False
            idx += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:].strip()
            pending_table_caption = None
            if heading.lower() == "title":
                saw_title_marker = True
                title = ""
                idx += 1
                continue
            elif heading.lower() == "abstract":
                state = "abstract"
            elif heading.lower() == "references":
                state = "references"
            else:
                state = "section"
                section_title = strip_heading_number(heading)
                body_parts.append(r"\section{" + format_inline(section_title) + "}\n")
            idx += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            sub = stripped[4:].strip()
            table_match = re.match(r"(?i)^table\s+\d+\.\s*(.+)$", sub)
            if table_match:
                pending_table_caption = table_match.group(1).strip()
            else:
                sub_title = strip_heading_number(sub)
                body_parts.append(r"\subsection{" + format_inline(sub_title) + "}\n")
            idx += 1
            continue

        if stripped.startswith("#### "):
            flush_paragraph()
            sub = stripped[5:].strip()
            sub_title = strip_heading_number(sub)
            body_parts.append(r"\paragraph{" + format_inline(sub_title) + "}\n")
            idx += 1
            continue

        # Bold table caption: **Table N. Caption text**
        bold_table_match = re.match(r"^\*\*(?:Table)\s+\d+\.?\s*(.+?)\*\*$", stripped)
        if bold_table_match:
            flush_paragraph()
            pending_table_caption = bold_table_match.group(1).strip()
            idx += 1
            continue

        if state == "references":
            if stripped:
                if re.match(r"^\d+\.\s+", stripped):
                    references.append(stripped)
                elif references:
                    references[-1] += " " + stripped
            idx += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while idx < len(lines):
                maybe = lines[idx].strip()
                if not maybe.startswith("|"):
                    break
                table_lines.append(maybe)
                idx += 1
            body_parts.append(table_to_latex(table_lines, pending_table_caption))
            pending_table_caption = None
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            idx += 1
            code_lines = []
            while idx < len(lines):
                maybe_raw = lines[idx].rstrip("\n")
                if maybe_raw.strip().startswith("```"):
                    idx += 1
                    break
                code_lines.append(maybe_raw)
                idx += 1
            code_block = "\n".join(
                [
                    r"\begin{verbatim}",
                    "\n".join(code_lines),
                    r"\end{verbatim}",
                ]
            )
            if state == "abstract":
                abstract_parts.append(code_block + "\n")
            else:
                body_parts.append(code_block + "\n")
            continue

        unordered_match = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if unordered_match or ordered_match:
            flush_paragraph()
            list_env = "itemize" if unordered_match else "enumerate"
            list_pattern = re.compile(r"^[-*]\s+(.+)$") if unordered_match else re.compile(r"^\d+\.\s+(.+)$")
            items = []
            while idx < len(lines):
                maybe_stripped = lines[idx].strip()
                match = list_pattern.match(maybe_stripped)
                if not match:
                    break
                items.append(format_inline(match.group(1).strip()))
                idx += 1
            list_lines = [r"\begin{" + list_env + "}"]
            for item in items:
                list_lines.append(r"\item " + item)
            list_lines.append(r"\end{" + list_env + "}")
            list_block = "\n".join(list_lines)
            if state == "abstract":
                abstract_parts.append(list_block + "\n")
            else:
                body_parts.append(list_block + "\n")
            continue

        if stripped in ("---", "***"):
            flush_paragraph()
            body_parts.append(r"\hrule" + "\n")
            idx += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph()
            quote_lines = []
            while idx < len(lines):
                maybe_stripped = lines[idx].strip()
                if not maybe_stripped.startswith("> "):
                    break
                quote_lines.append(format_inline(maybe_stripped[2:].strip()))
                idx += 1
            quote_block = "\\begin{quote}\n" + "\n".join(quote_lines) + "\n\\end{quote}"
            if state == "abstract":
                abstract_parts.append(quote_block + "\n")
            else:
                body_parts.append(quote_block + "\n")
            continue

        if not stripped:
            flush_paragraph()
            if state in ("abstract", "section"):
                if state == "abstract":
                    abstract_parts.append("\n")
                else:
                    body_parts.append("\n")
            idx += 1
            continue

        # Standalone markdown image: emit as block-level figure, not inline.
        standalone_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if standalone_img and state in ("abstract", "section"):
            flush_paragraph()
            alt_text = standalone_img.group(1)
            img_path = standalone_img.group(2)
            clean_alt = re.sub(r"^(?:Figure|Fig\.?)\s*\d+[.:]?\s*", "", alt_text).strip()
            resolved_path = img_path
            if _WORKSPACE_ROOT is not None and _OUTPUT_DIR is not None:
                for base in (_WORKSPACE_ROOT, _WORKSPACE_ROOT / "paper"):
                    candidate = base / img_path
                    if candidate.exists():
                        _INLINE_FIGURE_PATHS.add(str(candidate.resolve()))
                        resolved_path = os.path.relpath(str(candidate), str(_OUTPUT_DIR)).replace("\\", "/")
                        break
            if not clean_alt:
                fig_caption = escape_latex(resolved_path.split("/")[-1].replace("_", " "))
            else:
                fig_caption = format_inline(clean_alt)
            fig_block = "\n".join([
                r"\begin{figure}[htbp]",
                r"\centering",
                r"\includegraphics[width=\textwidth]{" + resolved_path + "}",
                r"\caption{" + fig_caption + "}",
                r"\end{figure}",
            ])
            if state == "abstract":
                abstract_parts.append(fig_block + "\n")
            else:
                body_parts.append(fig_block + "\n")
            idx += 1
            continue

        if state in ("abstract", "section", "preamble"):
            paragraph_lines.append(raw)
        idx += 1

    flush_paragraph()
    if not title:
        title = "Untitled Manuscript"
    return title, "".join(abstract_parts).strip(), "".join(body_parts).strip(), references


def build_latex(title, abstract_text, body_text, reference_lines, figure_blocks, workspace_path=None, output_dir=None):
    refs = []
    if reference_lines:
        refs.append(r"\begin{thebibliography}{99}")
        for i, line in enumerate(reference_lines, start=1):
            refs.append(r"\bibitem{ref%d} %s" % (i, convert_reference_line(line)))
        refs.append(r"\end{thebibliography}")

    fig_text = "\n".join(figure_blocks).strip()
    refs_text = "\n".join(refs).strip()

    pieces = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{graphicx}",
        r"\usepackage{adjustbox}",
        r"\usepackage{hyperref}",
        r"\usepackage{url}",
        r"\usepackage{textcomp}",
        r"\usepackage{setspace}",
        r"\usepackage{authblk}",
        r"\setstretch{1.15}",
    ]
    if figure_blocks and workspace_path and output_dir:
        fig_dir = workspace_path / "results" / "fig"
        if fig_dir.exists():
            rel_fig = os.path.relpath(str(fig_dir), str(output_dir)).replace("\\", "/") + "/"
            pieces.append(r"\graphicspath{{" + rel_fig + "}}")
    pieces.extend(
        [
        r"\providecommand{\textasciicircle}{\textasciicircum}",
        "",
        r"\title{" + format_inline(title) + "}",
        r"\author{Author Name\\Affiliation}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        abstract_text,
        r"\end{abstract}",
        "",
        body_text,
        "",
        ]
    )
    if fig_text:
        pieces.append(fig_text)
        pieces.append("")
    if refs_text:
        pieces.append(refs_text)
        pieces.append("")
    pieces.append(r"\end{document}")
    return "\n".join(pieces).strip() + "\n"


def insert_citations(body_text, reference_lines):
    ref_count = len(reference_lines)
    if ref_count == 0:
        return body_text

    # Protect verbatim blocks from citation replacement.
    verbatim_blocks = {}
    verbatim_counter = [0]

    def protect_verbatim(match):
        key = "@@VERBATIM%03d@@" % verbatim_counter[0]
        verbatim_counter[0] += 1
        verbatim_blocks[key] = match.group(0)
        return key

    protected = re.sub(
        r"\\begin\{verbatim\}.*?\\end\{verbatim\}",
        protect_verbatim,
        body_text,
        flags=re.DOTALL,
    )

    def bracket_repl(match):
        content = match.group(1)
        parts = [p.strip() for p in content.split(",")]
        all_refs = []
        for part in parts:
            if "-" in part:
                try:
                    start_str, end_str = part.split("-", 1)
                    start_val, end_val = int(start_str.strip()), int(end_str.strip())
                    all_refs.extend(range(start_val, end_val + 1))
                except ValueError:
                    return match.group(0)
            else:
                try:
                    all_refs.append(int(part))
                except ValueError:
                    return match.group(0)
        if all_refs and all(1 <= r <= ref_count for r in all_refs):
            cite_keys = ",".join("ref%d" % r for r in all_refs)
            return r"\cite{" + cite_keys + "}"
        return match.group(0)

    out = re.sub(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]", bracket_repl, protected)

    surname_year_to_idx = {}
    for idx, line in enumerate(reference_lines, start=1):
        entry = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if not entry:
            continue
        surname_match = re.match(r"^\s*([A-Za-z][A-Za-z'`-]*)", entry)
        year_match = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", entry)
        if not surname_match or not year_match:
            continue
        key = (surname_match.group(1).lower(), year_match.group(1).lower())
        if key not in surname_year_to_idx:
            surname_year_to_idx[key] = idx

    def author_year_repl(match):
        surname = match.group(1).lower()
        year = match.group(2).lower()
        ref_idx = surname_year_to_idx.get((surname, year))
        if ref_idx:
            return r"\cite{ref%d}" % ref_idx
        return match.group(0)

    if surname_year_to_idx:
        out = re.sub(r"\(([A-Z][A-Za-z'`-]+)(?:\s+et al\.)?,\s*((?:19|20)\d{2}[a-z]?)\)", author_year_repl, out)

    for key, block in verbatim_blocks.items():
        out = out.replace(key, block)
    return out


def extract_log_errors(log_path):
    if not log_path.exists():
        return ["No LaTeX log file found."]
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for idx, line in enumerate(lines):
        if line.startswith("!"):
            out.append(line)
            if idx + 1 < len(lines):
                out.append(lines[idx + 1])
            if idx + 2 < len(lines):
                out.append(lines[idx + 2])
    if not out:
        out.append("Compilation failed; no explicit '!'-prefixed LaTeX errors were captured.")
    return out


def find_latex_engine():
    """Find available LaTeX engine, preferring xelatex."""
    for engine in ["xelatex", "pdflatex", "lualatex"]:
        path = pathlib.Path("/usr/bin/" + engine)
        if path.exists():
            return str(path)
        found = shutil.which(engine)
        if found:
            return found
    return None


def run_latex(output_dir, tex_filename):
    engine = find_latex_engine()
    if not engine:
        print("ERROR: No LaTeX engine found. Install TeX Live:", file=sys.stderr)
        print("  Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-latex-extra", file=sys.stderr)
        print("  macOS: brew install --cask mactex", file=sys.stderr)
        print("  Or use --no-pdf to generate .tex only.", file=sys.stderr)
        return False
    print("Using LaTeX engine: %s" % engine)
    cmd = [engine, "-interaction=nonstopmode", "-output-directory=" + str(output_dir), tex_filename]
    for pass_num in (1, 2):
        print("Running %s pass %d/2..." % (os.path.basename(engine), pass_num))
        proc = subprocess.run(cmd, cwd=str(output_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            log_path = output_dir / "manuscript.log"
            print("ERROR: %s failed on pass %d." % (os.path.basename(engine), pass_num), file=sys.stderr)
            for line in extract_log_errors(log_path):
                print(line, file=sys.stderr)
            return False
    return True


def cleanup_aux(output_dir):
    for ext in (".aux", ".out"):
        path = output_dir / ("manuscript" + ext)
        if path.exists():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Compile manuscript.md to LaTeX PDF")
    parser.add_argument("workspace", help="Path to resorch workspace")
    parser.add_argument(
        "--manuscript",
        default=None,
        help="Path to manuscript.md (default: <workspace>/paper/manuscript.md)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <workspace>/paper/output/)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Only generate .tex, skip LaTeX compilation",
    )
    args = parser.parse_args()

    workspace = pathlib.Path(args.workspace).resolve()
    manuscript = pathlib.Path(args.manuscript).resolve() if args.manuscript else workspace / "paper" / "manuscript.md"
    output_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else workspace / "paper" / "output"
    if not manuscript.exists():
        print("ERROR: manuscript not found: %s" % manuscript, file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "manuscript.tex"
    pdf_path = output_dir / "manuscript.pdf"

    global _WORKSPACE_ROOT, _OUTPUT_DIR, _INLINE_FIGURE_PATHS
    _WORKSPACE_ROOT = workspace
    _OUTPUT_DIR = output_dir
    _INLINE_FIGURE_PATHS = set()  # Reset for each run

    print("Parsing manuscript...")
    title, abstract_text, body_text, refs = parse_manuscript(manuscript)
    body_text = insert_citations(body_text, refs)

    print("Converting figures...")
    figures = collect_figures(workspace, output_dir)

    print("Building LaTeX document...")
    tex = build_latex(title, abstract_text, body_text, refs, figures, workspace, output_dir)
    tex_path.write_text(tex, encoding="utf-8")

    print("Converting tables...")
    # Tables are converted during parse; this message signals that phase completed.

    if args.no_pdf:
        print("LaTeX source generated: %s" % tex_path)
        return 0

    if not run_latex(output_dir, tex_path.name):
        return 1

    if pdf_path.exists():
        cleanup_aux(output_dir)
        print("PDF generated: %s" % pdf_path)
        print("LaTeX source: %s" % tex_path)
        return 0

    print("ERROR: PDF was not generated: %s" % pdf_path, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
