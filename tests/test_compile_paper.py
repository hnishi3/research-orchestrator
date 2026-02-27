import importlib.util
import base64
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


def load_compile_paper_module():
    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "compile_paper.py"
    spec = importlib.util.spec_from_file_location("compile_paper", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FormatInlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_bold_r_squared_resolves_nested_placeholder(self):
        rendered = self.compile_paper.format_inline("**R^2**")
        self.assertEqual(rendered, r"\textbf{R\textsuperscript{2}}")
        self.assertNotIn("@@PH", rendered)

    def test_italic_chi_squared_resolves_nested_placeholder(self):
        rendered = self.compile_paper.format_inline("*chi^2*")
        self.assertEqual(rendered, r"\textit{chi\textsuperscript{2}}")
        self.assertNotIn("@@PH", rendered)

    def test_escape_latex_backslash_keeps_textbackslash_braces_intact(self):
        rendered = self.compile_paper.escape_latex(r"a\b")
        self.assertIn(r"\textbackslash{}", rendered)
        self.assertNotIn(r"\textbackslash\{\}", rendered)


class TestListParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_unordered_list_converts_to_itemize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text("- item1\n- item2\n", encoding="utf-8")
            _, _, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertIn(r"\begin{itemize}", body)
        self.assertIn(r"\item item1", body)
        self.assertIn(r"\item item2", body)

    def test_ordered_list_converts_to_enumerate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text("1. item1\n2. item2\n", encoding="utf-8")
            _, _, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertIn(r"\begin{enumerate}", body)
        self.assertIn(r"\item item1", body)
        self.assertIn(r"\item item2", body)


class TestMathSupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_inline_math_is_preserved_unescaped(self):
        rendered = self.compile_paper.format_inline("The value $x^2 + y^2$ is important")
        self.assertIn("$x^2 + y^2$", rendered)
        self.assertNotIn(r"\$x^2 + y^2\$", rendered)

    def test_display_math_converts_to_latex_display_delimiters(self):
        rendered = self.compile_paper.format_inline("$$E = mc^2$$")
        self.assertIn(r"\[E = mc^2\]", rendered)


class TestCodeFence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_fenced_code_block_converts_to_verbatim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text("```python\nprint('hello')\n```\n", encoding="utf-8")
            _, _, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertIn(r"\begin{verbatim}", body)
        self.assertIn(r"\end{verbatim}", body)


class TestCitationInsertion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_numeric_bracket_citations_inserted_within_range(self):
        refs = ["1. Ref one", "2. Ref two", "3. Ref three"]
        rendered = self.compile_paper.insert_citations("As shown in [1] and [2]", refs)
        self.assertEqual(rendered, r"As shown in \cite{ref1} and \cite{ref2}")

    def test_numeric_bracket_citation_out_of_range_is_unchanged(self):
        refs = ["1. Ref one", "2. Ref two", "3. Ref three"]
        rendered = self.compile_paper.insert_citations("See [99]", refs)
        self.assertEqual(rendered, "See [99]")


class TestPipeEscaping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_parse_table_row_handles_escaped_pipe(self):
        row = self.compile_paper.parse_table_row(r"| cell1 | value\|extra | cell3 |")
        self.assertEqual(len(row), 3)
        self.assertEqual(row[1], "value|extra")


class TestLogRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_cleanup_aux_keeps_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = pathlib.Path(tmpdir)
            log_file = output_dir / "manuscript.log"
            aux_file = output_dir / "manuscript.aux"
            out_file = output_dir / "manuscript.out"
            log_file.write_text("log", encoding="utf-8")
            aux_file.write_text("aux", encoding="utf-8")
            out_file.write_text("out", encoding="utf-8")

            self.compile_paper.cleanup_aux(output_dir)

            self.assertTrue(log_file.exists())
            self.assertFalse(aux_file.exists())
            self.assertFalse(out_file.exists())


class TestMarkdownLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_markdown_link_with_https_url_is_converted(self):
        rendered = self.compile_paper.format_inline("See [our paper](https://example.com/paper) for details")
        self.assertIn(r"\href{https://example.com/paper}", rendered)
        self.assertNotIn("[our paper]", rendered)

    def test_markdown_link_with_http_url_is_converted(self):
        rendered = self.compile_paper.format_inline("A [link](http://example.com) in text")
        self.assertIn(r"\href{http://example.com}{link}", rendered)


class TestMarkdownImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_markdown_image_with_alt_text_is_converted(self):
        rendered = self.compile_paper.format_inline("![Figure 1](results/fig/plot.png)")
        self.assertIn(r"\includegraphics", rendered)
        self.assertIn("plot.png", rendered)

    def test_markdown_image_without_alt_text_is_converted(self):
        rendered = self.compile_paper.format_inline("![](path/to/img.pdf)")
        self.assertIn(r"\includegraphics", rendered)
        self.assertIn("img.pdf", rendered)


class TestStricterMathRegex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_currency_values_are_escaped_not_math(self):
        rendered = self.compile_paper.format_inline("The cost is $5 and $10")
        self.assertIn(r"\$5 and \$10", rendered)
        self.assertNotRegex(rendered, r"(?<!\\)\$")

    def test_valid_inline_math_is_preserved(self):
        rendered = self.compile_paper.format_inline("The value $x^2$ is important")
        self.assertIn("$x^2$", rendered)
        self.assertNotIn(r"\$x^2\$", rendered)

    def test_currency_range_is_not_treated_as_math(self):
        rendered = self.compile_paper.format_inline("Price $5.00 to $10.00")
        self.assertIn(r"\$5.00 to \$10.00", rendered)
        self.assertNotRegex(rendered, r"(?<!\\)\$")


class TestCLIFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_module_exposes_argparse_and_main(self):
        self.assertTrue(hasattr(self.compile_paper, "argparse"))
        self.assertTrue(hasattr(self.compile_paper, "main"))

    def test_main_parses_cli_args_with_no_pdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            paper_dir = workspace / "paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            manuscript = paper_dir / "manuscript.md"
            manuscript.write_text("# Title\n\n## Abstract\nA short abstract.\n", encoding="utf-8")

            argv = ["compile_paper.py", str(workspace), "--no-pdf"]
            with mock.patch.object(sys, "argv", argv):
                exit_code = self.compile_paper.main()

            self.assertEqual(exit_code, 0)


class TestEngineFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_find_latex_engine_returns_string_or_none(self):
        engine = self.compile_paper.find_latex_engine()
        self.assertTrue(isinstance(engine, str) or engine is None)


class TestImagePathRebasing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_inline_markdown_image_path_is_rebased_from_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            img_path = workspace / "results" / "fig" / "test_plot.png"
            manuscript_path = workspace / "paper" / "manuscript.md"
            output_dir = workspace / "paper" / "output"

            img_path.parent.mkdir(parents=True, exist_ok=True)
            manuscript_path.parent.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + (b"\x00" * 100))
            manuscript_path.write_text("![Test](results/fig/test_plot.png)\n", encoding="utf-8")

            original_workspace_root = self.compile_paper._WORKSPACE_ROOT
            original_output_dir = self.compile_paper._OUTPUT_DIR
            try:
                self.compile_paper._WORKSPACE_ROOT = workspace
                self.compile_paper._OUTPUT_DIR = output_dir
                rendered = self.compile_paper.format_inline("![Test](results/fig/test_plot.png)")
            finally:
                self.compile_paper._WORKSPACE_ROOT = original_workspace_root
                self.compile_paper._OUTPUT_DIR = original_output_dir

            self.assertIn("../../results/fig/test_plot.png", rendered)


class TestURLEscaping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_url_ampersand_and_fragment_hash_are_escaped_in_href(self):
        rendered = self.compile_paper.format_inline("[link](https://example.com/path?a=1&b=2#frag)")
        self.assertIn(r"\href{https://example.com/path?a=1\&b=2\#frag}{link}", rendered)

    def test_url_underscore_is_escaped_in_href(self):
        rendered = self.compile_paper.format_inline("[paper](https://doi.org/10.1234/test_underscore)")
        self.assertIn(r"\href{https://doi.org/10.1234/test\_underscore}{paper}", rendered)


class TestBlockquotes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_multiline_blockquote_converts_to_quote_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text("> This is a quote\n> Second line\n\nNormal text\n", encoding="utf-8")
            _, _, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertIn(r"\begin{quote}", body)
        self.assertIn(r"\end{quote}", body)
        self.assertIn("This is a quote", body)


class TestFigureDeduplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_inline_image_not_duplicated_by_collect_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            paper_dir = workspace / "paper"
            output_dir = paper_dir / "output"
            manuscript = paper_dir / "manuscript.md"
            fig_path = workspace / "results" / "fig" / "plot.png"

            paper_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            fig_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC"))
            manuscript.write_text(
                "\n".join(
                    [
                        "# Figure Dedup Test",
                        "",
                        "## Abstract",
                        "Testing inline image handling.",
                        "",
                        "## 1. Methods",
                        "![](results/fig/plot.png)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            original_workspace_root = self.compile_paper._WORKSPACE_ROOT
            original_output_dir = self.compile_paper._OUTPUT_DIR
            original_inline_paths = self.compile_paper._INLINE_FIGURE_PATHS
            try:
                self.compile_paper._WORKSPACE_ROOT = workspace
                self.compile_paper._OUTPUT_DIR = output_dir
                self.compile_paper._INLINE_FIGURE_PATHS = set()
                self.compile_paper.parse_manuscript(manuscript)
                figures = self.compile_paper.collect_figures(workspace, output_dir)
            finally:
                self.compile_paper._WORKSPACE_ROOT = original_workspace_root
                self.compile_paper._OUTPUT_DIR = original_output_dir
                self.compile_paper._INLINE_FIGURE_PATHS = original_inline_paths

            self.assertEqual(figures, [])


class TestVerbatimProtection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_citation_insertion_preserves_verbatim_content(self):
        refs = ["1. Author A. Title. 2020."]
        body = "See [1] for details.\n\n\\begin{verbatim}\narray[1] = value\n\\end{verbatim}\n"
        result = self.compile_paper.insert_citations(body, refs)
        self.assertIn(r"\cite{ref1}", result)
        self.assertIn("array[1] = value", result)
        self.assertNotIn(r"array\cite{ref1}", result)


class TestBroadenedCitationRegex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_three_digit_citation_is_converted(self):
        refs = ["%d. Ref %d" % (i, i) for i in range(1, 101)]
        body = "See [100] for details."
        result = self.compile_paper.insert_citations(body, refs)
        self.assertIn(r"\cite{ref100}", result)


@unittest.skipUnless(
    load_compile_paper_module().find_latex_engine() is not None,
    "No LaTeX engine available",
)
class TestFigureDeduplicationEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_inline_image_compiles_with_single_figure_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            paper_dir = workspace / "paper"
            output_dir = paper_dir / "output"
            fig_path = workspace / "results" / "fig" / "test_img.png"
            manuscript = paper_dir / "manuscript.md"

            paper_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            fig_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC"))
            manuscript.write_text(
                "\n".join(
                    [
                        "# Figure Deduplication E2E",
                        "",
                        "## Abstract",
                        "Image deduplication integration test.",
                        "",
                        "## 1. Results",
                        "![Caption](results/fig/test_img.png)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            argv = ["compile_paper.py", str(workspace)]
            with mock.patch.object(sys, "argv", argv):
                exit_code = self.compile_paper.main()

            self.assertEqual(exit_code, 0)
            tex_path = output_dir / "manuscript.tex"
            self.assertTrue(tex_path.exists())
            tex = tex_path.read_text(encoding="utf-8")
            self.assertEqual(tex.count(r"\begin{figure}"), 1)


@unittest.skipUnless(
    load_compile_paper_module().find_latex_engine() is not None,
    "No LaTeX engine available",
)
class TestEndToEndCompilation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_main_compiles_pdf_and_retains_log_without_latex_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            paper_dir = workspace / "paper"
            output_dir = paper_dir / "output"
            fig_path = workspace / "results" / "fig" / "test_fig.png"
            manuscript = paper_dir / "manuscript.md"

            paper_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            fig_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQD3A0FDAAAAAElFTkSuQmCC"))
            manuscript.write_text(
                "\n".join(
                    [
                        "# A Test Manuscript",
                        "",
                        "## Abstract",
                        "This abstract has **bold** and *italic* text.",
                        "",
                        "## 1. Methods",
                        "Inline math example: $E=mc^2$ and a link [Example](https://example.com).",
                        "",
                        "### Table 1. Demo table",
                        "| Col A | Col B | Col C |",
                        "|:------|------:|:-----:|",
                        "| a1 | b1 | c1 |",
                        "| a2 | b2 | c2 |",
                        "",
                        "## References",
                        "1. Doe, J. (2020). First reference. DOI:10.1234/example1",
                        "2. Smith, A. (2021). Second reference. https://example.org/ref2",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            argv = ["compile_paper.py", str(workspace)]
            with mock.patch.object(sys, "argv", argv):
                exit_code = self.compile_paper.main()

            self.assertEqual(exit_code, 0)

            pdf_path = output_dir / "manuscript.pdf"
            tex_path = output_dir / "manuscript.tex"
            log_path = output_dir / "manuscript.log"

            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 5 * 1024)
            self.assertTrue(tex_path.exists())
            self.assertTrue(log_path.exists())

            tex = tex_path.read_text(encoding="utf-8")
            self.assertIn(r"\section{", tex)
            self.assertIn(r"\begin{table", tex)
            self.assertIn(r"\begin{thebibliography}", tex)
            self.assertIn(r"\href{", tex)

            log_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            self.assertFalse(any(line.startswith("!") for line in log_lines))


class TestBlockquoteSingleLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_single_line_blockquote_converts_to_quote_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text("> Single line quote\n", encoding="utf-8")
            _, _, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertIn(r"\begin{quote}", body)
        self.assertIn("Single line quote", body)
        self.assertIn(r"\end{quote}", body)


class TestH2TitleExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compile_paper = load_compile_paper_module()

    def test_h2_title_marker_extracts_title_from_next_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text(
                "## Title\nMy Manuscript Title\n\n## Abstract\nAn abstract.\n",
                encoding="utf-8",
            )
            title, abstract, body, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertEqual(title, "My Manuscript Title")
        self.assertNotIn("Untitled", title)
        self.assertNotIn(r"\section{Title}", body)

    def test_h1_title_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manuscript = pathlib.Path(tmpdir) / "manuscript.md"
            manuscript.write_text(
                "# My Direct Title\n\n## Abstract\nAn abstract.\n",
                encoding="utf-8",
            )
            title, _, _, _ = self.compile_paper.parse_manuscript(manuscript)
        self.assertEqual(title, "My Direct Title")


if __name__ == "__main__":
    unittest.main()
