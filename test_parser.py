"""
test_parser.py
--------------
Quick smoke-test for the resume parser.

Usage
-----
    # Set your Gemini API key first:
    $env:GEMINI_API_KEY = "your-key-here"          # PowerShell
    export GEMINI_API_KEY="your-key-here"           # bash / zsh

    # Run against the bundled sample (generates a temp PDF from the .txt):
    python test_parser.py

    # Run against your own file:
    python test_parser.py path/to/your_resume.pdf
    python test_parser.py path/to/your_resume.docx
"""

import sys
import json
import os
import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from parser.parse_resume import parse_resume, extract_text_from_pdf, extract_text_from_docx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sample_docx(txt_path: Path, out_path: Path) -> None:
    """Convert the plain-text sample resume into a DOCX for testing."""
    try:
        from docx import Document
        doc = Document()
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            doc.add_paragraph(line)
        doc.save(str(out_path))
        print(f"[setup] Created sample DOCX at: {out_path}")
    except ImportError:
        print("[setup] python-docx not installed — skipping DOCX generation.")


def _create_sample_pdf(txt_path: Path, out_path: Path) -> None:
    """Convert the plain-text sample resume into a PDF for testing."""
    try:
        from fpdf import FPDF  # optional dependency
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            pdf.cell(0, 7, txt=line, ln=True)
        pdf.output(str(out_path))
        print(f"[setup] Created sample PDF at: {out_path}")
    except ImportError:
        print("[setup] fpdf2 not installed — skipping PDF generation.")
        print("        Install it with:  pip install fpdf2")


def _print_section(title: str, data) -> None:
    """Pretty-print a section of the parsed resume."""
    border = "─" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    if isinstance(data, list):
        if not data:
            print("  (none)")
        for item in data:
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"  {k:>12} : {v}")
                print()
            else:
                print(f"  • {item}")
    else:
        print(f"  {data if data is not None else '(none)'}")


def _run_test(file_path: Path) -> None:
    """Run the parser on file_path and pretty-print the result."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY environment variable is not set.")
        print("   Set it and re-run to test the full Gemini pipeline.\n")
        print("   Falling back to TEXT EXTRACTION ONLY (no Gemini call).\n")
        _run_extraction_only(file_path)
        return

    print(f"\n🔍 Parsing: {file_path.name}")
    print(f"   Using Gemini model: gemini-1.5-flash")

    try:
        result = parse_resume(file_path, api_key=api_key)
    except Exception as exc:
        print(f"\n❌ Parser raised an exception:\n   {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Pretty-print parsed fields
    # -----------------------------------------------------------------------
    print("\n" + "═" * 60)
    print("  ✅  PARSED RESUME")
    print("═" * 60)

    _print_section("NAME",        result.get("name"))
    _print_section("EMAIL",       result.get("email"))
    _print_section("PHONE",       result.get("phone"))
    _print_section("SKILLS",      result.get("skills", []))
    _print_section("EXPERIENCE",  result.get("experience", []))
    _print_section("PROJECTS",    result.get("projects", []))
    _print_section("EDUCATION",   result.get("education", []))

    # -----------------------------------------------------------------------
    # Save raw JSON output
    # -----------------------------------------------------------------------
    out_file = ROOT / "data" / f"{file_path.stem}_parsed.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full JSON saved to: {out_file}")
    print("═" * 60 + "\n")


def _run_extraction_only(file_path: Path) -> None:
    """Show raw extracted text without calling Gemini."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            text = extract_text_from_docx(file_path)
        else:
            # Fallback: read as plain text (e.g., .txt sample)
            text = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"❌ Text extraction failed: {exc}")
        sys.exit(1)

    print("═" * 60)
    print("  📄  EXTRACTED TEXT (first 1000 chars)")
    print("═" * 60)
    print(textwrap.shorten(text, width=1000, placeholder=" … [truncated]"))
    print("═" * 60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        # User provided a specific file
        target = Path(sys.argv[1]).resolve()
        if not target.exists():
            print(f"❌ File not found: {target}")
            sys.exit(1)
    else:
        # Use the bundled sample — try DOCX first, then fall back to .txt
        sample_txt  = DATA_DIR / "sample_resume.txt"
        sample_docx = DATA_DIR / "sample_resume.docx"
        sample_pdf  = DATA_DIR / "sample_resume.pdf"

        if not sample_docx.exists():
            _create_sample_docx(sample_txt, sample_docx)

        if sample_docx.exists():
            target = sample_docx
        elif sample_pdf.exists():
            target = sample_pdf
        else:
            # Last resort: parse .txt directly (extraction-only mode)
            print("[info] No DOCX/PDF sample found — using .txt in extraction-only mode.")
            _run_extraction_only(sample_txt)
            sys.exit(0)

    _run_test(target)
