"""
Tests for src/reports.py.
"""

from src import reports


def test_build_report_includes_transcript():
    transcript = [
        ("user", "What are EV tax credits?"),
        ("model", "They changed significantly in 2026."),
    ]
    report = reports.build_report(transcript, sources=[])

    assert "What are EV tax credits?" in report
    assert "They changed significantly in 2026." in report


def test_build_report_deduplicates_sources():
    sources = [
        {"title": "EV Guide", "url": "https://example.com/a"},
        {"title": "EV Guide", "url": "https://example.com/a"},  # duplicate
        {"title": "Other Source", "url": "https://example.com/b"},
    ]
    report = reports.build_report([("user", "hi")], sources)

    assert report.count("https://example.com/a") == 1
    assert "https://example.com/b" in report


def test_build_report_with_no_sources_omits_sources_section():
    report = reports.build_report([("user", "hi")], sources=[])
    assert "## Sources" not in report


def test_save_report_writes_readable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "OUTPUT_DIR", tmp_path)

    path = reports.save_report("# Test Report\n\nHello.")

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "# Test Report\n\nHello."
    assert path.parent == tmp_path
