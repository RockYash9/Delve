"""
Tests for src/verification.py.

The Gemini client is entirely mocked — these test the decision logic
(when to skip an API call, how to handle a bad/missing response, how
to degrade gracefully on failure), not real model output quality.
"""

from unittest.mock import MagicMock

from src import verification


def _fake_client_returning(parsed_value):
    client = MagicMock()
    client.models.generate_content.return_value = MagicMock(parsed=parsed_value)
    return client


def test_check_contradictions_skips_api_call_with_fewer_than_two_sources(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(verification, "_get_client", lambda: mock_client)

    result = verification.check_contradictions([{"title": "A", "content": "x"}])

    assert result.has_contradictions is False
    assert result.contradictions == []
    mock_client.models.generate_content.assert_not_called()


def test_check_contradictions_returns_parsed_report(monkeypatch):
    expected = verification.ContradictionReport(
        has_contradictions=True,
        contradictions=[
            verification.Contradiction(
                topic="launch date",
                source_a_title="Source A",
                source_a_claim="March 2026",
                source_b_title="Source B",
                source_b_claim="June 2026",
                explanation="Sources disagree on the launch date.",
            )
        ],
    )
    monkeypatch.setattr(
        verification, "_get_client", lambda: _fake_client_returning(expected)
    )

    result = verification.check_contradictions(
        [
            {"title": "Source A", "content": "Launches March 2026"},
            {"title": "Source B", "content": "Launches June 2026"},
        ]
    )

    assert result.has_contradictions is True
    assert len(result.contradictions) == 1
    assert result.contradictions[0].topic == "launch date"


def test_check_contradictions_handles_unparseable_response(monkeypatch):
    monkeypatch.setattr(
        verification, "_get_client", lambda: _fake_client_returning(None)
    )

    result = verification.check_contradictions(
        [{"title": "A", "content": "x"}, {"title": "B", "content": "y"}]
    )

    assert result.has_contradictions is False
    assert result.contradictions == []


def test_check_contradictions_degrades_gracefully_on_api_failure(monkeypatch):
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")
    monkeypatch.setattr(verification, "_get_client", lambda: client)

    result = verification.check_contradictions(
        [{"title": "A", "content": "x"}, {"title": "B", "content": "y"}]
    )

    # Must not raise — Deep Research is an enhancement, not a hard dependency.
    assert result.has_contradictions is False


def test_verify_claims_skips_api_call_with_no_sources(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(verification, "_get_client", lambda: mock_client)

    result = verification.verify_claims("Some answer text.", [])

    assert result.total_claims_checked == 0
    mock_client.models.generate_content.assert_not_called()


def test_verify_claims_returns_parsed_report(monkeypatch):
    expected = verification.VerificationReport(
        total_claims_checked=5,
        supported_count=4,
        unsupported_claims=[
            verification.UnsupportedClaim(
                claim="The product costs $50.",
                reason="No source mentions a specific price.",
            )
        ],
    )
    monkeypatch.setattr(
        verification, "_get_client", lambda: _fake_client_returning(expected)
    )

    result = verification.verify_claims(
        "The product costs $50 and launches in March.",
        [{"title": "A", "content": "Launches in March."}],
    )

    assert result.total_claims_checked == 5
    assert result.supported_count == 4
    assert len(result.unsupported_claims) == 1
    assert "price" in result.unsupported_claims[0].reason


def test_verify_claims_degrades_gracefully_on_api_failure(monkeypatch):
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("API down")
    monkeypatch.setattr(verification, "_get_client", lambda: client)

    result = verification.verify_claims("answer", [{"title": "A", "content": "x"}])

    assert result.total_claims_checked == 0
    assert result.unsupported_claims == []
