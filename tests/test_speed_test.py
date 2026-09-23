from __future__ import annotations

import io
from collections.abc import Iterable
from unittest.mock import Mock

import pytest
import requests

import speed_test


class FakeResponse:
    def __init__(
        self,
        chunks: Iterable[bytes] = (),
        *,
        status_code: int = 200,
        reason: str = "OK",
        stream_error: requests.RequestException | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.status_code = status_code
        self.reason = reason
        self.stream_error = stream_error
        self.closed = False

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.reason = self.reason
            raise requests.HTTPError(response=response)

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        assert chunk_size == speed_test.CHUNK_SIZE
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error


class FakeSession:
    def __init__(self, outcomes: Iterable[FakeResponse | BaseException]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def sequence_timer(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def test_bytes_to_decimal_megabytes() -> None:
    assert speed_test.bytes_to_mb(2_500_000) == pytest.approx(2.5)


def test_summary_uses_only_successful_attempts() -> None:
    results = [
        speed_test.AttemptResult(2.0, 4_000_000),
        speed_test.AttemptResult(9.0, error="timeout"),
        speed_test.AttemptResult(1.0, 2_000_000),
    ]

    summary = speed_test.summarize_results(results)

    assert summary.successful_requests == 2
    assert summary.total_requests == 3
    assert summary.average_time_seconds == pytest.approx(1.5)
    assert summary.downloaded_mb == pytest.approx(6.0)
    assert summary.average_speed_mb_per_second == pytest.approx(2.0)


def test_summary_handles_no_successes() -> None:
    summary = speed_test.summarize_results(
        [speed_test.AttemptResult(1.0, error="failed")]
    )

    assert summary.successful_requests == 0
    assert summary.average_time_seconds == 0
    assert summary.downloaded_bytes == 0
    assert summary.average_speed_bytes_per_second == 0


def test_summary_handles_zero_duration_success() -> None:
    summary = speed_test.summarize_results([speed_test.AttemptResult(0.0, 10)])
    assert summary.successful_requests == 1
    assert summary.average_speed_bytes_per_second == 0


def test_download_once_streams_and_counts_actual_bytes() -> None:
    response = FakeResponse([b"abc", b"", b"defgh"])
    session = FakeSession([response])

    result = speed_test.download_once(
        session,  # type: ignore[arg-type]
        "https://example.com/file",
        12.0,
        timer=sequence_timer(5.0, 7.0),
    )

    assert result == speed_test.AttemptResult(2.0, 8)
    assert response.closed
    assert session.calls == [
        {
            "url": "https://example.com/file",
            "stream": True,
            "timeout": 12.0,
            "allow_redirects": True,
        }
    ]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (requests.Timeout("slow"), "request timed out"),
        (requests.ConnectionError("DNS failed"), "connection error: DNS failed"),
    ],
)
def test_download_once_handles_request_errors(
    error: requests.RequestException,
    message: str,
) -> None:
    result = speed_test.download_once(
        FakeSession([error]),  # type: ignore[arg-type]
        "https://example.com/file",
        30.0,
        timer=sequence_timer(10.0, 10.5),
    )

    assert result == speed_test.AttemptResult(0.5, error=message)


def test_download_once_handles_http_error() -> None:
    result = speed_test.download_once(
        FakeSession([FakeResponse(status_code=404, reason="Not Found")]),  # type: ignore[arg-type]
        "https://example.com/missing",
        30.0,
        timer=sequence_timer(1.0, 1.25),
    )

    assert result == speed_test.AttemptResult(0.25, error="HTTP 404 Not Found")


def test_partial_read_failure_discards_received_bytes() -> None:
    response = FakeResponse([b"partial"], stream_error=requests.ConnectionError("lost"))

    result = speed_test.download_once(
        FakeSession([response]),  # type: ignore[arg-type]
        "https://example.com/file",
        30.0,
        timer=sequence_timer(1.0, 2.0),
    )

    assert not result.successful
    assert result.downloaded_bytes == 0
    assert response.closed


def test_run_downloads_is_sequential_and_continues_after_error() -> None:
    first = FakeResponse([b"one"])
    third = FakeResponse([b"three"])
    session = FakeSession([first, requests.Timeout(), third])
    output = io.StringIO()

    results = speed_test.run_downloads(
        session,  # type: ignore[arg-type]
        "https://example.com/file",
        3,
        5.0,
        timer=sequence_timer(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        output=output,
    )

    assert [result.successful for result in results] == [True, False, True]
    assert len(session.calls) == 3
    assert ["[1/3] OK", "[2/3] ERROR", "[3/3] OK"] == [
        line.split(" — ")[0] for line in output.getvalue().splitlines()
    ]


def test_parser_defaults() -> None:
    args = speed_test.build_parser().parse_args(["https://example.com/file"])
    assert args.requests == 10
    assert args.timeout == 30.0


def test_parser_accepts_overrides() -> None:
    args = speed_test.build_parser().parse_args(
        ["http://example.com", "--requests", "3", "--timeout", "2.5"]
    )
    assert args.requests == 3
    assert args.timeout == 2.5


@pytest.mark.parametrize(
    "arguments",
    [
        ["not-a-url"],
        ["ftp://example.com/file"],
        ["https:///missing-host"],
        ["https://example.com", "--requests", "0"],
        ["https://example.com", "--timeout", "nan"],
    ],
)
def test_parser_rejects_invalid_input(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        speed_test.build_parser().parse_args(arguments)
    assert error.value.code != 0


def test_print_summary_contains_all_indicators() -> None:
    output = io.StringIO()
    speed_test.print_summary(
        speed_test.Summary(2, 3, 0.5, 3_000_000, 6_000_000),
        output=output,
    )
    text = output.getvalue()
    assert "Successful requests: 2/3" in text
    assert "Average request time: 0.50 s" in text
    assert "Downloaded: 3.00 MB" in text
    assert "Average download speed: 6.00 MB/s" in text


def test_main_returns_success_when_any_request_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        speed_test,
        "run_downloads",
        lambda *_args, **_kwargs: [
            speed_test.AttemptResult(1.0, error="failed"),
            speed_test.AttemptResult(1.0, 1_000_000),
        ],
    )
    assert speed_test.main(["https://example.com", "--requests", "2"], output=io.StringIO()) == 0


def test_main_returns_failure_when_all_requests_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        speed_test,
        "run_downloads",
        lambda *_args, **_kwargs: [speed_test.AttemptResult(1.0, error="failed")],
    )
    assert speed_test.main(["https://example.com"], output=io.StringIO()) == 1


def test_entrypoint_handles_ctrl_c_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupted(_argv: object = None) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(speed_test, "main", interrupted)

    assert speed_test.entrypoint([]) == 130
    captured = capsys.readouterr()
    assert "Interrupted by user" in captured.err
    assert "Traceback" not in captured.err
