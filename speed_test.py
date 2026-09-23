"""Sequential HTTP download speed measurement CLI."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TextIO
from urllib.parse import urlsplit

import requests

BYTES_PER_MB = 1_000_000
CHUNK_SIZE = 64 * 1024
DEFAULT_REQUESTS = 10
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """Outcome of one HTTP download attempt."""

    elapsed_seconds: float
    downloaded_bytes: int = 0
    error: str | None = None

    @property
    def successful(self) -> bool:
        """Return whether the full response body was downloaded."""

        return self.error is None

    @property
    def speed_bytes_per_second(self) -> float:
        """Return this attempt's speed, guarding zero-duration test results."""

        if not self.successful or self.elapsed_seconds <= 0:
            return 0.0
        return self.downloaded_bytes / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class Summary:
    """Aggregate metrics calculated from successful attempts."""

    successful_requests: int
    total_requests: int
    average_time_seconds: float
    downloaded_bytes: int
    average_speed_bytes_per_second: float

    @property
    def downloaded_mb(self) -> float:
        """Return the downloaded volume using decimal megabytes."""

        return bytes_to_mb(self.downloaded_bytes)

    @property
    def average_speed_mb_per_second(self) -> float:
        """Return the aggregate speed using decimal megabytes per second."""

        return bytes_to_mb(self.average_speed_bytes_per_second)


def bytes_to_mb(byte_count: int | float) -> float:
    """Convert bytes to decimal megabytes (1 MB = 1,000,000 bytes)."""

    return byte_count / BYTES_PER_MB


def summarize_results(results: Sequence[AttemptResult]) -> Summary:
    """Aggregate only fully successful download attempts."""

    successful = [result for result in results if result.successful]
    successful_count = len(successful)
    downloaded_bytes = sum(result.downloaded_bytes for result in successful)
    total_time = sum(result.elapsed_seconds for result in successful)
    average_time = total_time / successful_count if successful_count else 0.0
    average_speed = downloaded_bytes / total_time if total_time > 0 else 0.0

    return Summary(
        successful_requests=successful_count,
        total_requests=len(results),
        average_time_seconds=average_time,
        downloaded_bytes=downloaded_bytes,
        average_speed_bytes_per_second=average_speed,
    )


def _friendly_request_error(error: requests.RequestException) -> str:
    """Turn expected requests exceptions into concise user-facing messages."""

    if isinstance(error, requests.Timeout):
        return "request timed out"
    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code
        reason = error.response.reason or "HTTP error"
        return f"HTTP {status} {reason}"
    if isinstance(error, requests.ConnectionError):
        detail = str(error).strip()
        return f"connection error: {detail}" if detail else "connection error"
    return str(error).strip() or error.__class__.__name__


def download_once(
    session: requests.Session,
    url: str,
    timeout: float,
    *,
    timer: Callable[[], float] = time.perf_counter,
) -> AttemptResult:
    """Download one response body completely and record its actual byte count."""

    started_at = timer()
    try:
        with session.get(
            url,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            downloaded_bytes = sum(
                len(chunk)
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE)
                if chunk
            )
    except requests.RequestException as error:
        return AttemptResult(
            elapsed_seconds=max(0.0, timer() - started_at),
            error=_friendly_request_error(error),
        )

    return AttemptResult(
        elapsed_seconds=max(0.0, timer() - started_at),
        downloaded_bytes=downloaded_bytes,
    )


def format_attempt(index: int, total: int, result: AttemptResult) -> str:
    """Format one progress line."""

    prefix = f"[{index}/{total}]"
    if not result.successful:
        return f"{prefix} ERROR — {result.error} ({result.elapsed_seconds:.2f} s)"

    size_mb = bytes_to_mb(result.downloaded_bytes)
    speed_mb = bytes_to_mb(result.speed_bytes_per_second)
    return (
        f"{prefix} OK — {size_mb:.2f} MB in {result.elapsed_seconds:.2f} s "
        f"— {speed_mb:.2f} MB/s"
    )


def run_downloads(
    session: requests.Session,
    url: str,
    request_count: int,
    timeout: float,
    *,
    timer: Callable[[], float] = time.perf_counter,
    output: TextIO = sys.stdout,
) -> list[AttemptResult]:
    """Run downloads one by one and report each completed attempt."""

    results: list[AttemptResult] = []
    for index in range(1, request_count + 1):
        result = download_once(session, url, timeout, timer=timer)
        results.append(result)
        print(format_attempt(index, request_count, result), file=output)
    return results


def print_summary(summary: Summary, *, output: TextIO = sys.stdout) -> None:
    """Print all required aggregate indicators."""

    print("\nSummary", file=output)
    print(
        f"Successful requests: {summary.successful_requests}/{summary.total_requests}",
        file=output,
    )
    print(f"Average request time: {summary.average_time_seconds:.2f} s", file=output)
    print(f"Downloaded: {summary.downloaded_mb:.2f} MB", file=output)
    print(
        f"Average download speed: {summary.average_speed_mb_per_second:.2f} MB/s",
        file=output,
    )


def valid_url(value: str) -> str:
    """Argparse converter accepting absolute HTTP and HTTPS URLs only."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid URL: {error}") from error
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise argparse.ArgumentTypeError(
            "URL must be an absolute address using http:// or https://"
        )
    return value


def positive_int(value: str) -> int:
    """Argparse converter for positive integer values."""

    try:
        converted = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if converted <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return converted


def positive_float(value: str) -> float:
    """Argparse converter for positive finite numbers."""

    try:
        converted = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if converted <= 0 or converted == float("inf") or converted != converted:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return converted


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Measure download speed by repeatedly fetching a URL.",
    )
    parser.add_argument("url", type=valid_url, metavar="URL")
    parser.add_argument(
        "--requests",
        type=positive_int,
        default=DEFAULT_REQUESTS,
        help=f"number of sequential requests (default: {DEFAULT_REQUESTS})",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP operation timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, output: TextIO = sys.stdout) -> int:
    """Run the CLI and return its process exit code."""

    args = build_parser().parse_args(argv)
    print("Internet download speed test", file=output)
    print(f"URL: {args.url}", file=output)
    print(f"Requests: {args.requests}\n", file=output)

    with requests.Session() as session:
        results = run_downloads(
            session,
            args.url,
            args.requests,
            args.timeout,
            output=output,
        )

    summary = summarize_results(results)
    print_summary(summary, output=output)
    return 0 if summary.successful_requests else 1


def entrypoint(argv: Sequence[str] | None = None) -> int:
    """Handle user interruption without displaying a traceback."""

    try:
        return main(argv)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(entrypoint())
