"""
Read-only GitHub REST access and reviewed-commit authorization.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import parse_qsl, unquote, urlsplit

import requests

_API_VERSION = "2026-03-10"
_DEFAULT_TIMEOUT = (5, 20)
_DEFAULT_MAX_PAGES = 100
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_OWNER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPO_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9_.-])?\Z")
_LOGIN_PATTERN = _OWNER_PATTERN
_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9._~-]+\Z")
_QUERY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_QUERY_VALUE_PATTERN = re.compile(r"[A-Za-z0-9._~-]*\Z")
_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING"})
_SUBSTANTIVE_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
_CHECK_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "waiting", "requested", "pending"}
)
_CHECK_CONCLUSIONS = frozenset(
    {
        "action_required",
        "cancelled",
        "failure",
        "neutral",
        "success",
        "skipped",
        "stale",
        "timed_out",
        "startup_failure",
    }
)


class GitHubError(RuntimeError):
    """Base class for sanitized GitHub client failures."""


class GitHubValidationError(GitHubError, ValueError):
    """Raised when caller-controlled GitHub input is unsafe or malformed."""


class GitHubAPIError(GitHubError):
    """Raised when GitHub cannot provide a successful response."""


class GitHubDataError(GitHubError):
    """Raised when a successful GitHub response has an invalid shape."""


@dataclass(frozen=True)
class ReviewResult:
    """Immutable reviewed-commit authorization result."""

    approved: bool
    pr_number: int | None
    reason: str


@dataclass(frozen=True)
class _ReviewState:
    state: str
    commit_id: str
    submitted_at: datetime


class GitHubClient:
    """A small read-only client for the GitHub evidence used by the queue."""

    def __init__(
        self,
        token: str,
        repo: str,
        *,
        base_url: str = "https://api.github.com",
        session: Any | None = None,
        timeout: tuple[float, float] = _DEFAULT_TIMEOUT,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        """
        Initialize the client with a bearer token and ``owner/repository`` name.

        :param token: A GitHub token. It is installed directly on the session and
            is never retained in a printable client field.
        :param repo: The repository in ``owner/name`` form.
        :param base_url: The HTTPS GitHub REST API origin.
        :param session: An optional requests-compatible session.
        :param timeout: Connect and read timeouts in seconds.
        :param max_pages: The finite pagination safety bound.

        :raises GitHubValidationError: If any constructor input is unsafe.
        """
        _validate_token(token)
        _validate_repo(repo)
        self.base_url = _validate_base_url(base_url)
        self.repo = repo
        self._timeout = _validate_timeout(timeout)
        if type(max_pages) is not int or not 1 <= max_pages <= 1_000:
            raise GitHubValidationError("max_pages must be an integer from 1 to 1000")
        self._max_pages = max_pages
        self._session = requests.Session() if session is None else session
        headers = getattr(self._session, "headers", None)
        if not hasattr(headers, "update"):
            raise GitHubValidationError("session must provide mutable headers")
        cast(Any, headers).update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
            }
        )

    def __repr__(self) -> str:
        """Return a token-free representation."""
        return (
            f"{type(self).__name__}(repo={self.repo!r}, "
            f"base_url={self.base_url!r}, max_pages={self._max_pages!r})"
        )

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> object:
        """
        GET and JSON-decode a relative GitHub API path.

        :param path: A validated relative API path beginning with ``/``.
        :param params: Optional scalar query parameters.

        :returns: The JSON-decoded payload.

        :raises GitHubAPIError: If the request fails.
        :raises GitHubDataError: If GitHub returns malformed JSON.
        :raises GitHubValidationError: If the path or parameters are unsafe.
        """
        response = self._request(path, params=params)
        return self._decode_json(response)

    def paginated_get(self, path: str, *, key: str | None = None) -> list[dict[str, object]]:
        """
        Read all object rows from a list or keyed-list GitHub endpoint.

        :param path: A relative API path, optionally with an existing query string.
        :param key: The list field for endpoints that return an object envelope.

        :returns: All rows in server order.

        :raises GitHubDataError: If an envelope, row, or page count is malformed.
        """
        _validate_api_path(path)
        query_fields = {name for name, _ in parse_qsl(urlsplit(path).query)}
        if query_fields & {"page", "per_page"}:
            raise GitHubValidationError("paginated path must not set page or per_page")
        if key is not None and (type(key) is not str or _QUERY_NAME_PATTERN.fullmatch(key) is None):
            raise GitHubValidationError("pagination key is invalid")

        rows: list[dict[str, object]] = []
        for page in range(1, self._max_pages + 1):
            payload = self.get(path, params={"per_page": 100, "page": page})
            if key is None:
                batch = payload
            elif type(payload) is dict:
                batch = cast(dict[object, object], payload).get(key)
            else:
                raise GitHubDataError("GitHub API returned malformed paginated response")
            if type(batch) is not list or len(batch) > 100:
                raise GitHubDataError("GitHub API returned malformed paginated response")
            typed_batch = cast(list[object], batch)
            if any(type(row) is not dict for row in typed_batch):
                raise GitHubDataError("GitHub API returned malformed paginated response")
            rows.extend(cast(list[dict[str, object]], typed_batch))
            if len(typed_batch) < 100:
                return rows

        raise GitHubDataError("GitHub pagination limit exceeded")

    def file_exists(self, path: str, *, ref: str) -> bool:
        """
        Check that a file exists at an exact full commit SHA.

        :param path: A safe repository-relative file path.
        :param ref: The exact 40-character commit SHA.

        :returns: ``False`` only when GitHub returns 404, otherwise ``True`` for a
            valid file response.

        :raises GitHubAPIError: For authorization, server, or network failures.
        :raises GitHubDataError: For malformed successful responses.
        """
        _validate_script_path(path)
        _validate_sha(ref)
        response = self._request(
            f"/repos/{self.repo}/contents/{path}",
            params={"ref": ref},
            allow_not_found=True,
        )
        if response.status_code == 404:
            return False
        payload = self._decode_json(response)
        if (
            type(payload) is not dict
            or payload.get("type") != "file"
            or payload.get("path") != path
        ):
            raise GitHubDataError("GitHub API returned malformed contents response")
        return True

    def reviewed_commit(
        self,
        sha: str,
        *,
        script_path: str,
        allowed_reviewers: set[str],
        required_checks: set[str],
    ) -> ReviewResult:
        """
        Authorize an exact PR head reviewed by a protected reviewer.

        The selected pull request must be open or merged, an authorized reviewer's
        current substantive review must approve this exact SHA, no current
        authorized review on that pull request may request changes, every required
        check must be uniquely completed with success, and the script must exist at
        the same SHA.

        :param sha: The requested full commit SHA.
        :param script_path: The protected script selected by policy.
        :param allowed_reviewers: The explicit non-empty protected reviewer set.
        :param required_checks: The explicit non-empty protected check-name set.

        :returns: A deterministic, fail-closed authorization result.

        :raises GitHubValidationError: If caller-controlled input is malformed.
        :raises GitHubAPIError: If GitHub evidence cannot be retrieved.
        """
        _validate_sha(sha)
        _validate_script_path(script_path)
        reviewers = _normalize_reviewers(allowed_reviewers)
        checks_required = _normalize_required_checks(required_checks)

        try:
            qualifying_pulls = self._qualifying_pull_numbers(sha)
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub pull evidence")
        if not qualifying_pulls:
            return ReviewResult(
                False,
                None,
                "no qualifying pull request has the requested SHA as its head",
            )

        approved_pulls: list[int] = []
        blocked = False
        try:
            for number in qualifying_pulls:
                reviews = self.paginated_get(f"/repos/{self.repo}/pulls/{number}/reviews")
                approved, pull_blocked = _effective_review_decision(reviews, reviewers, sha)
                if approved and not pull_blocked:
                    approved_pulls.append(number)
                blocked = blocked or pull_blocked
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub review evidence")

        if blocked:
            return ReviewResult(
                False,
                None,
                "an authorized reviewer currently requests changes",
            )
        if not approved_pulls:
            return ReviewResult(
                False,
                None,
                "no authorized reviewer approved the requested SHA",
            )
        selected_pr = min(approved_pulls)

        try:
            check_runs = self.paginated_get(
                f"/repos/{self.repo}/commits/{sha}/check-runs?filter=latest",
                key="check_runs",
            )
            unsuccessful = _unsuccessful_required_checks(check_runs, checks_required)
        except GitHubDataError:
            return ReviewResult(False, None, "malformed GitHub check evidence")
        if unsuccessful:
            return ReviewResult(
                False,
                selected_pr,
                "required checks are not uniquely successful: " + ", ".join(unsuccessful),
            )

        try:
            exists = self.file_exists(script_path, ref=sha)
        except GitHubDataError:
            return ReviewResult(False, selected_pr, "malformed GitHub contents evidence")
        if not exists:
            return ReviewResult(
                False,
                selected_pr,
                "script does not exist at the requested SHA",
            )
        return ReviewResult(
            True,
            selected_pr,
            "approved PR with successful checks and script",
        )

    def _qualifying_pull_numbers(self, sha: str) -> list[int]:
        pulls = self.paginated_get(f"/repos/{self.repo}/commits/{sha}/pulls")
        numbers: list[int] = []
        seen: set[int] = set()
        for pull in pulls:
            number = pull.get("number")
            if type(number) is not int or number <= 0 or number in seen:
                raise GitHubDataError("GitHub API returned malformed pull evidence")
            seen.add(number)
            details = self.get(f"/repos/{self.repo}/pulls/{number}")
            if _is_qualifying_pull(details, sha):
                numbers.append(number)
        return sorted(numbers)

    def _request(
        self,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        _validate_api_path(path)
        _validate_query_params(params)
        try:
            response = self._session.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self._timeout,
            )
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        status_code = getattr(response, "status_code", None)
        if type(status_code) is not int:
            raise GitHubDataError("GitHub API returned malformed HTTP response")
        if allow_not_found and status_code == 404:
            return response
        try:
            response.raise_for_status()
        except requests.RequestException:
            raise GitHubAPIError("GitHub API request failed") from None
        return response

    @staticmethod
    def _decode_json(response: Any) -> object:
        try:
            return response.json()
        except (TypeError, ValueError):
            raise GitHubDataError("GitHub API returned malformed JSON") from None


def _validate_token(token: object) -> None:
    if (
        type(token) is not str
        or not token
        or token != token.strip()
        or any(ord(character) < 33 or ord(character) == 127 for character in token)
    ):
        raise GitHubValidationError("GitHub token is invalid")


def _validate_repo(repo: object) -> None:
    if type(repo) is not str:
        raise GitHubValidationError("repository must use owner/name form")
    parts = repo.split("/")
    if (
        len(parts) != 2
        or _OWNER_PATTERN.fullmatch(parts[0]) is None
        or _REPO_PATTERN.fullmatch(parts[1]) is None
        or parts[1] in {".", ".."}
    ):
        raise GitHubValidationError("repository must use safe owner/name form")


def _validate_base_url(base_url: object) -> str:
    if type(base_url) is not str or not base_url:
        raise GitHubValidationError("base_url must be a valid HTTPS URL")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GitHubValidationError("base_url must be a valid HTTPS URL")
    _validate_url_path(parsed.path or "/")
    return base_url.rstrip("/")


def _validate_timeout(timeout: object) -> tuple[float, float]:
    if type(timeout) is not tuple or len(timeout) != 2:
        raise GitHubValidationError("timeout must contain connect and read seconds")
    values = cast(tuple[object, object], timeout)
    if any(
        type(value) not in {int, float}
        or not math.isfinite(cast(float, value))
        or cast(float, value) <= 0
        for value in values
    ):
        raise GitHubValidationError("timeout values must be finite and positive")
    return cast(tuple[float, float], timeout)


def _validate_api_path(path: object) -> None:
    if type(path) is not str or not path.startswith("/") or path.startswith("//"):
        raise GitHubValidationError("GitHub API path must be relative")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise GitHubValidationError("GitHub API path is invalid")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise GitHubValidationError("GitHub API path must be relative")
    _validate_url_path(parsed.path)
    try:
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise GitHubValidationError("GitHub API query is invalid") from None
    for name, value in query:
        if (
            _QUERY_NAME_PATTERN.fullmatch(name) is None
            or _QUERY_VALUE_PATTERN.fullmatch(value) is None
        ):
            raise GitHubValidationError("GitHub API query is invalid")


def _validate_url_path(path: str) -> None:
    if "\\" in path or "%" in path:
        raise GitHubValidationError("URL path is invalid")
    if path == "/":
        return
    segments = path.split("/")[1:]
    if any(
        not segment
        or segment in {".", ".."}
        or _PATH_SEGMENT_PATTERN.fullmatch(unquote(segment)) is None
        for segment in segments
    ):
        raise GitHubValidationError("URL path is invalid")


def _validate_query_params(params: Mapping[str, object] | None) -> None:
    if params is None:
        return
    if not isinstance(params, Mapping):
        raise GitHubValidationError("GitHub API parameters must be a mapping")
    for name, value in params.items():
        if type(name) is not str or _QUERY_NAME_PATTERN.fullmatch(name) is None:
            raise GitHubValidationError("GitHub API parameter name is invalid")
        if type(value) is int:
            if value < 0:
                raise GitHubValidationError("GitHub API parameter value is invalid")
        elif type(value) is str:
            if not value or _QUERY_VALUE_PATTERN.fullmatch(value) is None:
                raise GitHubValidationError("GitHub API parameter value is invalid")
        else:
            raise GitHubValidationError("GitHub API parameter value is invalid")


def _validate_sha(sha: object) -> None:
    if type(sha) is not str or _SHA_PATTERN.fullmatch(sha) is None:
        raise GitHubValidationError("SHA must be a lowercase 40-character hexadecimal value")


def _validate_script_path(path: object) -> None:
    if type(path) is not str or not path or path.startswith("/"):
        raise GitHubValidationError("script path must be repository-relative")
    if "\\" in path or "%" in path or "?" in path or "#" in path:
        raise GitHubValidationError("script path is invalid")
    segments = path.split("/")
    if any(
        not segment or segment in {".", ".."} or _PATH_SEGMENT_PATTERN.fullmatch(segment) is None
        for segment in segments
    ):
        raise GitHubValidationError("script path is invalid")


def _normalize_reviewers(reviewers: object) -> frozenset[str]:
    if type(reviewers) not in {set, frozenset} or not reviewers:
        raise GitHubValidationError("allowed_reviewers must be an explicit non-empty set")
    normalized: set[str] = set()
    for reviewer in cast(set[object] | frozenset[object], reviewers):
        if type(reviewer) is not str or _LOGIN_PATTERN.fullmatch(reviewer) is None:
            raise GitHubValidationError("allowed_reviewers contains an invalid GitHub login")
        login = reviewer.casefold()
        if login in normalized:
            raise GitHubValidationError("allowed_reviewers contains duplicate GitHub logins")
        normalized.add(login)
    return frozenset(normalized)


def _normalize_required_checks(checks: object) -> frozenset[str]:
    if type(checks) not in {set, frozenset} or not checks:
        raise GitHubValidationError("required_checks must be an explicit non-empty set")
    normalized: set[str] = set()
    for check in cast(set[object] | frozenset[object], checks):
        if not _is_valid_check_name(check):
            raise GitHubValidationError("required_checks contains an invalid check name")
        normalized.add(cast(str, check))
    return frozenset(normalized)


def _is_valid_check_name(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= 100
        and all(ord(character) >= 32 and ord(character) != 127 for character in value)
    )


def _parse_timestamp(value: object) -> datetime:
    if type(value) is not str or not value:
        raise GitHubDataError("GitHub API returned malformed timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise GitHubDataError("GitHub API returned malformed timestamp") from None
    if parsed.tzinfo is None:
        raise GitHubDataError("GitHub API returned malformed timestamp")
    return parsed.astimezone(timezone.utc)


def _is_qualifying_pull(payload: object, sha: str) -> bool:
    if type(payload) is not dict:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    pull = cast(dict[object, object], payload)
    head = pull.get("head")
    if type(head) is not dict:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    head_sha = cast(dict[object, object], head).get("sha")
    if type(head_sha) is not str or _SHA_PATTERN.fullmatch(head_sha) is None:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    state = pull.get("state")
    merged_at = pull.get("merged_at")
    if state not in {"open", "closed"}:
        raise GitHubDataError("GitHub API returned malformed pull evidence")
    if state == "open":
        if merged_at is not None:
            raise GitHubDataError("GitHub API returned malformed pull evidence")
        qualifies_by_state = True
    elif merged_at is None:
        qualifies_by_state = False
    else:
        _parse_timestamp(merged_at)
        qualifies_by_state = True
    return head_sha == sha and qualifies_by_state


def _effective_review_decision(
    rows: list[dict[str, object]],
    allowed_reviewers: frozenset[str],
    sha: str,
) -> tuple[bool, bool]:
    substantive: dict[str, list[_ReviewState]] = {}
    for row in rows:
        state = row.get("state")
        commit_id = row.get("commit_id")
        submitted_at = row.get("submitted_at")
        user = row.get("user")
        if type(state) is not str or state not in _REVIEW_STATES or type(user) is not dict:
            raise GitHubDataError("GitHub API returned malformed review evidence")
        login = cast(dict[object, object], user).get("login")
        if type(login) is not str or _LOGIN_PATTERN.fullmatch(login) is None:
            raise GitHubDataError("GitHub API returned malformed review evidence")
        if state == "PENDING":
            raise GitHubDataError("GitHub API returned pending review evidence")
        if type(commit_id) is not str or _SHA_PATTERN.fullmatch(commit_id) is None:
            raise GitHubDataError("GitHub API returned malformed review evidence")
        timestamp = _parse_timestamp(submitted_at)
        normalized_login = login.casefold()
        if normalized_login not in allowed_reviewers or state == "COMMENTED":
            continue
        substantive.setdefault(normalized_login, []).append(
            _ReviewState(state, commit_id, timestamp)
        )

    latest: list[_ReviewState] = []
    for reviewer_rows in substantive.values():
        latest_time = max(row.submitted_at for row in reviewer_rows)
        tied = [row for row in reviewer_rows if row.submitted_at == latest_time]
        if len({(row.state, row.commit_id) for row in tied}) != 1:
            raise GitHubDataError("GitHub API returned ambiguous review ordering")
        latest.append(tied[0])
    blocked = any(row.state == "CHANGES_REQUESTED" for row in latest)
    approved = any(row.state == "APPROVED" and row.commit_id == sha for row in latest)
    return approved, blocked


def _unsuccessful_required_checks(
    rows: list[dict[str, object]],
    required_checks: frozenset[str],
) -> list[str]:
    matching: dict[str, list[tuple[str, object]]] = {name: [] for name in required_checks}
    for row in rows:
        name = row.get("name")
        status = row.get("status")
        conclusion = row.get("conclusion")
        if (
            not _is_valid_check_name(name)
            or type(status) is not str
            or status not in _CHECK_STATUSES
            or (
                conclusion is not None
                and (type(conclusion) is not str or conclusion not in _CHECK_CONCLUSIONS)
            )
        ):
            raise GitHubDataError("GitHub API returned malformed check evidence")
        if cast(str, name) in matching:
            matching[cast(str, name)].append((status, conclusion))
    return sorted(
        name
        for name, evidence in matching.items()
        if len(evidence) != 1 or evidence[0] != ("completed", "success")
    )
