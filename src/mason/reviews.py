"""Review records: a critic's findings, kept before compute is spent.

A review is one markdown file under ``<workspace>/mason/reviews/``. The
``review`` tool writes it when a critic finishes, and the file stands
alone: the verdict, the findings, and the text that was reviewed, so a
later session, a report, or a person can read why a plan was approved or
sent back without opening a transcript. The lead's transcript names the
file, and the environment block shows the latest review of the plan, so
the findings survive compaction the way the plan and the notebook do.

The verdict is one of three words. ``approve`` and ``revise`` come from
the critic's ``finish`` call; ``none`` records a review that ended without
a verdict (a turn budget, an error streak, a finish that named no
verdict). A review without a verdict never approves anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mason.errors import MasonError
from mason.session import MasonSession
from mason.skills import SkillError, split_frontmatter

Verdict = Literal["approve", "revise", "none"]
VERDICTS: frozenset[str] = frozenset({"approve", "revise", "none"})

#: The subject word for the living plan; any other subject is a file path.
PLAN_SUBJECT = "plan"

_REVIEWED_TEXT_HEADING = "\n# Reviewed text\n\n"


class ReviewError(MasonError):
    """A review record that cannot be read, naming the file and the rule."""


@dataclass(frozen=True)
class Review:
    """One persisted review: what was judged, by whom, and the verdict."""

    subject: str
    digest: str
    verdict: Verdict
    reviewer: str
    session: str
    transcript: str
    at: str
    findings: str
    text: str
    path: Path


def digest(text: str) -> str:
    """A short, stable fingerprint of the reviewed text.

    Examples:
        >>> digest("a plan\\n") == digest("a plan\\n")
        True
        >>> len(digest("x"))
        16
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def reviews_dir(session: MasonSession) -> Path:
    """Where this workspace keeps its reviews, beside the session transcripts."""
    return session.sessions_dir.parent / "reviews"


def write_review(
    session: MasonSession,
    *,
    subject: str,
    text: str,
    verdict: Verdict,
    reviewer: str,
    transcript: str,
    findings: str,
) -> Path:
    """Persist one review for *session* and return its path.

    Files are named after the session transcript with an ordinal, like
    delegation transcripts, so a session's reviews sort beside it.
    """
    directory = reviews_dir(session)
    directory.mkdir(parents=True, exist_ok=True)
    stem = session.transcript_path.stem
    ordinal = 1 + sum(1 for _ in directory.glob(f"{stem}-review-*.md"))
    path = directory / f"{stem}-review-{ordinal}.md"
    header = {
        "subject": subject,
        "digest": digest(text),
        "verdict": verdict,
        "reviewer": reviewer,
        "session": stem,
        "transcript": transcript,
        "at": datetime.now(UTC).isoformat(),
    }
    # Every value is JSON-quoted, which is valid YAML and keeps a digest of
    # digits or a timestamp from parsing back as a number or a datetime.
    lines = ["---"] + [f"{key}: {json.dumps(value)}" for key, value in header.items()] + ["---"]
    body = f"# Findings\n\n{findings.strip()}\n{_REVIEWED_TEXT_HEADING}{text.rstrip()}\n"
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
    return path


def _parse(path: Path) -> Review:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReviewError(f"{path}: cannot read it: {e}") from e
    try:
        meta, body = split_frontmatter(raw)
    except SkillError as e:
        raise ReviewError(f"{path}: {e}") from None
    fields: dict[str, str] = {}
    for key in ("subject", "digest", "verdict", "reviewer", "session", "transcript", "at"):
        value: Any = meta.get(key)
        if not isinstance(value, str) or not value:
            raise ReviewError(f"{path}: the header lacks a {key!r} string")
        fields[key] = value
    if fields["verdict"] not in VERDICTS:
        raise ReviewError(
            f"{path}: verdict {fields['verdict']!r} is not one of {', '.join(sorted(VERDICTS))}"
        )
    findings, marker, text = body.partition(_REVIEWED_TEXT_HEADING)
    if not marker:
        raise ReviewError(f"{path}: the body has no '# Reviewed text' section")
    verdict: Verdict = fields["verdict"]  # type: ignore[assignment]
    return Review(
        subject=fields["subject"],
        digest=fields["digest"],
        verdict=verdict,
        reviewer=fields["reviewer"],
        session=fields["session"],
        transcript=fields["transcript"],
        at=fields["at"],
        findings=findings.removeprefix("# Findings\n").strip(),
        text=text,
        path=path,
    )


def load_reviews(session: MasonSession) -> list[Review]:
    """Every review in this workspace, oldest first.

    A review the harness cannot read is a loud :class:`ReviewError`: these
    files are written by the harness, so a broken one is damage, not noise.
    """
    directory = reviews_dir(session)
    if not directory.is_dir():
        return []
    return [_parse(path) for path in sorted(directory.glob("*-review-*.md"))]


def latest_plan_review(session: MasonSession) -> Review | None:
    """The most recent review of the plan, whatever its verdict."""
    plan_reviews = [r for r in load_reviews(session) if r.subject == PLAN_SUBJECT]
    return plan_reviews[-1] if plan_reviews else None


def plan_is_approved(session: MasonSession) -> bool:
    """Whether the plan on disk, as it reads now, carries an approving review.

    The digest ties the approval to the text: a plan edited after its
    approval is a different plan and must be reviewed again, and a session
    that resumes with an approved plan unchanged need not.
    """
    current = session.plan_text()
    if not current.strip():
        return False
    wanted = digest(current)
    return any(
        r.subject == PLAN_SUBJECT and r.verdict == "approve" and r.digest == wanted
        for r in load_reviews(session)
    )


def review_block(session: MasonSession, max_chars: int = 3_000) -> str:
    """The ``# Latest review`` section of the environment block, or empty.

    Shown to the lead so the critic's findings outlive compaction. When the
    plan has changed since the review, the block says so: the findings may
    no longer apply, and an approval no longer holds.
    """
    review = latest_plan_review(session)
    if review is None:
        return ""
    current = session.plan_text()
    lines = [
        "# Latest review of the plan",
        f"verdict: {review.verdict}; reviewer: {review.reviewer}; at: {review.at[:19]}; "
        f"record: {review.path.name}",
    ]
    if not current.strip():
        lines.append("PLAN.md has been removed since this review; the verdict no longer holds.")
    elif digest(current) != review.digest:
        lines.append(
            "PLAN.md has changed since this review: its findings may no longer apply, "
            "and its verdict no longer holds. Review the current plan again before "
            "you rely on it."
        )
    findings = review.findings
    if len(findings) > max_chars:
        findings = findings[:max_chars] + f"\n[... findings truncated; read {review.path}]"
    lines.append("")
    lines.append(findings)
    return "\n".join(lines)
