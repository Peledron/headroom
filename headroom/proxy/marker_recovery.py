"""Turn a blocked recovery marker into a next step the caller can take.

The marker guards refuse a tool call whose input still carries a recovery
marker whose hash is not in the compression store. Refusing is right: the
alternative is writing marker text into a file. The old refusal message told
the model to re-emit the call "with the real content written out in full",
which is advice it cannot follow, because the content it is missing is the
content that was masked away.

Two things make the refusal recoverable instead of terminal.

First, a name for what went wrong. A hash absent from the store is almost
never an expired entry (capacity is a whole session and the TTL is a day). It
is a hash that was retyped rather than copied, one character off.

Second, the near miss itself. Every stored key is the same 24 hex characters,
so a one-character slip shows up as a Hamming distance of one against exactly
one stored key, and a dropped tail shows up as a prefix. Both are cheap to
look for and both point at the entry the caller meant. The suggestion goes in
the message rather than being applied silently: expanding the wrong entry
would put the wrong bytes into a file edit, which is the failure the guard
exists to prevent.

Third, the refusal has to leave the turn able to move. A refusal that ends the
turn stops the client's agent loop and waits for a human to type something,
which turns a mistyped hash into an interruption. Nothing here needs a person,
so the guard keeps whatever tool calls were clean, and when the blocked call
was the only one, the caller hands ``RETRY_PROMPT`` plus the message back to
the model itself.
"""

from __future__ import annotations

# A slip of one or two characters is a transcription error worth pointing at.
# Beyond that the candidate is a different hash that happens to look similar,
# and naming it would send the caller after the wrong content.
MAX_SUGGESTION_DISTANCE = 2
MAX_SUGGESTIONS = 3

# Sent as the user turn of the self-heal continuation, after the refusal itself
# has been replayed as the assistant turn. It says what state the world is in
# (nothing ran) so the model does not re-issue the call defensively, and it
# asks for one concrete next step rather than an apology.
RETRY_PROMPT = (
    "That tool call was blocked before it ran, so nothing was written and "
    "nothing changed on disk. Recover the content the way the message above "
    "describes, then continue the work."
)


def _hamming(left: str, right: str) -> int | None:
    """Substitution distance, or None when the lengths differ."""
    if len(left) != len(right):
        return None
    return sum(1 for a, b in zip(left, right) if a != b)


def nearest_stored_hashes(
    hash_key: str,
    keys: list[str],
    *,
    max_distance: int = MAX_SUGGESTION_DISTANCE,
    limit: int = MAX_SUGGESTIONS,
) -> list[str]:
    """Stored keys close enough to be what the caller meant to write.

    Two shapes are checked, because two slips account for what goes wrong:
    a wrong character (same length, small Hamming distance) and a truncated
    or over-copied hash (one string is a prefix of the other). Both run in a
    single pass over the keys and only on the refusal path, so the cost lands
    where a request is already being turned away.
    """
    if not hash_key or hash_key == "unknown":
        return []
    needle = hash_key.lower()
    scored: list[tuple[int, str]] = []
    for key in keys:
        candidate = key.lower()
        if candidate == needle:
            # Present after all. Nothing to suggest, and the caller's problem
            # is elsewhere (an expired entry, say), so say nothing rather than
            # suggesting the hash back to itself.
            return []
        distance = _hamming(needle, candidate)
        if distance is None:
            shorter, longer = sorted((needle, candidate), key=len)
            if longer.startswith(shorter):
                distance = len(longer) - len(shorter)
            else:
                continue
        if distance <= max_distance:
            scored.append((distance, key))
    scored.sort(key=lambda pair: (pair[0], pair[1]))
    return [key for _, key in scored[:limit]]


def blocked_message(hash_key: str, *, suggestions: list[str] | None = None) -> str:
    """The refusal the caller reads, with the recovery that actually works."""
    if hash_key == "unknown":
        opening = (
            "Headroom blocked this tool call: its input contained recovery marker "
            "text with no readable hash."
        )
    else:
        opening = (
            "Headroom blocked this tool call: its input contained a recovery "
            f"marker with unknown hash {hash_key}."
        )
    parts = [opening]
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"The closest stored hash is {suggestions[0]}.")
        else:
            parts.append("The closest stored hashes are " + ", ".join(suggestions) + ".")
        parts.append(
            "A hash one character off is a retyped hash. Copy it from the marker "
            "instead of writing it out."
        )
    if suggestions:
        # Retrieval can only work against a hash the store actually holds, and
        # the suggestions are the ones it holds. Point at those, never at the
        # hash that just missed.
        parts.append(
            "To recover: call the headroom_retrieve tool with one of those stored "
            "hashes, or re-read the file or re-run the command that produced the "
            "content, since disk is the source of truth."
        )
    else:
        # No near miss means the store does not hold anything resembling this
        # hash, so retrieval cannot resolve it either. Saying otherwise sends
        # the caller into a retry loop against a call that must fail, which is
        # the whole reason this branch exists.
        parts.append(
            "Nothing in the store resembles that hash, so retrieval cannot "
            "resolve it. Re-read the file or re-run the command that produced "
            "the content, since disk is the source of truth."
        )
    parts.append("Do not put marker text into a tool input.")
    return " ".join(parts)
