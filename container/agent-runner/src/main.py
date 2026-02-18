# start container/agent-runner/src/main.py
"""NanoClaw Python agent runner.

Runs inside an Apple Container / Docker container.
Reads ContainerInput from stdin, runs Claude agent SDK,
emits ContainerOutput to stdout wrapped in sentinel markers.

Input protocol:
    Stdin: Full ContainerInput JSON (read until EOF)
    IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
           Files: {type:"message", text:"..."}.json — polled and consumed
           Sentinel: /workspace/ipc/input/_close — signals session end

Stdout protocol:
    Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
    Multiple results may be emitted (one per agent turn result).
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
from claude_agent_sdk.types import (
    PreCompactHookInput,
    PreToolUseHookInput,
    ResultMessage,
    SystemMessage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPC_INPUT_DIR = "/workspace/ipc/input"
IPC_INPUT_CLOSE_SENTINEL = "/workspace/ipc/input/_close"
IPC_POLL_INTERVAL = 0.5  # seconds

OUTPUT_START_MARKER = "---NANOCLAW_OUTPUT_START---"
OUTPUT_END_MARKER = "---NANOCLAW_OUTPUT_END---"

SECRET_ENV_VARS = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"]

ALLOWED_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "Task",
    "TaskOutput",
    "TaskStop",
    "TeamCreate",
    "TeamDelete",
    "SendMessage",
    "TodoWrite",
    "ToolSearch",
    "Skill",
    "NotebookEdit",
    "mcp__nanoclaw__*",
]


# ---------------------------------------------------------------------------
# Logging / output helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """Write a prefixed log message to stderr.

    Args:
        msg: Human-readable message to log.
    """
    print(f"[agent-runner] {msg}", file=sys.stderr, flush=True)


def write_output(output_dict: dict) -> None:
    """Write a ContainerOutput dict to stdout wrapped in sentinel markers.

    Flushes stdout immediately so the host process can detect the boundary.

    Args:
        output_dict: Dictionary conforming to ContainerOutput shape.
    """
    print(OUTPUT_START_MARKER, flush=True)
    print(json.dumps(output_dict), flush=True)
    print(OUTPUT_END_MARKER, flush=True)


# ---------------------------------------------------------------------------
# Stdin / ContainerInput
# ---------------------------------------------------------------------------


def read_stdin() -> dict:
    """Read all stdin and parse as JSON.

    Returns:
        Parsed ContainerInput as a plain dict.

    Raises:
        SystemExit: On JSON parse failure.
    """
    raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        write_output(
            {
                "status": "error",
                "result": None,
                "new_session_id": None,
                "error": f"Failed to parse stdin JSON: {exc}",
            }
        )
        sys.exit(1)


def build_sdk_env(secrets: dict[str, str] | None) -> dict[str, str]:
    """Build the environment dict to pass to the SDK.

    Merges ``os.environ`` with any secrets from ContainerInput.
    Secrets are kept separate from ``os.environ`` so Bash subprocesses
    (which inherit ``os.environ``) never see them; only the SDK CLI
    subprocess receives them.

    Args:
        secrets: Optional mapping of secret env var names to values.

    Returns:
        Full environment dict for the SDK process.
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if v is not None}
    if secrets:
        env.update(secrets)
    return env


# ---------------------------------------------------------------------------
# IPC helpers
# ---------------------------------------------------------------------------


def should_close() -> bool:
    """Check whether the _close sentinel file exists and remove it.

    Returns:
        True if the sentinel was present (and has been deleted), False otherwise.
    """
    sentinel = Path(IPC_INPUT_CLOSE_SENTINEL)
    if sentinel.exists():
        try:
            sentinel.unlink()
        except OSError:
            pass
        return True
    return False


def drain_ipc_input() -> list[str]:
    """Scan IPC input directory for pending .json message files.

    Reads, parses, and deletes each file found.  Files must contain a JSON
    object with ``{"type": "message", "text": "..."}``.

    Returns:
        List of message text strings found, in filename-sorted order.
    """
    ipc_dir = Path(IPC_INPUT_DIR)
    ipc_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    try:
        files = sorted(p for p in ipc_dir.iterdir() if p.suffix == ".json")
    except OSError as exc:
        log(f"IPC drain error listing dir: {exc}")
        return messages

    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            file_path.unlink()
            if data.get("type") == "message" and data.get("text"):
                messages.append(data["text"])
        except (OSError, json.JSONDecodeError) as exc:
            log(f"Failed to process input file {file_path.name}: {exc}")
            try:
                file_path.unlink()
            except OSError:
                pass

    return messages


async def wait_for_ipc_message() -> str | None:
    """Async-poll IPC directory until a new message arrives or _close fires.

    Returns:
        Concatenated message text (newline-separated if multiple), or None
        if the _close sentinel was detected.
    """
    while True:
        if should_close():
            return None
        messages = drain_ipc_input()
        if messages:
            return "\n".join(messages)
        await asyncio.sleep(IPC_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Transcript helpers (for PreCompact hook)
# ---------------------------------------------------------------------------


def _get_session_summary(session_id: str, transcript_path: str) -> str | None:
    """Look up the human-readable summary for a session in sessions-index.json.

    Args:
        session_id: Claude session ID to search for.
        transcript_path: Path to the current transcript file.  The sessions
            index is expected to live in the same directory.

    Returns:
        Summary string if found, otherwise None.
    """
    index_path = Path(transcript_path).parent / "sessions-index.json"
    if not index_path.exists():
        log(f"Sessions index not found at {index_path}")
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for entry in index.get("entries", []):
            if entry.get("sessionId") == session_id and entry.get("summary"):
                return entry["summary"]
    except (OSError, json.JSONDecodeError) as exc:
        log(f"Failed to read sessions index: {exc}")
    return None


def _sanitize_filename(summary: str) -> str:
    """Convert a session summary into a filesystem-safe slug.

    Args:
        summary: Raw summary string.

    Returns:
        Lowercase alphanumeric slug, max 50 chars.
    """
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", summary.lower())
    slug = slug.strip("-")
    return slug[:50]


def _generate_fallback_name() -> str:
    """Generate a fallback filename component when no session summary exists.

    Returns:
        String like ``conversation-1430``.
    """
    now = datetime.now()
    return f"conversation-{now.hour:02d}{now.minute:02d}"


def _parse_transcript(content: str) -> list[dict[str, str]]:
    """Parse a JSONL transcript into a list of ``{role, content}`` dicts.

    Each line is expected to be a JSON object with one of:
    - ``{"type": "user", "message": {"content": <str | list>}}``
    - ``{"type": "assistant", "message": {"content": [{type, text}, ...]}}``

    Args:
        content: Raw JSONL transcript text.

    Returns:
        List of ``{"role": "user"|"assistant", "content": str}`` dicts.
    """
    messages: list[dict[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if entry.get("type") == "user":
            raw = entry.get("message", {}).get("content", "")
            if isinstance(raw, list):
                text = "".join(block.get("text", "") for block in raw if isinstance(block, dict))
            else:
                text = str(raw) if raw else ""
            if text:
                messages.append({"role": "user", "content": text})

        elif entry.get("type") == "assistant":
            blocks = entry.get("message", {}).get("content", [])
            if isinstance(blocks, list):
                text = "".join(
                    b.get("text", "")
                    for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                messages.append({"role": "assistant", "content": text})

    return messages


def _format_transcript_markdown(
    messages: list[dict[str, str]], title: str | None = None
) -> str:
    """Render parsed transcript messages as a Markdown document.

    Args:
        messages: List of ``{role, content}`` dicts from :func:`_parse_transcript`.
        title: Optional conversation title for the heading.

    Returns:
        Markdown-formatted string.
    """
    now = datetime.now()
    date_str = now.strftime("%b %-d, %I:%M %p")

    lines: list[str] = [
        f"# {title or 'Conversation'}",
        "",
        f"Archived: {date_str}",
        "",
        "---",
        "",
    ]
    for msg in messages:
        sender = "User" if msg["role"] == "user" else "Andy"
        body = msg["content"]
        if len(body) > 2000:
            body = body[:2000] + "..."
        lines.append(f"**{sender}**: {body}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def create_pre_compact_hook():
    """Create a PreCompact hook that archives the transcript before compaction.

    Returns:
        Async hook callable suitable for use in a :class:`HookMatcher`.
    """

    async def pre_compact_hook(
        input_data: PreCompactHookInput,
        tool_use_id: str | None,
        context,
    ) -> dict:
        """Archive transcript to /workspace/group/conversations/ before compact.

        Args:
            input_data: Hook input with ``transcript_path`` and ``session_id``.
            tool_use_id: Unused for PreCompact events.
            context: Hook execution context.

        Returns:
            Empty dict (no modifications to the compact operation).
        """
        transcript_path: str = input_data.get("transcript_path", "")
        session_id: str = input_data.get("session_id", "")

        if not transcript_path or not Path(transcript_path).exists():
            log("No transcript found for archiving")
            return {}

        try:
            content = Path(transcript_path).read_text(encoding="utf-8")
            messages = _parse_transcript(content)
            if not messages:
                log("No messages to archive")
                return {}

            summary = _get_session_summary(session_id, transcript_path)
            name = _sanitize_filename(summary) if summary else _generate_fallback_name()

            conversations_dir = Path("/workspace/group/conversations")
            conversations_dir.mkdir(parents=True, exist_ok=True)

            date_prefix = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_prefix}-{name}.md"
            file_path = conversations_dir / filename

            markdown = _format_transcript_markdown(messages, summary)
            file_path.write_text(markdown, encoding="utf-8")
            log(f"Archived conversation to {file_path}")
        except Exception as exc:  # noqa: BLE001
            log(f"Failed to archive transcript: {exc}")

        return {}

    return pre_compact_hook


def create_sanitize_bash_hook():
    """Create a PreToolUse/Bash hook that strips secret env vars from commands.

    Prepends ``unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN 2>/dev/null;``
    to every Bash command so that subprocesses spawned by Claude Code cannot
    read API credentials from the environment.

    Returns:
        Async hook callable suitable for use in a :class:`HookMatcher`.
    """
    unset_prefix = f"unset {' '.join(SECRET_ENV_VARS)} 2>/dev/null; "

    async def sanitize_bash_hook(
        input_data: PreToolUseHookInput,
        tool_use_id: str | None,
        context,
    ) -> dict:
        """Strip secret env vars from a Bash tool command.

        Args:
            input_data: Hook input with ``tool_input`` containing ``command``.
            tool_use_id: Tool use identifier.
            context: Hook execution context.

        Returns:
            Hook-specific output updating the bash command, or empty dict if
            no command is present.
        """
        tool_input: dict = input_data.get("tool_input", {})
        command: str | None = tool_input.get("command")
        if not command:
            return {}

        new_command = unset_prefix + command
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "updatedInput": {**tool_input, "command": new_command},
            }
        }

    return sanitize_bash_hook


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


def _load_global_claude_md(is_main: bool) -> str | None:
    """Load the global CLAUDE.md shared across all non-main groups.

    Args:
        is_main: If True, global CLAUDE.md is not loaded (main group manages
            its own global context).

    Returns:
        Content of /workspace/global/CLAUDE.md or None.
    """
    if is_main:
        return None
    global_md_path = Path("/workspace/global/CLAUDE.md")
    if global_md_path.exists():
        try:
            return global_md_path.read_text(encoding="utf-8")
        except OSError as exc:
            log(f"Failed to read global CLAUDE.md: {exc}")
    return None


def _discover_extra_dirs() -> list[str]:
    """Scan /workspace/extra for additional CLAUDE.md directories.

    Returns:
        Sorted list of absolute paths to subdirectories of /workspace/extra/.
    """
    extra_base = Path("/workspace/extra")
    if not extra_base.exists():
        return []
    dirs = sorted(
        str(p) for p in extra_base.iterdir() if p.is_dir()
    )
    if dirs:
        log(f"Additional directories: {', '.join(dirs)}")
    return dirs


async def run_agent(
    prompt: str,
    session_id: str | None,
    container_input: dict,
    sdk_env: dict[str, str],
) -> tuple[str | None, bool]:
    """Run a single agent turn using :class:`ClaudeSDKClient`.

    Opens a streaming connection, sends ``prompt``, polls IPC for follow-up
    messages during the agent turn, and emits each :class:`ResultMessage` via
    :func:`write_output`.

    Args:
        prompt: Initial user prompt text.
        session_id: Optional Claude session ID to resume, or None to start fresh.
        container_input: Parsed ContainerInput dict.
        sdk_env: Full environment dict for the SDK subprocess.

    Returns:
        A ``(new_session_id, closed)`` tuple where ``new_session_id`` is the
        session ID captured from the init message (or None if not seen), and
        ``closed`` is True if the _close sentinel was consumed during this turn.
    """
    global_md = _load_global_claude_md(container_input.get("is_main", False))
    extra_dirs = _discover_extra_dirs()

    system_prompt: dict | None = None
    if global_md:
        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": global_md,
        }

    options = ClaudeAgentOptions(
        cwd="/workspace/group",
        allowed_tools=ALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        resume=session_id,
        env=sdk_env,
        setting_sources=["project", "user"],
        mcp_servers={
            "nanoclaw": {
                "command": "python",
                "args": ["/app/src/ipc_mcp_stdio.py"],
                "env": {
                    "NANOCLAW_CHAT_JID": container_input.get("chat_jid", ""),
                    "NANOCLAW_GROUP_FOLDER": container_input.get("group_folder", ""),
                    "NANOCLAW_IS_MAIN": "1" if container_input.get("is_main") else "0",
                },
            }
        },
        add_dirs=extra_dirs,
        system_prompt=system_prompt,
        hooks={
            "PreCompact": [HookMatcher(hooks=[create_pre_compact_hook()])],
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[create_sanitize_bash_hook()])
            ],
        },
    )

    new_session_id: str | None = None
    message_count = 0
    result_count = 0
    closed_during_run = False

    # Use __aenter__ (connects with empty interactive stream), then send the
    # initial prompt via client.query().  Do NOT call connect() again — that
    # would attempt a second subprocess launch.
    async with ClaudeSDKClient(options=options) as client:
        # Send the initial prompt into the already-open stream.
        await client.query(prompt)

        # Concurrently poll IPC for follow-up messages and read agent messages.
        ipc_polling_active = True

        async def poll_ipc() -> None:
            """Deliver IPC follow-ups to the client and detect close sentinel."""
            nonlocal closed_during_run, ipc_polling_active
            while ipc_polling_active:
                await asyncio.sleep(IPC_POLL_INTERVAL)
                if not ipc_polling_active:
                    break
                if should_close():
                    log("Close sentinel detected during agent run, disconnecting")
                    closed_during_run = True
                    await client.disconnect()
                    return
                messages = drain_ipc_input()
                for text in messages:
                    log(f"Piping IPC message into active run ({len(text)} chars)")
                    await client.query(text)

        ipc_task = asyncio.create_task(poll_ipc())

        try:
            async for message in client.receive_messages():
                message_count += 1
                msg_type = getattr(message, "type", type(message).__name__)
                log(f"[msg #{message_count}] type={msg_type}")

                if isinstance(message, SystemMessage):
                    subtype = message.subtype
                    if subtype == "init":
                        new_session_id = message.data.get("session_id")
                        log(f"Session initialized: {new_session_id}")
                    elif subtype == "task_notification":
                        task_id = message.data.get("task_id", "?")
                        status = message.data.get("status", "?")
                        summary = message.data.get("summary", "")
                        log(f"Task notification: task={task_id} status={status} summary={summary}")

                elif isinstance(message, ResultMessage):
                    result_count += 1
                    text_result = message.result
                    log(
                        f"Result #{result_count}: subtype={message.subtype}"
                        + (f" text={text_result[:200]!r}" if text_result else "")
                    )
                    write_output(
                        {
                            "status": "success",
                            "result": text_result,
                            "new_session_id": new_session_id,
                            "error": None,
                        }
                    )
                    # ResultMessage signals end of this turn; stop iterating.
                    break

        except Exception as exc:  # noqa: BLE001
            log(f"Error receiving agent messages: {exc}")
            raise
        finally:
            ipc_polling_active = False
            ipc_task.cancel()
            try:
                await ipc_task
            except (asyncio.CancelledError, Exception):
                pass

    log(
        f"Agent run done. Messages: {message_count}, results: {result_count}, "
        f"new_session_id: {new_session_id or 'none'}, "
        f"closed_during_run: {closed_during_run}"
    )
    return new_session_id, closed_during_run


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------


async def main() -> None:
    """Main entry point for the NanoClaw Python agent runner.

    Reads ContainerInput from stdin, runs the agent loop, and exits cleanly.
    The loop runs until the _close sentinel is received or a fatal error occurs.
    """
    # --- Parse stdin ---
    container_input = read_stdin()

    # Remove temp input file (may have been written by the container entrypoint
    # and contains secrets — delete ASAP).
    try:
        Path("/tmp/input.json").unlink()
    except OSError:
        pass

    log(f"Received input for group: {container_input.get('group_folder', '?')}")

    # --- Build SDK environment ---
    sdk_env = build_sdk_env(container_input.get("secrets"))

    # --- Set up IPC directory ---
    Path(IPC_INPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Clean up any stale _close sentinel from a previous container run.
    try:
        Path(IPC_INPUT_CLOSE_SENTINEL).unlink()
    except OSError:
        pass

    # --- Build initial prompt ---
    session_id: str | None = container_input.get("session_id")
    prompt: str = container_input.get("prompt", "")

    if container_input.get("is_scheduled_task"):
        prompt = (
            "[SCHEDULED TASK - The following message was sent automatically "
            "and is not coming directly from the user or group.]\n\n" + prompt
        )

    # Drain any IPC messages that arrived before this container started.
    pending = drain_ipc_input()
    if pending:
        log(f"Draining {len(pending)} pending IPC message(s) into initial prompt")
        prompt += "\n" + "\n".join(pending)

    # --- Query loop ---
    try:
        while True:
            log(f"Starting agent run (session: {session_id or 'new'})...")

            new_session_id, closed = await run_agent(
                prompt, session_id, container_input, sdk_env
            )
            if new_session_id:
                session_id = new_session_id

            if closed:
                log("Close sentinel consumed during agent run, exiting")
                break

            # Emit a session-update marker so the host can track the session ID.
            write_output(
                {
                    "status": "success",
                    "result": None,
                    "new_session_id": session_id,
                    "error": None,
                }
            )

            log("Agent run complete, waiting for next IPC message...")

            next_msg = await wait_for_ipc_message()
            if next_msg is None:
                log("Close sentinel received, exiting")
                break

            log(f"Got new message ({len(next_msg)} chars), starting new agent run")
            prompt = next_msg

    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        log(f"Fatal agent error: {error_message}")
        write_output(
            {
                "status": "error",
                "result": None,
                "new_session_id": session_id,
                "error": error_message,
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
# end container/agent-runner/src/main.py
