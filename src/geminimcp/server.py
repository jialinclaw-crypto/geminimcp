"""FastMCP server implementation for the Gemini MCP project.

Patched by Claude — fixes: stderr separation, total timeout, error detection,
--prompt deprecation, retry logic, and improved error reporting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import shutil
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

log = logging.getLogger(__name__)

mcp = FastMCP("Gemini MCP Server-from guda.studio")

# Configurable via environment variable
DEFAULT_TIMEOUT = float(os.environ.get("GEMINIMCP_TIMEOUT", "120"))

# Module-level stderr capture (safe: MCP stdio is serial, one call at a time)
_last_stderr: list[str] = []


class GeminiTimeout(Exception):
    """Raised when the Gemini CLI exceeds the total timeout."""


# ---------------------------------------------------------------------------
# Subprocess runner with timeout and stderr separation
# ---------------------------------------------------------------------------

def run_shell_command(
    cmd: list[str],
    cwd: str | None = None,
    total_timeout: float = DEFAULT_TIMEOUT,
) -> Generator[str, None, None]:
    """Execute a command and stream its stdout line-by-line.

    stderr is captured separately into module-level `_last_stderr`.
    Raises GeminiTimeout if the process does not finish within total_timeout.
    """
    global _last_stderr
    _last_stderr = []

    popen_cmd = list(cmd)  # copy to avoid mutating caller's list
    gemini_path = shutil.which("gemini") or popen_cmd[0]
    popen_cmd[0] = gemini_path

    process = subprocess.Popen(
        popen_cmd,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # FIX: separate stderr from stdout
        universal_newlines=True,
        encoding="utf-8",
        cwd=cwd,
    )

    stdout_queue: queue.Queue[str | None] = queue.Queue()
    GRACEFUL_SHUTDOWN_DELAY = 0.3
    deadline = time.monotonic() + total_timeout

    def is_turn_completed(line: str) -> bool:
        try:
            data = json.loads(line)
            return data.get("type") == "turn.completed"
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    def read_stdout() -> None:
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                stripped = line.strip()
                if not stripped:
                    continue  # skip empty lines
                stdout_queue.put(stripped)
                if is_turn_completed(stripped):
                    time.sleep(GRACEFUL_SHUTDOWN_DELAY)
                    process.terminate()
                    break
            process.stdout.close()
        stdout_queue.put(None)

    def read_stderr() -> None:
        if process.stderr:
            for line in iter(process.stderr.readline, ""):
                stripped = line.strip()
                if stripped:
                    _last_stderr.append(stripped)
                    log.debug("gemini_stderr: %s", stripped[:200])
            process.stderr.close()

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Yield lines with timeout check
    while True:
        if time.monotonic() > deadline:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            raise GeminiTimeout(
                f"Gemini CLI did not complete within {total_timeout:.0f}s"
            )

        try:
            line = stdout_queue.get(timeout=0.5)
            if line is None:
                break
            yield line
        except queue.Empty:
            if process.poll() is not None and not stdout_thread.is_alive():
                break

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    # Drain remaining items
    while not stdout_queue.empty():
        try:
            line = stdout_queue.get_nowait()
            if line is not None:
                yield line
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# Error classification for retry logic
# ---------------------------------------------------------------------------

def _classify_error(err_message: str, stderr_lines: list[str]) -> str:
    """Classify error as 'transient', 'auth', or 'fatal'."""
    combined = (err_message + " " + " ".join(stderr_lines)).lower()

    # Transient: rate limits, temporary outages
    transient_patterns = [
        "429", "resource_exhausted", "rate limit", "temporarily unavailable",
        "connection reset", "connection refused", "quota", "capacity",
    ]
    if any(p in combined for p in transient_patterns):
        return "transient"

    # Auth: token/credential issues
    auth_patterns = [
        "token", "oauth", "refresh", "unauthorized", "403", "credential",
        "authentication", "login",
    ]
    if any(p in combined for p in auth_patterns):
        return "auth"

    return "fatal"


def _format_stderr_tail(lines: list[str], max_lines: int = 10) -> str:
    """Format the last N stderr lines for error reporting."""
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n[stderr]\n" + "\n".join(tail)


# ---------------------------------------------------------------------------
# Single execution attempt
# ---------------------------------------------------------------------------

def _execute_once(
    cmd: list[str],
    cwd: str,
    return_all_messages: bool,
    total_timeout: float,
) -> Dict[str, Any]:
    """Run Gemini CLI once and return the result dict."""
    global _last_stderr

    all_messages: list[dict] = []
    agent_messages = ""
    success = True
    err_message = ""
    thread_id: Optional[str] = None

    try:
        for line in run_shell_command(cmd, cwd=cwd, total_timeout=total_timeout):
            try:
                line_dict = json.loads(line)
            except json.JSONDecodeError:
                err_message += f"\n[json decode error] {line[:200]}"
                continue

            all_messages.append(line_dict)
            item_type = line_dict.get("type", "")
            item_role = line_dict.get("role", "")

            # Capture assistant messages
            if item_type == "message" and item_role == "assistant":
                content = line_dict.get("content", "")
                if content:
                    agent_messages += content

            # Capture session ID
            if line_dict.get("session_id") is not None:
                thread_id = line_dict.get("session_id")

            # FIX: Re-enable error/fail event detection (was commented out)
            if "fail" in item_type or "error" in item_type:
                success = False
                error_obj = line_dict.get("error", {})
                if isinstance(error_obj, dict):
                    err_detail = error_obj.get("message", "")
                else:
                    err_detail = str(error_obj)
                err_detail = err_detail or line_dict.get("message", "")
                err_message += f"\n[gemini {item_type}] {err_detail}"
                # Do NOT break — continue reading to capture session_id

    except GeminiTimeout as e:
        stderr_tail = _format_stderr_tail(_last_stderr)
        return {
            "success": False,
            "error": (
                f"{e}\n\n"
                f"Possible causes: OAuth token refresh stall, "
                f"rate limit (429) retry loop, or network issue.\n"
                f"{stderr_tail}\n\n"
                f"[partial output] {agent_messages[:500]}"
            ),
            "SESSION_ID": thread_id or "",
        }

    # Post-loop validation
    if thread_id is None:
        success = False
        stderr_tail = _format_stderr_tail(_last_stderr)
        err_message = (
            "Failed to get SESSION_ID from the Gemini session."
            + stderr_tail + "\n" + err_message
        )

    if success and not agent_messages:
        success = False
        err_message = (
            "No agent_messages received. "
            "This might be due to Gemini performing a tool call. "
            "You can continue using the SESSION_ID.\n" + err_message
        )

    if success:
        result: Dict[str, Any] = {
            "success": True,
            "SESSION_ID": thread_id,
            "agent_messages": agent_messages,
        }
    else:
        result = {
            "success": False,
            "error": err_message,
            "SESSION_ID": thread_id or "",
        }

    if return_all_messages:
        result["all_messages"] = all_messages

    return result


# ---------------------------------------------------------------------------
# MCP tool entry point with retry
# ---------------------------------------------------------------------------

MAX_RETRIES = 2
RETRY_DELAYS = [5, 15]


@mcp.tool(
    name="gemini",
    description="""
    Invokes the Gemini CLI to execute AI-driven tasks, returning structured JSON events and a session identifier for conversation continuity.

    **Return structure:**
        - `success`: boolean indicating execution status
        - `SESSION_ID`: unique identifier for resuming this conversation in future calls
        - `agent_messages`: concatenated assistant response text
        - `all_messages`: (optional) complete array of JSON events when `return_all_messages=True`
        - `error`: error description when `success=False`

    **Best practices:**
        - Always capture and reuse `SESSION_ID` for multi-turn interactions
        - Enable `sandbox` mode when file modifications should be isolated
        - Use `return_all_messages` only when detailed execution traces are necessary (increases payload size)
        - Only pass `model` when the user has explicitly requested a specific model
    """,
    meta={"version": "1.0.0", "author": "guda.studio (patched)"},
)
async def gemini(
    PROMPT: Annotated[str, "Instruction for the task to send to gemini."],
    cd: Annotated[Path, "Set the workspace root for gemini before executing the task."],
    sandbox: Annotated[
        bool,
        Field(description="Run in sandbox mode. Defaults to `False`."),
    ] = False,
    SESSION_ID: Annotated[
        str,
        "Resume the specified session of the gemini. Defaults to empty string, start a new session.",
    ] = "",
    return_all_messages: Annotated[
        bool,
        "Return all messages from the gemini session. Defaults to False.",
    ] = False,
    model: Annotated[
        str,
        "The model to use. Strictly prohibited unless explicitly specified by the user.",
    ] = "",
) -> Dict[str, Any]:
    """Execute a gemini CLI session and return the results."""

    if not cd.exists():
        return {
            "success": False,
            "error": f"Directory `{cd.absolute().as_posix()}` does not exist.",
        }

    # FIX: Use positional argument instead of deprecated --prompt
    cmd = ["gemini", "-o", "stream-json", "-y"]

    if sandbox:
        cmd.append("--sandbox")
    if model:
        cmd.extend(["--model", model])
    if SESSION_ID:
        cmd.extend(["--resume", SESSION_ID])

    # Positional prompt MUST come last, after -- separator
    prompt = PROMPT
    if os.name == "nt":
        prompt = windows_escape(prompt)
    cmd.extend(["--", prompt])

    cwd = cd.absolute().as_posix()

    # Retry loop for transient errors (429, network)
    for attempt in range(MAX_RETRIES + 1):
        result = _execute_once(cmd, cwd, return_all_messages, DEFAULT_TIMEOUT)

        if result["success"]:
            return result

        error_class = _classify_error(
            result.get("error", ""), _last_stderr
        )

        if error_class == "transient" and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt]
            log.warning(
                "Transient error (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, MAX_RETRIES, delay,
                result.get("error", "")[:200],
            )
            await asyncio.sleep(delay)
            continue

        if error_class == "auth":
            result["error"] += (
                "\n\n[hint] OAuth token may be expired. "
                "Run `gemini` in terminal to trigger interactive refresh, "
                "then retry."
            )

        return result

    return result  # should not reach here, but just in case


# Keep windows_escape for Windows compatibility
def windows_escape(prompt: str) -> str:
    result = prompt.replace("\\", "\\\\")
    result = result.replace('"', '\\"')
    result = result.replace("\n", "\\n")
    result = result.replace("\r", "\\r")
    result = result.replace("\t", "\\t")
    result = result.replace("\b", "\\b")
    result = result.replace("\f", "\\f")
    result = result.replace("'", "\\'")
    return result


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
