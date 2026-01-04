# -*- coding: utf-8 -*-
import os
import json
import logging
import concurrent.futures
import re
import subprocess
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import sys
import platform

from flask import Flask, render_template, request, jsonify, session
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
    import psutil  # type: ignore
except Exception:
    psutil = None
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

if load_dotenv:
    load_dotenv()

ROOT = Path(__file__).resolve().parent
ISSUES_PATH = ROOT / "issues.json"
EXEC_LOG_PATH = ROOT / "execution_log.json"

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
app.secret_key = os.environ.get("AI_TS_SESSION_SECRET", "dev-ai-troubleshooter")

TECH_KEYWORDS = {
    "screen", "flickering", "display", "wifi", "internet", "connection", "network", "boot", "startup", "slow",
    "performance", "blue", "bsod", "crash", "error", "issue", "problem", "windows", "computer", "system", "file",
    "disk", "memory", "cpu", "process", "service", "audio", "sound", "speaker", "headphone", "usb", "port",
    "printer", "keyboard", "mouse", "touchpad", "update", "permission", "access", "registry", "search", "taskbar",
    "menu", "defender", "firewall", "cache", "clear", "clean", "temporary", "temp"
}

_GEMINI_MODEL = None
_GEMINI_MODEL_NAME = None

if os.environ.get("AI_TS_DEBUG_GEMINI", "").strip().lower() in {"1", "true", "yes", "on"}:
    app.logger.setLevel(logging.INFO)


def is_admin() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def load_issues() -> List[dict]:
    try:
        mtime = ISSUES_PATH.stat().st_mtime
    except FileNotFoundError:
        return []
    cached = _ISSUES_CACHE.get("data")
    if cached is not None and _ISSUES_CACHE.get("mtime") == mtime:
        return cached  # type: ignore[return-value]
    try:
        with open(ISSUES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        app.logger.error("ISSUES: failed to load issues.json: %s", exc)
        return cached if cached is not None else []
    _ISSUES_CACHE["mtime"] = mtime
    _ISSUES_CACHE["data"] = data
    return data


def _extract_first_int(text: str, max_value: int) -> int | None:
    if not text:
        return None
    for match in re.finditer(r"\b(\d{1,3})\b", text):
        try:
            value = int(match.group(1))
        except Exception:
            continue
        if 1 <= value <= max_value:
            return value
    return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def _hash_texts(texts: List[str]) -> str:
    payload = "\n".join(texts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _allow_tfidf_fallback() -> bool:
    return _env_flag("AI_TS_ALLOW_TFIDF_FALLBACK", default=True)


def _get_gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _gemini_configured() -> bool:
    return genai is not None and bool(_get_gemini_api_key())


def _gemini_debug_enabled() -> bool:
    return _env_flag("AI_TS_DEBUG_GEMINI", default=False)


def _log_gemini_debug(message: str, payload: object | None = None) -> None:
    if not _gemini_debug_enabled():
        return
    if payload is None:
        app.logger.info("Gemini debug: %s", message)
        return
    if isinstance(payload, str):
        app.logger.info("Gemini debug: %s\n%s", message, payload)
        return
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=True)
    except Exception:
        text = str(payload)
    app.logger.info("Gemini debug: %s\n%s", message, text)


def _gemini_timeout_seconds() -> float:
    raw = os.environ.get("GEMINI_TIMEOUT_SECONDS", "20").strip()
    try:
        value = float(raw)
    except Exception:
        value = 20.0
    if value <= 0:
        value = 20.0
    return value


def _format_solution_lines(solutions: List[dict], default_source: str | None = None) -> List[str]:
    lines = []
    for idx, sol in enumerate(solutions, 1):
        cmd = (sol.get("command") or "").strip()
        desc = (sol.get("description") or "").strip()
        source = sol.get("source") or default_source
        prefix = f"[{source}] " if source else ""
        if cmd and desc:
            lines.append(f"{idx}. {prefix}{cmd} | {desc}")
        elif cmd:
            lines.append(f"{idx}. {prefix}{cmd}")
        elif desc:
            lines.append(f"{idx}. {prefix}{desc}")
    return lines


class _TTLCache:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._data: dict[tuple, tuple[float, object]] = {}

    def get(self, key: tuple) -> object | None:
        now = time.monotonic()
        entry = self._data.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at <= now:
            del self._data[key]
            return None
        return value

    def set(self, key: tuple, value: object) -> None:
        now = time.monotonic()
        self._data[key] = (now + self._ttl_seconds, value)
        if len(self._data) > self._max_entries:
            self._prune(now)

    def _prune(self, now: float) -> None:
        expired_keys = [key for key, (expiry, _) in self._data.items() if expiry <= now]
        for key in expired_keys:
            del self._data[key]
        if len(self._data) <= self._max_entries:
            return
        overflow = len(self._data) - self._max_entries
        for key in list(self._data.keys())[:overflow]:
            del self._data[key]


_ISSUES_CACHE: dict[str, object] = {"mtime": None, "data": None}
_GEMINI_CHOOSE_CACHE = _TTLCache(max_entries=256, ttl_seconds=300)
_GEMINI_SUGGEST_CACHE = _TTLCache(max_entries=256, ttl_seconds=300)


def _gemini_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def _choose_cache_key(user_error: str, problems: List[str]) -> tuple:
    return (_gemini_model_name(), _normalize_text(user_error), _hash_texts(problems))


def _suggest_cache_key(user_error: str, problem: str, existing: List[dict]) -> tuple:
    commands = [sol.get("command", "").strip() for sol in existing if sol.get("command")]
    return (
        _gemini_model_name(),
        _normalize_text(user_error),
        _normalize_text(problem),
        _hash_texts(commands),
    )


def _get_ai_mode() -> str:
    return "gemini" if _gemini_configured() else "tfidf"


def _get_gemini_model() -> tuple[object | None, str | None]:
    if genai is None:
        return None, "Gemini client library not installed"
    api_key = _get_gemini_api_key()
    if not api_key:
        return None, "Gemini API key not configured"

    model_name = _gemini_model_name()
    _log_gemini_debug(
        "initialize",
        {"model": model_name, "api_key_configured": True, "client_loaded": True},
    )
    global _GEMINI_MODEL, _GEMINI_MODEL_NAME
    if _GEMINI_MODEL is not None and _GEMINI_MODEL_NAME == model_name:
        return _GEMINI_MODEL, None

    genai.configure(api_key=api_key)
    _GEMINI_MODEL_NAME = model_name
    _GEMINI_MODEL = genai.GenerativeModel(model_name)
    return _GEMINI_MODEL, None


def _get_response_text(response: object, stage: str) -> str:
    try:
        text = response.text or ""
        if text:
            return text
    except Exception as exc:
        _log_gemini_debug(f"{stage} response_text_error", str(exc))

    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reasons = [getattr(cand, "finish_reason", None) for cand in candidates]
            _log_gemini_debug(f"{stage} finish_reasons", reasons)
    except Exception as exc:
        _log_gemini_debug(f"{stage} response_meta_error", str(exc))

    try:
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            part_texts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    part_texts.append(part_text)
            if part_texts:
                return "\n".join(part_texts)
    except Exception as exc:
        _log_gemini_debug(f"{stage} response_parse_error", str(exc))

    _log_gemini_debug(f"{stage} response_repr", repr(response))
    return ""


def _gemini_generate_content(
    model: object,
    prompt: str,
    generation_config: dict,
    stage: str,
) -> Tuple[object | None, str | None]:
    timeout_seconds = _gemini_timeout_seconds()

    def _call(config: dict) -> Tuple[object | None, str | None]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.generate_content, prompt, generation_config=config)
            try:
                return future.result(timeout=timeout_seconds), None
            except concurrent.futures.TimeoutError:
                return None, f"Gemini request timed out after {timeout_seconds}s"
            except Exception as exc:
                return None, str(exc)

    response, error = _call(generation_config)
    if error:
        _log_gemini_debug(f"{stage} request_error", error)
        fallback_config = {
            key: value
            for key, value in generation_config.items()
            if key not in {"response_mime_type", "response_schema"}
        }
        if fallback_config != generation_config:
            response, error = _call(fallback_config)
            if error:
                _log_gemini_debug(f"{stage} request_error", error)
                return None, f"Gemini request failed: {error}"
            return response, None
        return None, f"Gemini request failed: {error}"

    return response, None


def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def collect_system_info() -> dict:
    info: dict = {
        "ram": None,
        "disk": None,
        "battery": None,
        "network": None,
        "wifi": None,
        "processes": [],
    }

    # RAM
    if psutil:
        vm = psutil.virtual_memory()
        info["ram"] = {"percent": vm.percent, "used_gb": round(vm.used / (1024**3), 1), "total_gb": round(vm.total / (1024**3), 1)}
    # Disk
    if psutil:
        root_path = Path.home().anchor or "/"
        du = psutil.disk_usage(root_path)
        info["disk"] = {"percent": du.percent, "used_gb": round(du.used / (1024**3), 1), "total_gb": round(du.total / (1024**3), 1)}
    # Battery
    if psutil and hasattr(psutil, "sensors_battery"):
        batt = psutil.sensors_battery()
        if batt:
            info["battery"] = {"percent": batt.percent, "plugged": bool(batt.power_plugged)}
    # Network/WiFi
    if psutil:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        active = []
        for name, st in stats.items():
            if st.isup:
                ips = [a.address for a in addrs.get(name, []) if getattr(a, "family", None) and a.address]
                active.append({"interface": name, "speed": st.speed, "ips": ips})
        info["network"] = active
    # Wi-Fi SSID (Windows)
    if os.name == "nt":
        wifi_out = _safe_run(["netsh", "wlan", "show", "interfaces"])
        if wifi_out:
            ssid = None
            for line in wifi_out.splitlines():
                if "SSID" in line and "BSSID" not in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        ssid = parts[1].strip()
                        break
            if ssid:
                info["wifi"] = {"ssid": ssid}
    # Top processes
    if psutil:
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(p.info)
            except Exception:
                continue
        procs = sorted(procs, key=lambda x: (x.get("cpu_percent", 0) or 0, x.get("memory_percent", 0) or 0), reverse=True)[:5]
        info["processes"] = procs

    return info


def embed(text: str, vector_size: int = 50) -> np.ndarray:
    import re
    from collections import Counter
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    if not words:
        return np.zeros((1, vector_size))

    word_counts = Counter(words)
    features: List[float] = []
    for keyword in sorted(list(TECH_KEYWORDS))[:25]:
        features.append(1.0 if keyword in word_counts else 0.0)

    total_words = len(words)
    common = word_counts.most_common(15)
    for i in range(15):
        if i < len(common):
            word, count = common[i]
            boost = 2.0 if word in TECH_KEYWORDS else 1.0
            features.append((count / total_words) * boost)
        else:
            features.append(0.0)

    features.extend([
        len(words) / 20.0,
        len(set(words)) / max(len(words), 1),
        sum(1 for w in words if w in TECH_KEYWORDS) / max(len(words), 1),
        1 if any(x in words for x in ["not", "won't", "can't"]) else 0,
        1 if any(x in text for x in ["slow", "hang", "freeze", "lag"]) else 0,
        1 if any(x in text for x in ["error", "fail", "crash", "problem"]) else 0,
        1 if any(x in text for x in ["network", "wifi", "internet", "connection"]) else 0,
        1 if any(x in text for x in ["screen", "display", "monitor", "visual"]) else 0,
        1 if any(x in text for x in ["boot", "start", "startup", "turn"]) else 0,
        1 if any(x in text for x in ["blue", "bsod", "crash", "restart"]) else 0,
        1 if any(x in text for x in ["cache", "clear", "clean", "temporary"]) else 0,
    ])

    features = features[:vector_size]
    while len(features) < vector_size:
        features.append(0.0)

    vec = np.array(features)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


def find_best_match(user_error: str, issues: List[dict]) -> Tuple[dict | None, float]:
    if not user_error.strip():
        return None, 0.0
    user_vec = embed(user_error)
    best_issue, best_score = None, 0.0
    for issue in issues:
        searchable = issue.get("problem", "")
        for sol in issue.get("solutions", []):
            searchable += " " + sol.get("description", "")
        issue_vec = embed(searchable)
        try:
            score = cosine_similarity(user_vec, issue_vec)[0][0]
        except Exception:
            score = 0.0
        if score > best_score:
            best_score, best_issue = score, issue
    return best_issue, best_score


def _score_issue(user_error: str, issue: dict) -> float:
    searchable = issue.get("problem", "")
    for sol in issue.get("solutions", []):
        searchable += " " + sol.get("description", "")
    try:
        score = cosine_similarity(embed(user_error), embed(searchable))[0][0]
        return float(score)
    except Exception:
        return 0.0


def _normalize_command(command: str) -> str:
    return " ".join(command.lower().split())


def _merge_solutions(primary: List[dict], extra: List[dict]) -> List[dict]:
    merged: List[dict] = []
    seen = set()
    for sol in primary + extra:
        cmd = (sol.get("command") or "").strip()
        desc = (sol.get("description") or "").strip()
        if not cmd and not desc:
            continue
        key = _normalize_command(cmd) if cmd else f"desc:{desc.lower()}"
        if key in seen:
            continue
        seen.add(key)
        entry = {"command": cmd, "description": desc}
        source = sol.get("source")
        if source:
            entry["source"] = source
        merged.append(entry)
    return merged


def _gemini_choose_problem(user_error: str, issues: List[dict]) -> Tuple[dict | None, float, str | None]:
    model, err = _get_gemini_model()
    if err:
        _log_gemini_debug("choose_problem model_error", err)
        return None, 0.0, err

    problems = [issue.get("problem", "").strip() for issue in issues if issue.get("problem")]
    if not problems:
        return None, 0.0, "No problem statements available"

    cache_key = _choose_cache_key(user_error, problems)
    cached_index = _GEMINI_CHOOSE_CACHE.get(cache_key)
    if isinstance(cached_index, int) and 1 <= cached_index <= len(problems):
        app.logger.info("ANALYZE: gemini choose cache hit (%s)", cached_index)
        selected_problem = problems[cached_index - 1]
        issue_by_problem = {issue.get("problem"): issue for issue in issues}
        return issue_by_problem.get(selected_problem), 0.0, None

    problem_lines = "\n".join(f"{i + 1}|{p}" for i, p in enumerate(problems))
    prompt = (
        "Return only the number (1-" + str(len(problems)) + ") of the best matching problem.\n"
        "No other text.\n"
        f"Issue: {user_error}\n"
        "Problems:\n"
        f"{problem_lines}\n"
    )
    _log_gemini_debug("choose_problem prompt", prompt)

    response, request_error = _gemini_generate_content(
        model, prompt, {"temperature": 0.0, "max_output_tokens": 8}, "choose_problem"
    )
    if request_error:
        return None, 0.0, request_error

    response_text = _get_response_text(response, "choose_problem")
    if not response_text:
        return None, 0.0, "Gemini response was empty"
    _log_gemini_debug("choose_problem response", response_text)
    problem_index = _extract_first_int(response_text, len(problems))
    if problem_index is None:
        _log_gemini_debug("choose_problem parse_error", response_text)
        return None, 0.0, "Gemini did not return a valid problem index"

    _GEMINI_CHOOSE_CACHE.set(cache_key, problem_index)
    app.logger.info("ANALYZE: gemini problem index: %s", problem_index)
    selected_problem = problems[problem_index - 1]
    confidence_value = 0.0

    issue_by_problem = {issue.get("problem"): issue for issue in issues}
    return issue_by_problem.get(selected_problem), confidence_value, None


def _gemini_suggest_commands(user_error: str, problem: str, existing: List[dict]) -> Tuple[List[dict], str | None]:
    model, err = _get_gemini_model()
    if err:
        _log_gemini_debug("suggest_commands model_error", err)
        return [], err

    cache_key = _suggest_cache_key(user_error, problem, existing)
    cached = _GEMINI_SUGGEST_CACHE.get(cache_key)
    if isinstance(cached, list):
        app.logger.info("ANALYZE: gemini suggest cache hit (%d)", len(cached))
        return cached, None

    existing_lines = "\n".join(
        f"- {sol.get('command', '').strip()}" for sol in existing if sol.get("command")
    )
    if not existing_lines:
        existing_lines = "- (none)"

    prompt = (
        "You are a Windows troubleshooting assistant.\n"
        "Return either:\n"
        "1) The exact text: I don't know\n"
        "2) Up to 5 lines, each line: <command> :: <description>\n"
        "No extra text. Use only cmd.exe or PowerShell commands.\n"
        "Use safe, common troubleshooting commands. Avoid destructive commands.\n"
        "Use placeholders like C:\\path\\to\\file if needed.\n\n"
        f"User issue: {user_error}\n"
        f"Selected problem statement: {problem}\n"
        "Existing database commands:\n"
        f"{existing_lines}\n"
    )
    _log_gemini_debug("suggest_commands prompt", prompt)

    response, request_error = _gemini_generate_content(
        model, prompt, {"temperature": 0.2, "max_output_tokens": 256}, "suggest_commands"
    )
    if request_error:
        return [], request_error

    response_text = _get_response_text(response, "suggest_commands")
    if not response_text:
        return [], "Gemini response was empty"
    _log_gemini_debug("suggest_commands response", response_text)
    if "i don't know" in response_text.lower():
        _log_gemini_debug("suggest_commands parsed", "I don't know")
        _GEMINI_SUGGEST_CACHE.set(cache_key, [])
        return [], None

    suggestions: List[dict] = []
    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\\d\\.\\)\\s]+", "", line).strip()
        if not line:
            continue
        if "::" in line:
            cmd_part, desc_part = line.split("::", 1)
        elif "|" in line:
            cmd_part, desc_part = line.split("|", 1)
        else:
            continue
        cmd = cmd_part.strip()
        desc = desc_part.strip()
        if not cmd or not desc:
            continue
        suggestions.append({"command": cmd, "description": desc, "source": "gemini"})
        if len(suggestions) >= 5:
            break

    _log_gemini_debug("suggest_commands parsed", suggestions)
    _GEMINI_SUGGEST_CACHE.set(cache_key, suggestions)
    return suggestions, None


def load_exec_log() -> List[dict]:
    if not EXEC_LOG_PATH.exists():
        return []
    try:
        return json.loads(EXEC_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_exec_log(entries: List[dict]) -> None:
    EXEC_LOG_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def log_execution(user_problem: str, matched_problem: str, description: str, command: str, result: dict) -> None:
    entries = load_exec_log()
    entries.append({
        "timestamp": datetime.now().isoformat(),
        "user_problem": user_problem,
        "matched_problem": matched_problem,
        "solution_description": description,
        "command_executed": command,
        "execution_success": result.get("success", False),
        "return_code": result.get("return_code", -1),
        "output": result.get("output", ""),
        "error": result.get("error", ""),
        "simulation_mode": result.get("simulation_mode", False),
        "session_id": request.remote_addr,
    })
    save_exec_log(entries)


def confidence_label(score: float) -> str:
    if score >= 0.9:
        return "> 90%"
    if score >= 0.8:
        return "> 80%"
    if score >= 0.7:
        return "> 70%"
    if score >= 0.6:
        return "> 60%"
    if score >= 0.5:
        return "> 50%"
    return "Low Confidence"


@app.route("/")
def home():
    return render_template("enhanced_index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    user_error = (data.get("error") or "").strip()
    if not user_error:
        return jsonify({"error": "No error message provided"}), 400

    issues = load_issues()
    if not issues:
        return jsonify({"error": "No issues database loaded"}), 500

    app.logger.info("ANALYZE: user issue: %s", user_error)
    best = None
    score = 0.0
    match_source = "gemini"
    gemini_error = None

    if _gemini_configured():
        app.logger.info("ANALYZE: first API call sent (choose problem)")
        best, score, gemini_error = _gemini_choose_problem(user_error, issues)
        if best:
            app.logger.info(
                "ANALYZE: chosen problem statement from Gemini: %s",
                best.get("problem", ""),
            )
    else:
        gemini_error = "Gemini client library not installed" if genai is None else "Gemini API key not configured"
        app.logger.warning("ANALYZE: gemini unavailable: %s", gemini_error)

    if best is None:
        if gemini_error:
            app.logger.error("Gemini match failed: %s", gemini_error)
        if _allow_tfidf_fallback():
            app.logger.warning("ANALYZE: gemini failed; falling back to tfidf matcher")
        best, score = find_best_match(user_error, issues)
        match_source = "tfidf"
        if best:
            app.logger.info("ANALYZE: tfidf match selected: %s", best.get("problem", ""))

    if best is None or (match_source == "tfidf" and score < 0.1):
        return jsonify({"error": "No matching solution found"}), 404

    if score <= 0:
        score = _score_issue(user_error, best)

    db_solutions = list(best.get("solutions", []))
    db_lines = _format_solution_lines(db_solutions, default_source="database")
    app.logger.info("ANALYZE: database solutions (%d)", len(db_solutions))
    if db_lines:
        app.logger.info("ANALYZE: database solution details:\n%s", "\n".join(db_lines))

    gemini_solutions = []
    if _gemini_configured():
        app.logger.info("ANALYZE: second API call sent (suggest commands)")
        gemini_solutions, gemini_suggest_error = _gemini_suggest_commands(
            user_error, best.get("problem", ""), best.get("solutions", [])
        )
        if gemini_suggest_error:
            app.logger.error("Gemini suggestions failed: %s", gemini_suggest_error)
        else:
            gemini_lines = _format_solution_lines(gemini_solutions, default_source="gemini")
            app.logger.info("ANALYZE: gemini suggested fixes (%d)", len(gemini_solutions))
            if gemini_lines:
                app.logger.info("ANALYZE: gemini suggestion details:\n%s", "\n".join(gemini_lines))

    combined_solutions = _merge_solutions(db_solutions, gemini_solutions)
    combined_lines = _format_solution_lines(combined_solutions, default_source="database")
    app.logger.info("ANALYZE: final fixes displayed (%d)", len(combined_solutions))
    if combined_lines:
        app.logger.info("ANALYZE: final fix details:\n%s", "\n".join(combined_lines))

    matched_issue = dict(best)
    matched_issue["solutions"] = combined_solutions

    session["matched_issue"] = matched_issue
    session["user_error"] = user_error
    session["confidence"] = float(score)

    solutions = [
        {"id": i, "command": sol.get("command", ""), "description": sol.get("description", "")}
        for i, sol in enumerate(combined_solutions)
    ]

    return jsonify({
        "success": True,
        "problem": matched_issue.get("problem", ""),
        "confidence_display": confidence_label(score),
        "user_error": user_error,
        "solutions": solutions,
        "solution_count": len(solutions),
        "match_source": match_source,
    })


@app.route("/execute", methods=["POST"])
def execute():
    data = request.get_json(force=True)
    sid = int(data.get("solution_id", 0))
    matched = session.get("matched_issue") or {}
    user_error = session.get("user_error", "")
    solutions = matched.get("solutions", [])
    if not solutions or sid >= len(solutions):
        return jsonify({"error": "No pending solutions or invalid ID"}), 400

    if not is_admin():
        return jsonify({"error": "This action requires Administrator privileges. Please run app.py as Administrator."}), 403

    sol = solutions[sid]
    cmd = sol.get("command", "")
    desc = sol.get("description", "")
    app.logger.info("APPLY: command: %s", cmd)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        result = {
            "success": proc.returncode == 0,
            "output": proc.stdout.strip(),
            "error": proc.stderr.strip(),
            "return_code": proc.returncode,
            "command": cmd,
            "simulation_mode": False,
            "solution_id": sid,
            "description": desc,
        }
    except subprocess.TimeoutExpired:
        result = {
            "success": False,
            "output": "",
            "error": "Command timed out (30s)",
            "return_code": -1,
            "command": cmd,
            "simulation_mode": False,
            "solution_id": sid,
            "description": desc,
        }
    except Exception as e:
        result = {
            "success": False,
            "output": "",
            "error": str(e),
            "return_code": -1,
            "command": cmd,
            "simulation_mode": False,
            "solution_id": sid,
            "description": desc,
        }

    if result.get("success"):
        output = result.get("output") or ""
        if output:
            app.logger.info("APPLY: output: %s", output)
        else:
            app.logger.info("APPLY: output: Successfully executed (no output)")
    else:
        app.logger.error("APPLY: error: %s", result.get("error") or "Unknown error")

    log_execution(user_error, matched.get("problem", ""), desc, cmd, result)
    return jsonify(result)


@app.route("/execute-all", methods=["POST"])
def execute_all():
    matched = session.get("matched_issue") or {}
    user_error = session.get("user_error", "")
    solutions = matched.get("solutions", [])
    if not solutions:
        return jsonify({"error": "No pending solutions"}), 400

    if not is_admin():
        return jsonify({"error": "This action requires Administrator privileges. Please run app.py as Administrator."}), 403

    results = []
    overall = True
    for i, sol in enumerate(solutions):
        cmd = sol.get("command", "")
        desc = sol.get("description", "")
        app.logger.info("APPLY: command: %s", cmd)
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            res = {
                "success": proc.returncode == 0,
                "output": proc.stdout.strip(),
                "error": proc.stderr.strip(),
                "return_code": proc.returncode,
                "command": cmd,
                "description": desc,
                "solution_id": i,
            }
        except subprocess.TimeoutExpired:
            res = {
                "success": False,
                "output": "",
                "error": "Command timed out (30s)",
                "return_code": -1,
                "command": cmd,
                "description": desc,
                "solution_id": i,
            }
        except Exception as e:
            res = {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1,
                "command": cmd,
                "description": desc,
                "solution_id": i,
            }
        if not res["success"]:
            overall = False
            app.logger.error("APPLY: error: %s", res.get("error") or "Unknown error")
        else:
            output = res.get("output") or ""
            if output:
                app.logger.info("APPLY: output: %s", output)
            else:
                app.logger.info("APPLY: output: Successfully executed (no output)")
        results.append(res)
        log_execution(user_error, matched.get("problem", ""), desc, cmd, res)

    return jsonify({
        "success": overall,
        "results": results,
        "total_solutions": len(results),
        "successful_solutions": sum(1 for r in results if r.get("success")),
    })


@app.route("/logs")
def logs():
    entries = load_exec_log()
    limit = int(request.args.get("limit", 50))
    entries = sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    return jsonify({"success": True, "logs": entries, "total_count": len(entries)})


@app.route("/stats")
def stats():
    entries = load_exec_log()
    total = len(entries)
    success = sum(1 for e in entries if e.get("execution_success"))
    return jsonify({
        "success": True,
        "stats": {
            "total_executions": total,
            "successful_executions": success,
            "success_rate": round((success / total * 100) if total else 0, 1),
            "recent_activity_24h": 0,
            "most_common_problems": [],
            "ai_mode": _get_ai_mode(),
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "mode": _get_ai_mode(),
        "ai_components": "loaded",
        "active_plugin": "Default"
    })


@app.route("/system-info")
def system_info():
    return jsonify({"success": True, "platform": platform.system(), "data": collect_system_info()})


# Plugin stubs to satisfy UI calls
@app.route("/plugins")
def plugins():
    return jsonify({
        "success": True,
        "available_plugins": ["default-plugin"],
        "plugin_info": {
            "default-plugin": {
                "name": "Aegiron Plugin",
                "type": "builtin",
                "status": "ready",
                "description": "Built-in TF-IDF style matcher (Aegiron default, no external dependencies).",
                "is_active": True
            }
        },
        "active_plugin": "default-plugin",
        "status": "loaded",
        "message": "Using built-in Aegiron plugin."
    })


@app.route("/plugins/switch", methods=["POST"])
def plugins_switch():
    data = request.get_json(force=True)
    name = (data.get("plugin_name") or "").lower()
    if name not in {"default-plugin", "tfidf", "aegiron"}:
        return jsonify({"error": f"Plugin '{name}' not available; using default-plugin"}), 400
    return jsonify({
        "success": True,
        "message": "Switched to default-plugin (only available plugin).",
        "active_plugin": "default-plugin"
    })


@app.route("/plugins/benchmark", methods=["POST"])
def plugins_benchmark():
    # Minimal fake benchmark to satisfy UI expectations
    return jsonify({
        "success": True,
        "benchmark_results": {
            "default-plugin": {
                "detailed_results": [
                    {"query": "sample", "success": True, "execution_time": 0.01, "embedding_dimension": 50}
                ],
                "summary": {"success_rate": 100.0, "average_execution_time": 0.01, "total_queries": 1}
            }
        },
        "active_plugin": "default-plugin"
    })


if __name__ == "__main__":
    # Elevate on Windows if not already admin
    if os.name == "nt":
        try:
            import ctypes

            if not ctypes.windll.shell32.IsUserAnAdmin():
                # Relaunch with admin privileges
                script = f'"{Path(__file__).resolve()}"'
                params = script
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, params, None, 1
                )
                raise SystemExit(0)
        except Exception:
            pass

    app.run(host="0.0.0.0", port=5050, debug=True)
