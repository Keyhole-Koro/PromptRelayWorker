#!/usr/bin/env python3
import argparse
import base64
import json
import random
import shutil
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from google.auth import default
from google.auth.transport.requests import Request

from imagen_generate import generate_imagen_base64

READABILITY_CUTOFF_START = 0.55
READABILITY_CUTOFF_END = 0.70
FINAL_PICK_TOP_K = 5
GENOME_PROMPT = "Invent diverse, weird but readable visual ingredients for a party guessing game."
REQUIRED_CONSTRAINTS: list[str] = [
    "keep one clear subject",
    "keep the scene readable at a glance",
    "avoid visual clutter",
]
TWIST_TYPES: list[str] = ["material", "scale", "role", "physics", "causality"]
BRAINSTORM_ROUNDS = 3
BRAINSTORM_BATCH = 18


def clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def load_settings(path: str) -> None:
    global READABILITY_CUTOFF_START
    global READABILITY_CUTOFF_END
    global FINAL_PICK_TOP_K
    global GENOME_PROMPT
    global REQUIRED_CONSTRAINTS
    global TWIST_TYPES
    global BRAINSTORM_ROUNDS
    global BRAINSTORM_BATCH

    raw = Path(path).read_text(encoding="utf-8")
    settings = json.loads(raw)
    if not isinstance(settings, dict):
        raise RuntimeError("evolve settings must be a JSON object")

    READABILITY_CUTOFF_START = float(settings.get("readability_cutoff_start", 0.55))
    READABILITY_CUTOFF_END = float(settings.get("readability_cutoff_end", 0.70))
    FINAL_PICK_TOP_K = int(settings.get("final_pick_top_k", 5))
    BRAINSTORM_ROUNDS = int(settings.get("brainstorm_rounds", 3))
    BRAINSTORM_BATCH = int(settings.get("brainstorm_batch", 18))

    genome_prompt = settings.get("genome_prompt")
    if isinstance(genome_prompt, str) and genome_prompt.strip():
        GENOME_PROMPT = genome_prompt.strip()

    required_constraints = settings.get("required_constraints")
    if isinstance(required_constraints, list) and all(isinstance(v, str) for v in required_constraints):
        REQUIRED_CONSTRAINTS = [str(v) for v in required_constraints]

    twist_types = settings.get("twist_types")
    if isinstance(twist_types, list) and all(isinstance(v, str) for v in twist_types) and twist_types:
        TWIST_TYPES = [str(v) for v in twist_types]


def log_step(run_id: str, item_id: str, step: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"runId": run_id, "itemId": item_id, "step": step}
    payload.update(fields)
    sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def generation_progress(generation: int, generations: int) -> float:
    denom = max(1, generations - 1)
    return clamp01(generation / denom)


def get_access_token() -> str:
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    token = credentials.token
    if not token:
        raise RuntimeError("failed to get ADC access token")
    return token


def call_vertex_json(url: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    token = get_access_token()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vertex request failed status={error.code} body={body_text}") from error


def parse_status(err: Exception) -> int | None:
    text = str(err)
    marker = "status="
    idx = text.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    code = text[start : start + 3]
    return int(code) if code.isdigit() else None


def with_retry(fn, max_retries: int):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            status = parse_status(exc)
            retriable = status in (429, 503)
            if attempt >= max_retries or not retriable:
                raise
            backoff = 0.2 * (2**attempt) + random.random() * 0.15
            time.sleep(backoff)
            attempt += 1


def try_parse_json_text(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    candidates = [raw]
    if raw.startswith("```"):
        stripped = raw.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        candidates.append(raw[first : last + 1])

    # Sometimes model returns escaped JSON as a quoted string.
    if raw.startswith("\"") and raw.endswith("\""):
        try:
            unescaped = json.loads(raw)
            if isinstance(unescaped, str):
                candidates.append(unescaped.strip())
        except json.JSONDecodeError:
            pass

    for piece in candidates:
        try:
            parsed = json.loads(piece)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                nested = try_parse_json_text(parsed)
                if nested is not None:
                    return nested
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("gemini response missing candidates")
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parsed = try_parse_json_text(text)
            if parsed is not None:
                return parsed
    raise RuntimeError("gemini response missing parseable JSON")


def extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first.get("content"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def parse_brainstorm_text_fallback(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "subjects": [],
        "scenes": [],
        "actions": [],
        "moods": [],
        "lightings": [],
        "lenses": [],
        "constraints": [],
        "style_mediums": [],
        "twists": [],
    }

    # Accept lines such as:
    # subjects: a, b, c
    # twists: material: glass body | physics: sideways gravity
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key_raw, value_raw = line.split(":", 1)
        key = normalize_text(key_raw).replace(" ", "_")
        value = value_raw.strip()
        if not value:
            continue

        if key in ("subjects", "scenes", "actions", "moods", "lightings", "lenses", "constraints", "style_mediums"):
            parts = [v.strip(" -\t") for v in re.split(r"[;,|]", value)]
            out[key].extend([p for p in parts if p])
            continue

        if key == "twists":
            parts = [v.strip(" -\t") for v in re.split(r"[;|]", value)]
            for p in parts:
                if not p:
                    continue
                if ":" in p:
                    ttype_raw, desc_raw = p.split(":", 1)
                    ttype = normalize_text(ttype_raw).replace(" ", "")
                    if ttype in TWIST_TYPES:
                        out["twists"].append(
                            {
                                "type": ttype,
                                "description": desc_raw.strip(),
                                "strengthHint": 1,
                            }
                        )
    return out


def normalize_text(v: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", v.lower())).strip()


def dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = v.strip()
        if not s:
            continue
        key = normalize_text(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def dedupe_twists(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in values:
        if not isinstance(t, dict):
            continue
        ttype = str(t.get("type", "")).strip()
        desc = str(t.get("description", "")).strip()
        if ttype not in TWIST_TYPES or not desc:
            continue
        strength_hint = t.get("strengthHint", t.get("strength", 1))
        try:
            strength = 2 if int(strength_hint) == 2 else 1
        except Exception:  # noqa: BLE001
            strength = 1
        key = f"{ttype}|{normalize_text(desc)}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": ttype, "description": desc, "strength": strength})
    return out


def brainstorm_once(
    *,
    project: str,
    model: str,
    max_retries: int,
    recent_subjects: list[str],
) -> dict[str, Any]:
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    avoid = ", ".join(recent_subjects[:12]) if recent_subjects else "none"

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "\n".join(
                            [
                                GENOME_PROMPT,
                                "Brainstorm a large and diverse ingredient pool.",
                                "Return ONLY JSON.",
                                f"Avoid reusing these recent subjects when possible: {avoid}",
                                f"Generate about {BRAINSTORM_BATCH} entries per list.",
                            ]
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.95,
            "topP": 0.98,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": [
                    "subjects",
                    "scenes",
                    "actions",
                    "moods",
                    "lightings",
                    "lenses",
                    "constraints",
                    "twists",
                    "style_mediums"
                ],
                "properties": {
                    "subjects": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "scenes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "actions": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "moods": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "lightings": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "lenses": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "style_mediums": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "twists": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "required": ["type", "description"],
                            "properties": {
                                "type": {"type": "STRING"},
                                "description": {"type": "STRING"},
                                "strengthHint": {"type": "INTEGER"}
                            }
                        }
                    }
                }
            }
        }
    }

    raw = with_retry(lambda: call_vertex_json(url, body), max_retries)
    try:
        return extract_json_from_gemini_response(raw)
    except Exception:
        # Fallback with simpler output contract when schema mode still returns non-JSON text.
        fallback_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "\n".join(
                                [
                                    GENOME_PROMPT,
                                    "Return STRICT JSON object only, no markdown.",
                                    f"Avoid these recent subjects when possible: {avoid}",
                                    f"Generate about {BRAINSTORM_BATCH} entries per list.",
                                    "Required keys:",
                                    "subjects, scenes, actions, moods, lightings, lenses, constraints, style_mediums, twists",
                                    "twists items must include: type, description, strengthHint",
                                ]
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.9,
                "topP": 0.98,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        raw_fallback = with_retry(lambda: call_vertex_json(url, fallback_body), max_retries)
        try:
            return extract_json_from_gemini_response(raw_fallback)
        except Exception:
            # Last fallback: accept plain text lists and parse heuristically.
            text_body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "\n".join(
                                    [
                                        GENOME_PROMPT,
                                        "Return plain text lines in this exact format:",
                                        "subjects: ...",
                                        "scenes: ...",
                                        "actions: ...",
                                        "moods: ...",
                                        "lightings: ...",
                                        "lenses: ...",
                                        "style_mediums: photo, illustration",
                                        "constraints: ...",
                                        "twists: type: description | type: description",
                                        f"Avoid these recent subjects: {avoid}",
                                    ]
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": {"temperature": 0.7, "topP": 0.95, "maxOutputTokens": 1024},
            }
            raw_text = with_retry(lambda: call_vertex_json(url, text_body), max_retries)
            text = extract_text_from_gemini_response(raw_text)
            parsed = parse_brainstorm_text_fallback(text)
            return parsed


def build_brainstorm_pool(
    *,
    project: str,
    model: str,
    run_id: str,
    item_id: str,
    max_retries: int,
    recent_subjects: list[str],
) -> dict[str, Any]:
    agg: dict[str, Any] = {
        "subjects": [],
        "scenes": [],
        "actions": [],
        "moods": [],
        "lightings": [],
        "lenses": [],
        "constraints": [],
        "style_mediums": [],
        "twists": [],
    }

    for i in range(max(1, BRAINSTORM_ROUNDS)):
        try:
            data = brainstorm_once(project=project, model=model, max_retries=max_retries, recent_subjects=recent_subjects)
        except Exception as error:  # noqa: BLE001
            log_step(run_id, item_id, "brainstorm_round_failed", round=i, error=str(error))
            continue
        for key in ("subjects", "scenes", "actions", "moods", "lightings", "lenses", "constraints", "style_mediums"):
            values = data.get(key)
            if isinstance(values, list):
                agg[key].extend([str(v) for v in values if isinstance(v, str)])
        twists = data.get("twists")
        if isinstance(twists, list):
            agg["twists"].extend(twists)
        log_step(run_id, item_id, "brainstorm_round_done", round=i, subjects=len(agg["subjects"]))

    for key in ("subjects", "scenes", "actions", "moods", "lightings", "lenses", "constraints", "style_mediums"):
        agg[key] = dedupe_strings(agg[key])

    agg["twists"] = dedupe_twists(agg["twists"])

    # enforce required minimum entries
    if not agg["style_mediums"]:
        agg["style_mediums"] = ["photo", "illustration"]
    agg["style_mediums"] = [m for m in agg["style_mediums"] if m in ("photo", "illustration")] or ["photo", "illustration"]

    agg["constraints"] = dedupe_strings(agg["constraints"] + REQUIRED_CONSTRAINTS)

    for key in ("subjects", "scenes", "actions", "moods", "lightings"):
        if not agg[key]:
            raise RuntimeError(f"brainstorm pool missing required key: {key}")

    log_step(
        run_id,
        item_id,
        "brainstorm_pool_ready",
        subjects=len(agg["subjects"]),
        scenes=len(agg["scenes"]),
        actions=len(agg["actions"]),
        twists=len(agg["twists"]),
    )
    return agg


def choose_with_avoid(values: list[str], avoid: list[str]) -> str:
    normalized_avoid = {normalize_text(v) for v in avoid}
    filtered = [v for v in values if normalize_text(v) not in normalized_avoid]
    if filtered:
        return random.choice(filtered)
    return random.choice(values)


def compose_genome_from_pool(pool: dict[str, Any], recent_subjects: list[str]) -> dict[str, Any]:
    subject = choose_with_avoid(pool["subjects"], recent_subjects)
    scene = random.choice(pool["scenes"])
    action = random.choice(pool["actions"])

    twist_count = 1 if random.random() < 0.75 else (2 if random.random() < 0.35 else 0)
    twists = pool["twists"].copy()
    random.shuffle(twists)
    twist = twists[: min(twist_count, len(twists))]

    medium = random.choice(pool["style_mediums"])
    mood = random.choice(pool["moods"])
    lighting = random.choice(pool["lightings"])
    lens = random.choice(pool["lenses"]) if pool["lenses"] and random.random() < 0.6 else None

    framing = "center" if random.random() < 0.5 else "rule_of_thirds"

    sampled_constraints = random.sample(pool["constraints"], k=min(2, len(pool["constraints"])))
    constraints = dedupe_strings(sampled_constraints + REQUIRED_CONSTRAINTS)

    genome = {
        "baseScene": scene,
        "subject": subject,
        "action": action,
        "twist": [{"type": t["type"], "description": t["description"], "strength": t["strength"]} for t in twist],
        "style": {
            "medium": medium,
            "mood": mood,
            "lighting": lighting,
            **({"lens": lens} if lens else {}),
        },
        "composition": {
            "focus": "single_subject",
            "framing": framing,
            "backgroundClarity": "clear",
        },
        "constraints": constraints,
    }
    return genome


def build_prompt(genome: dict[str, Any]) -> str:
    twists = genome.get("twist", [])
    if isinstance(twists, list) and twists:
        twist_text = ", ".join(f"{t['type']}:{t['description']}(strength={t['strength']})" for t in twists if isinstance(t, dict))
    else:
        twist_text = "none"

    lens = genome.get("style", {}).get("lens") if isinstance(genome.get("style"), dict) else None
    lens_text = f", lens {lens}" if isinstance(lens, str) and lens else ""

    return ". ".join(
        [
            f"Single {genome['subject']} in {genome['baseScene']}",
            f"Action: {genome['action']}",
            f"Twist design: {twist_text}",
            f"Visual style: {genome['style']['medium']}, mood {genome['style']['mood']}, lighting {genome['style']['lighting']}{lens_text}",
            f"Composition: framing {genome['composition']['framing']}, single_subject focus, clear background",
            "Keep the image clean and instantly understandable, with one obvious focal point.",
            "Use natural details and coherent scene logic so the weirdness feels intentional rather than random.",
        ]
    )


def score_with_gemini(*, project: str, model: str, prompt: str, image_base64: str, max_retries: int) -> dict[str, Any]:
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    req = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "\n".join(
                            [
                                "Score the image against the prompt and return ONLY JSON.",
                                "readability: how clearly a player can identify scene/subject/action.",
                                "twist: how noticeable and interesting the twist is.",
                                "aesthetic: overall visual quality.",
                                "Each score must be number between 0 and 1.",
                                f"Prompt: {prompt}",
                            ]
                        )
                    },
                    {"inlineData": {"mimeType": "image/png", "data": image_base64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "topP": 0,
            "maxOutputTokens": 256,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": ["readability", "twist", "aesthetic", "short_reason"],
                "properties": {
                    "readability": {"type": "NUMBER"},
                    "twist": {"type": "NUMBER"},
                    "aesthetic": {"type": "NUMBER"},
                    "labels": {
                        "type": "OBJECT",
                        "properties": {"scene": {"type": "STRING"}, "subject": {"type": "STRING"}, "action": {"type": "STRING"}},
                    },
                    "flags": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "short_reason": {"type": "STRING"},
                },
            },
        },
    }

    raw = with_retry(lambda: call_vertex_json(url, req), max_retries)
    parsed = extract_json_from_gemini_response(raw)

    readability = clamp01(float(parsed.get("readability", 0)))
    twist = clamp01(float(parsed.get("twist", 0)))
    aesthetic = clamp01(float(parsed.get("aesthetic", 0)))
    fitness = clamp01(0.5 * readability + 0.35 * twist + 0.15 * aesthetic)
    labels = parsed.get("labels") if isinstance(parsed.get("labels"), dict) else {}
    flags = parsed.get("flags") if isinstance(parsed.get("flags"), list) else []
    short_reason = parsed.get("short_reason") if isinstance(parsed.get("short_reason"), str) else "scored by gemini"

    return {
        "readability": readability,
        "twist": twist,
        "aesthetic": aesthetic,
        "fitness": fitness,
        "labels": labels,
        "flags": [f for f in flags if isinstance(f, str)],
        "short_reason": short_reason,
    }


def twist_signature(genome: dict[str, Any]) -> str:
    twists = genome.get("twist", [])
    if not isinstance(twists, list):
        return "none"
    parts: list[str] = []
    for t in twists:
        if not isinstance(t, dict):
            continue
        parts.append(f"{t.get('type', '')}:{t.get('description', '')}:{t.get('strength', '')}")
    return "|".join(sorted(parts)) if parts else "none"


def genome_signature(genome: dict[str, Any]) -> str:
    style = genome.get("style", {}) if isinstance(genome.get("style"), dict) else {}
    comp = genome.get("composition", {}) if isinstance(genome.get("composition"), dict) else {}
    return "|".join(
        [
            str(genome.get("baseScene", "")),
            str(genome.get("subject", "")),
            str(genome.get("action", "")),
            str(style.get("medium", "")),
            str(style.get("mood", "")),
            str(style.get("lighting", "")),
            str(comp.get("framing", "")),
            twist_signature(genome),
        ]
    )


def genome_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    fields = 0
    diff = 0
    for key in ("subject", "action"):
        fields += 1
        if a.get(key) != b.get(key):
            diff += 1
    for key in ("medium", "mood", "lighting"):
        fields += 1
        sa = a.get("style", {}) if isinstance(a.get("style"), dict) else {}
        sb = b.get("style", {}) if isinstance(b.get("style"), dict) else {}
        if sa.get(key) != sb.get(key):
            diff += 1
    fields += 1
    ca = a.get("composition", {}) if isinstance(a.get("composition"), dict) else {}
    cb = b.get("composition", {}) if isinstance(b.get("composition"), dict) else {}
    if ca.get("framing") != cb.get("framing"):
        diff += 1
    fields += 1
    if twist_signature(a) != twist_signature(b):
        diff += 1
    return diff / fields


def novelty_score(genome: dict[str, Any], refs: list[dict[str, Any]], seen_signatures: set[str]) -> float:
    if not refs:
        return 1.0
    distances = [genome_distance(genome, ref) for ref in refs]
    novelty = sum(distances) / len(distances)
    if genome_signature(genome) in seen_signatures:
        novelty *= 0.5
    return clamp01(novelty)


def blended_fitness(readability: float, twist: float, aesthetic: float, novelty: float) -> float:
    return clamp01(0.40 * readability + 0.25 * twist + 0.15 * aesthetic + 0.20 * novelty)


def rarity_bonus(value: str, freq: dict[str, int], total: int) -> float:
    count = freq.get(value, 1)
    return clamp01(1.0 - (count - 1) / max(1, total - 1))


def save_base64_png(base64_png: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(base64.b64decode(base64_png))


def save_to_available_pool(
    *,
    local_data_dir: str,
    available_prefix: str,
    item_id: str,
    selected_image_path: str,
    meta: dict[str, Any],
) -> str:
    clean_prefix = available_prefix.strip("/").strip()
    if not clean_prefix:
        raise RuntimeError("available prefix must not be empty")

    target_dir = Path(local_data_dir) / clean_prefix / item_id
    target_dir.mkdir(parents=True, exist_ok=True)

    final_png = target_dir / "final.png"
    meta_json = target_dir / "meta.json"

    shutil.copyfile(selected_image_path, final_png)
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--imagenRegion", required=True)
    parser.add_argument("--imagenModel", required=True)
    parser.add_argument("--geminiModel", required=True)
    parser.add_argument("--localDataDir", required=True)
    parser.add_argument("--runId", required=True)
    parser.add_argument("--itemId", required=True)
    parser.add_argument("--aspectRatio", required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--parents", type=int, required=True)
    parser.add_argument("--maxVertexRetries", type=int, required=True)
    parser.add_argument("--settingsPath", required=True)
    parser.add_argument("--saveToPool", action="store_true")
    parser.add_argument("--availablePrefix", default="pool/available")
    args = parser.parse_args()

    load_settings(args.settingsPath)
    log_step(
        args.runId,
        args.itemId,
        "start",
        generations=args.generations,
        population=args.population,
        parents=args.parents,
        aspectRatio=args.aspectRatio,
        brainstormRounds=BRAINSTORM_ROUNDS,
        settingsPath=args.settingsPath,
    )

    best_overall: dict[str, Any] | None = None
    all_candidates: list[dict[str, Any]] = []
    prompt_history: list[dict[str, Any]] = []
    recent_subjects: list[str] = []

    for generation in range(args.generations):
        progress = generation_progress(generation, args.generations)
        readability_cutoff = READABILITY_CUTOFF_START + (READABILITY_CUTOFF_END - READABILITY_CUTOFF_START) * progress
        log_step(args.runId, args.itemId, "generation_start", generation=generation, readabilityCutoff=round(readability_cutoff, 3))

        pool = build_brainstorm_pool(
            project=args.project,
            model=args.geminiModel,
            run_id=args.runId,
            item_id=args.itemId,
            max_retries=args.maxVertexRetries,
            recent_subjects=recent_subjects,
        )

        genomes = [compose_genome_from_pool(pool, recent_subjects) for _ in range(args.population)]

        novelty_refs = genomes
        seen_signatures: set[str] = set()
        candidates: list[dict[str, Any]] = []

        for i, genome in enumerate(genomes):
            candidate_id = f"{args.itemId}-g{generation}-i{i}"
            prompt = build_prompt(genome)

            image_base64 = with_retry(
                lambda: generate_imagen_base64(
                    project=args.project,
                    region=args.imagenRegion,
                    model=args.imagenModel,
                    prompt=prompt,
                    aspect_ratio=args.aspectRatio,
                ),
                args.maxVertexRetries,
            )

            temp_image_path = Path(args.localDataDir) / "runs" / args.runId / candidate_id / "candidate.png"
            save_base64_png(image_base64, temp_image_path)
            (temp_image_path.parent / "prompt.txt").write_text(prompt, encoding="utf-8")

            try:
                scores = score_with_gemini(
                    project=args.project,
                    model=args.geminiModel,
                    prompt=prompt,
                    image_base64=image_base64,
                    max_retries=args.maxVertexRetries,
                )
            except Exception:
                scores = {
                    "readability": 0.0,
                    "twist": 0.0,
                    "aesthetic": 0.0,
                    "fitness": 0.0,
                    "labels": {},
                    "flags": ["gemini_score_failed_fallback"],
                    "short_reason": "fallback: scoring failed",
                }

            novelty = novelty_score(genome, novelty_refs, seen_signatures)
            scores["novelty"] = novelty
            scores["fitness"] = blended_fitness(scores["readability"], scores["twist"], scores["aesthetic"], novelty)
            seen_signatures.add(genome_signature(genome))

            recent_subjects.insert(0, str(genome.get("subject", "")))
            recent_subjects = recent_subjects[:24]

            prompt_history.append({"generation": generation, "candidateIndex": i, "prompt": prompt, "scores": scores})
            log_step(
                args.runId,
                args.itemId,
                "candidate_scored",
                generation=generation,
                candidateIndex=i,
                subject=str(genome.get("subject", "")),
                readability=round(float(scores["readability"]), 3),
                fitness=round(float(scores["fitness"]), 3),
                novelty=round(float(scores.get("novelty", 0)), 3),
            )

            candidates.append(
                {
                    "genome": genome,
                    "prompt": prompt,
                    "imageBase64": image_base64,
                    "tempImagePath": str(temp_image_path),
                    "scores": scores,
                    "generation": generation,
                }
            )

        all_candidates.extend(candidates)

        subject_freq: dict[str, int] = {}
        twist_freq: dict[str, int] = {}
        scene_freq: dict[str, int] = {}
        for candidate in candidates:
            genome = candidate["genome"]
            subject_key = str(genome.get("subject", ""))
            scene_key = str(genome.get("baseScene", ""))
            twist_key = twist_signature(genome)
            subject_freq[subject_key] = subject_freq.get(subject_key, 0) + 1
            scene_freq[scene_key] = scene_freq.get(scene_key, 0) + 1
            twist_freq[twist_key] = twist_freq.get(twist_key, 0) + 1

        for candidate in candidates:
            genome = candidate["genome"]
            subject_key = str(genome.get("subject", ""))
            scene_key = str(genome.get("baseScene", ""))
            twist_key = twist_signature(genome)
            diversity = (
                0.45 * rarity_bonus(subject_key, subject_freq, len(candidates))
                + 0.35 * rarity_bonus(twist_key, twist_freq, len(candidates))
                + 0.20 * rarity_bonus(scene_key, scene_freq, len(candidates))
            )
            candidate["scores"]["diversity_bonus"] = clamp01(diversity)
            candidate["scores"]["fitness"] = clamp01(0.8 * candidate["scores"]["fitness"] + 0.2 * diversity)

        survivors = [c for c in candidates if c["scores"]["readability"] >= readability_cutoff]
        ranked_survivors = sorted(survivors, key=lambda x: x["scores"]["fitness"], reverse=True)
        ranked_all = sorted(candidates, key=lambda x: x["scores"]["fitness"], reverse=True)

        generation_best = ranked_survivors[0] if ranked_survivors else (ranked_all[0] if ranked_all else None)
        if generation_best is None:
            raise RuntimeError("generation produced no candidates")

        if best_overall is None or generation_best["scores"]["fitness"] > best_overall["scores"]["fitness"]:
            best_overall = generation_best

        log_step(
            args.runId,
            args.itemId,
            "generation_summary",
            generation=generation,
            candidates=len(candidates),
            survivors=len(survivors),
        )

    if best_overall is None:
        raise RuntimeError("evolution did not produce any candidate")

    ranked_final = sorted(all_candidates, key=lambda x: x["scores"]["fitness"], reverse=True)
    if ranked_final:
        k = min(FINAL_PICK_TOP_K, len(ranked_final))
        chosen = random.choice(ranked_final[:k])
        best_prompt = chosen["prompt"]
        best_scores = chosen["scores"]
        best_generation = chosen["generation"]
        best_genome = chosen["genome"]
        best_temp_path = chosen["tempImagePath"]
    else:
        best_prompt = best_overall["prompt"]
        best_scores = best_overall["scores"]
        best_generation = best_overall["generation"]
        best_genome = best_overall["genome"]
        best_temp_path = best_overall["tempImagePath"]

    log_step(
        args.runId,
        args.itemId,
        "final_selected",
        generation=best_generation,
        subject=str(best_genome.get("subject", "")),
        fitness=round(float(best_scores.get("fitness", 0)), 3),
    )

    result = {
        "itemId": args.itemId,
        "runId": args.runId,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aspectRatio": args.aspectRatio,
        "prompt": best_prompt,
        "genome": best_genome,
        "scores": best_scores,
        "generation": best_generation,
        "tempImagePath": best_temp_path,
        "promptHistory": prompt_history,
    }

    if args.saveToPool:
        saved_dir = save_to_available_pool(
            local_data_dir=args.localDataDir,
            available_prefix=args.availablePrefix,
            item_id=args.itemId,
            selected_image_path=best_temp_path,
            meta=result,
        )
        log_step(args.runId, args.itemId, "saved_to_pool", availableDir=saved_dir)

    log_step(args.runId, args.itemId, "done")
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(str(exc))
        raise SystemExit(1)
