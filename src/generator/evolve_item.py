#!/usr/bin/env python3
import argparse
import base64
import json
import random
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
PARENT_SUBJECT_LIMIT = 2
FINAL_PICK_TOP_K = 5
GENOME_PROMPT = "Create a weird but readable single-subject image concept for a party guessing game."
REQUIRED_CONSTRAINTS: list[str] = ["no text", "no logos", "no collage", "no extra subjects"]
TWIST_TYPES: list[str] = ["material", "scale", "role", "physics", "causality"]


def clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def load_settings(path: str) -> None:
    global READABILITY_CUTOFF_START
    global READABILITY_CUTOFF_END
    global PARENT_SUBJECT_LIMIT
    global FINAL_PICK_TOP_K
    global GENOME_PROMPT
    global REQUIRED_CONSTRAINTS
    global TWIST_TYPES

    raw = Path(path).read_text(encoding="utf-8")
    settings = json.loads(raw)
    if not isinstance(settings, dict):
        raise RuntimeError("evolve settings must be a JSON object")

    READABILITY_CUTOFF_START = float(settings.get("readability_cutoff_start", 0.55))
    READABILITY_CUTOFF_END = float(settings.get("readability_cutoff_end", 0.70))
    PARENT_SUBJECT_LIMIT = int(settings.get("parent_subject_limit", 2))
    FINAL_PICK_TOP_K = int(settings.get("final_pick_top_k", 5))

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
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    raise RuntimeError("gemini response missing parseable JSON")


def normalize_genome(raw: dict[str, Any]) -> dict[str, Any]:
    base_scene = str(raw.get("baseScene", "unknown scene"))
    subject = str(raw.get("subject", "subject"))
    action = str(raw.get("action", "doing something"))

    style_raw = raw.get("style") if isinstance(raw.get("style"), dict) else {}
    medium = str(style_raw.get("medium", "photo"))
    if medium not in ("photo", "illustration"):
        medium = "photo"
    style = {
        "medium": medium,
        "mood": str(style_raw.get("mood", "mysterious")),
        "lighting": str(style_raw.get("lighting", "soft daylight")),
    }
    lens = style_raw.get("lens")
    if isinstance(lens, str) and lens.strip():
        style["lens"] = lens.strip()

    comp_raw = raw.get("composition") if isinstance(raw.get("composition"), dict) else {}
    framing = str(comp_raw.get("framing", "center"))
    if framing not in ("center", "rule_of_thirds"):
        framing = "center"
    composition = {
        "focus": "single_subject",
        "framing": framing,
        "backgroundClarity": "clear",
    }

    twists_raw = raw.get("twist") if isinstance(raw.get("twist"), list) else []
    twists: list[dict[str, Any]] = []
    for t in twists_raw[:2]:
        if not isinstance(t, dict):
            continue
        ttype = str(t.get("type", "material"))
        if ttype not in TWIST_TYPES:
            continue
        description = str(t.get("description", "unexpected behavior"))
        strength = t.get("strength", 1)
        strength_num = 2 if int(strength) == 2 else 1
        twists.append({"type": ttype, "description": description, "strength": strength_num})

    constraints_raw = raw.get("constraints") if isinstance(raw.get("constraints"), list) else []
    constraints = [str(v) for v in constraints_raw if isinstance(v, str) and v.strip()]
    if not constraints:
        constraints = REQUIRED_CONSTRAINTS.copy()
    for needed in REQUIRED_CONSTRAINTS:
        if needed not in constraints:
            constraints.append(needed)

    return {
        "baseScene": base_scene,
        "subject": subject,
        "action": action,
        "twist": twists,
        "style": style,
        "composition": composition,
        "constraints": constraints,
    }


def build_prompt(genome: dict[str, Any]) -> str:
    twist = genome.get("twist", [])
    if isinstance(twist, list) and twist:
        twist_text = ", ".join(f"{t['type']}:{t['description']}(strength={t['strength']})" for t in twist if isinstance(t, dict))
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
            f"Constraints: {', '.join(genome['constraints'])}",
        ]
    )


def suggest_genome(
    *,
    project: str,
    model: str,
    run_id: str,
    item_id: str,
    generation: int,
    candidate_index: int,
    max_retries: int,
    recent_subjects: list[str],
    parent_genome: dict[str, Any] | None,
) -> dict[str, Any]:
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    avoid_subjects = ", ".join(recent_subjects[:8]) if recent_subjects else "none"

    parent_hint = ""
    if parent_genome is not None:
        parent_hint = f"Parent genome to mutate lightly: {json.dumps(parent_genome, ensure_ascii=False)}"

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "\n".join(
                            [
                                GENOME_PROMPT,
                                "Return only JSON.",
                                "Create one strange but instantly understandable single-subject concept.",
                                f"Avoid repeating these recent subjects when possible: {avoid_subjects}",
                                "Use 0..2 twist entries.",
                                "Allowed twist types: " + ", ".join(TWIST_TYPES),
                                "Composition focus must be single_subject and backgroundClarity must be clear.",
                                parent_hint,
                            ]
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "topP": 0.95,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "required": ["baseScene", "subject", "action", "style", "composition", "constraints"],
                "properties": {
                    "baseScene": {"type": "STRING"},
                    "subject": {"type": "STRING"},
                    "action": {"type": "STRING"},
                    "twist": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "required": ["type", "description", "strength"],
                            "properties": {
                                "type": {"type": "STRING"},
                                "description": {"type": "STRING"},
                                "strength": {"type": "INTEGER"},
                            },
                        },
                    },
                    "style": {
                        "type": "OBJECT",
                        "required": ["medium", "mood", "lighting"],
                        "properties": {
                            "medium": {"type": "STRING"},
                            "mood": {"type": "STRING"},
                            "lighting": {"type": "STRING"},
                            "lens": {"type": "STRING"},
                        },
                    },
                    "composition": {
                        "type": "OBJECT",
                        "required": ["focus", "framing", "backgroundClarity"],
                        "properties": {
                            "focus": {"type": "STRING"},
                            "framing": {"type": "STRING"},
                            "backgroundClarity": {"type": "STRING"},
                        },
                    },
                    "constraints": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
            },
        },
    }

    raw = with_retry(lambda: call_vertex_json(url, body), max_retries)
    parsed = extract_json_from_gemini_response(raw)
    genome = normalize_genome(parsed)

    log_step(
        run_id,
        item_id,
        "genome_suggested",
        generation=generation,
        candidateIndex=candidate_index,
        subject=genome.get("subject", ""),
        baseScene=genome.get("baseScene", ""),
    )
    return genome


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


def select_diverse_parents(parent_pool: list[dict[str, Any]], parent_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    subject_count: dict[str, int] = {}
    for candidate in parent_pool:
        genome = candidate["genome"]
        subject = str(genome.get("subject", ""))
        current = subject_count.get(subject, 0)
        if current >= PARENT_SUBJECT_LIMIT:
            continue
        selected.append(genome)
        subject_count[subject] = current + 1
        if len(selected) >= parent_count:
            return selected

    for candidate in parent_pool:
        if len(selected) >= parent_count:
            break
        genome = candidate["genome"]
        if genome in selected:
            continue
        selected.append(genome)
    return selected


def rarity_bonus(value: str, freq: dict[str, int], total: int) -> float:
    count = freq.get(value, 1)
    return clamp01(1.0 - (count - 1) / max(1, total - 1))


def save_base64_png(base64_png: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(base64.b64decode(base64_png))


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

        genomes: list[dict[str, Any]] = []
        for i in range(args.population):
            genome = suggest_genome(
                project=args.project,
                model=args.geminiModel,
                run_id=args.runId,
                item_id=args.itemId,
                generation=generation,
                candidate_index=i,
                max_retries=args.maxVertexRetries,
                recent_subjects=recent_subjects,
                parent_genome=None,
            )
            genomes.append(genome)

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
            recent_subjects = recent_subjects[:16]

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

        parent_pool = ranked_survivors if ranked_survivors else ranked_all
        parent_count = max(1, min(args.parents, len(parent_pool)))
        selected_parents = select_diverse_parents(parent_pool, parent_count)

        log_step(
            args.runId,
            args.itemId,
            "generation_summary",
            generation=generation,
            candidates=len(candidates),
            survivors=len(survivors),
            selectedParents=len(selected_parents),
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
    log_step(args.runId, args.itemId, "done")
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(str(exc))
        raise SystemExit(1)
