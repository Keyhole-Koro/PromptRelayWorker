#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

from google.auth import default
from google.auth.transport.requests import Request


def as_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_access_token() -> str:
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    token = credentials.token
    if not token:
        raise RuntimeError("failed to get ADC access token")
    return token


def generate_imagen_base64(
    *,
    project: str,
    region: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    timeout_s: int = 120,
) -> str:
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{region}/publishers/google/models/{model}:predict"
    )

    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
        },
    }

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
    except urllib.error.HTTPError as error:
        body_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vertex request failed status={error.code} body={body_text}") from error

    top = as_object(json.loads(raw))
    predictions = top.get("predictions")
    first = as_object(predictions[0]) if isinstance(predictions, list) and predictions else {}
    nested_image = as_object(first.get("image"))

    bytes_base64 = first.get("bytesBase64Encoded")
    if not isinstance(bytes_base64, str):
        bytes_base64 = nested_image.get("bytesBase64Encoded")
    if not isinstance(bytes_base64, str) or not bytes_base64:
        raise RuntimeError("imagen response missing bytesBase64Encoded")

    return bytes_base64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--aspectRatio", required=True)
    args = parser.parse_args()

    bytes_base64 = generate_imagen_base64(
        project=args.project,
        region=args.region,
        model=args.model,
        prompt=args.prompt,
        aspect_ratio=args.aspectRatio,
    )

    sys.stdout.write(json.dumps({"bytesBase64Encoded": bytes_base64}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(str(exc))
        raise SystemExit(1)
