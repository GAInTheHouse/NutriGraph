"""
Extract ingredients (and optionally a dish name) from a food image using
Gemini 2.5 Flash Lite via the Vertex AI REST API.
"""

import base64
import json
import os
from pathlib import Path
from typing import Union

import requests


# ── Prompts ───────────────────────────────────────────────────────────────────

INGREDIENTS_PROMPT = """You are an expert culinary image analyzer. Look at this food image and:
1. Identify the name of the dish (e.g. "Spaghetti Carbonara", "Chicken Caesar Salad").
2. List the exact ingredients you can identify in the dish (e.g., vegetables, proteins, sauces, herbs, grains). Use short, clear names.

CRITICAL INSTRUCTION: You must respond with ONLY a valid JSON object. Do not include any conversational text, explanations, or Markdown code blocks (do not use ```).

Use this exact format:
{"dish_name": "Dish Name Here", "ingredients": ["ingredient 1", "ingredient 2", "ingredient 3"]}"""


def _image_to_base64_and_mime(
    image_input: Union[str, Path, bytes],
    mime_type: str = "image/jpeg",
) -> tuple[str, str]:
    """Turn a path or bytes into a base64 string and mime type for the REST API."""
    if isinstance(image_input, bytes):
        return base64.b64encode(image_input).decode("utf-8"), mime_type
    
    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    part_mime = mime_map.get(suffix, mime_type)
    return base64.b64encode(data).decode("utf-8"), part_mime


def _parse_ingredients_json(text: str) -> dict:
    """Parse model output into a JSON object; tolerate markdown code fences."""
    text = text.strip()
    # Remove optional markdown code block
    if "```json" in text:
        text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _resolve_api_key(api_key: Union[str, None]) -> str:
    """Load .env and return the effective API key, raising ValueError if absent."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = api_key or os.environ.get("VERTEXAI_API_KEY")
    if not key:
        raise ValueError(
            "Vertex AI API Key required. Set VERTEXAI_API_KEY in your .env file or environment."
        )
    return key


def _call_gemini(
    prompt: str,
    image_input: Union[str, Path, bytes, list],
    mime_type: str,
    api_key: str,
) -> str:
    """
    Send a prompt + one or more images to Gemini 2.5 Flash Lite and return the raw text response.

    Raises:
        RuntimeError: If the Vertex AI API returns a non-2xx status.
        ValueError: If the response JSON has an unexpected structure.
    """
    if not isinstance(image_input, list):
        image_input = [image_input]

    parts = [{"text": prompt}]
    for img in image_input:
        b64_data, resolved_mime = _image_to_base64_and_mime(img, mime_type=mime_type)
        parts.append({"inlineData": {"mimeType": resolved_mime, "data": b64_data}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"gemini-2.5-flash-lite:generateContent?key={api_key}"
    )
    response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)

    if not response.ok:
        raise RuntimeError(f"Vertex AI API error: {response.status_code} - {response.text}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected response structure from Vertex AI: {data}") from exc


def extract_ingredients_from_image(
    image_input: Union[str, Path, bytes, list[Union[str, Path, bytes]]],
    *,
    mime_type: str = "image/jpeg",
    api_key: Union[str, None] = None,
) -> dict:
    """
    Extract the dish name and ingredients from one or multiple food images using Gemini 2.5 Flash Lite.

    Args:
        image_input: A single image (Path, str, or bytes) or a list of multiple images.
        mime_type: Default MIME type when image_input contains bytes.
        api_key: Vertex AI API Key. Falls back to VERTEXAI_API_KEY env var.

    Returns:
        ``{"dish_name": "Spaghetti Carbonara", "ingredients": ["tomato", "basil", "mozzarella"]}``

        If the model omits ``dish_name`` or ``ingredients``, defaults are applied.

    Raises:
        FileNotFoundError: If an image path does not exist.
        ValueError: If the API key is missing or the response cannot be parsed.
        RuntimeError: If the Vertex AI API request fails.
    """
    key = _resolve_api_key(api_key)
    text = _call_gemini(INGREDIENTS_PROMPT, image_input, mime_type, key)
    result = _parse_ingredients_json(text)
    result.setdefault("dish_name", "Analyzed Dish")
    result.setdefault("ingredients", [])
    return result
