"""
Extract ingredients from a food image using Gemini 2.5 Flash Lite via Vertex AI REST API.
"""

import base64
import json
import os
from pathlib import Path
from typing import Union

import requests


INGREDIENTS_PROMPT = """You are an expert culinary image analyzer. Look at this food image and list the exact ingredients you can identify in the dish.

List every visible or identifiable ingredient (e.g., vegetables, proteins, sauces, herbs, grains). Use short, clear names.

CRITICAL INSTRUCTION: You must respond with ONLY a valid JSON object. Do not include any conversational text, explanations, or Markdown code blocks (do not use ```).

Use this exact format:
{"ingredients": ["ingredient 1", "ingredient 2", "ingredient 3"]}"""


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


def extract_ingredients_from_image(
    image_input: Union[str, Path, bytes, list[Union[str, Path, bytes]]],
    *,
    mime_type: str = "image/jpeg",
    api_key: Union[str, None] = None,
) -> dict:
    """
    Extract ingredients from one or multiple food images using Gemini 2.5 Flash Lite via Vertex AI REST API.

    Args:
        image_input: A single image (Path, str, or bytes) or a list of multiple images.
        mime_type: Default MIME type when image_input contains bytes (e.g. "image/jpeg", "image/png").
        api_key: Vertex AI API Key. If None, uses VERTEXAI_API_KEY from environment.

    Returns:
        A dict with at least "ingredients": list of strings, e.g.:
        {"ingredients": ["tomato", "basil", "mozzarella"]}

    Raises:
        FileNotFoundError: If an image_input path does not exist.
        ValueError: If api_key is missing or the model response could not be parsed.
        RuntimeError: If the API request fails.
    """
    # Load environment variables from .env file
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

    # Normalize image_input to a list
    if not isinstance(image_input, list):
        image_input = [image_input]

    # Prepare parts for the JSON payload
    parts = [{"text": INGREDIENTS_PROMPT}]
    
    for img in image_input:
        b64_data, resolved_mime = _image_to_base64_and_mime(img, mime_type=mime_type)
        parts.append({
            "inlineData": {
                "mimeType": resolved_mime,
                "data": b64_data
            }
        })

    # Construct the JSON payload for the REST API
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload)
    
    if not response.ok:
        raise RuntimeError(f"Vertex AI API error: {response.status_code} - {response.text}")
    
    response_data = response.json()
    
    try:
        # Navigate the JSON response structure
        model_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected response structure from Vertex AI: {response_data}")

    return _parse_ingredients_json(model_text)
