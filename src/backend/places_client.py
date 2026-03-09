"""
Google Places API client for NutriGraph.

Wraps the Places Text Search endpoint to look up real-world restaurant
establishments. All outbound HTTP calls to Google are made here so the
Streamlit frontend never touches the API key directly.
"""
import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()

from src.core.models import PlaceSearchResult  # noqa: E402

logger = logging.getLogger(__name__)

# Places API (New) — POST endpoint
_PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Only request the fields we actually use; field masking controls billing tier.
_FIELD_MASK = "places.id,places.displayName,places.formattedAddress"


class PlacesAPIError(Exception):
    """Raised when the Google Places API (New) returns an unexpected error."""


class GooglePlacesClient:
    """
    Thin wrapper around the Google Places API (New) Text Search endpoint.

    Uses ``POST https://places.googleapis.com/v1/places:searchText`` with the
    API key in the ``X-Goog-Api-Key`` header and a ``X-Goog-FieldMask`` to
    limit billed fields to only what NutriGraph needs.

    Initialisation reads ``GOOGLE_PLACES_API_KEY`` from the environment.
    A ``ValueError`` is raised immediately if the key is absent so callers
    get a clear error rather than a cryptic 403 from Google at query time.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GOOGLE_PLACES_API_KEY is not set. "
                "Add it to your .env file or environment before using the Places client."
            )
        self._api_key = api_key

    def search_restaurants(self, query: str) -> list[PlaceSearchResult]:
        """
        Search for restaurant establishments matching *query*.

        Calls the Places API (New) Text Search endpoint with
        ``"includedType": "restaurant"`` so results are filtered to
        food/dining establishments.

        Args:
            query: Free-text search string (e.g. "Shake Shack New York").

        Returns:
            List of :class:`PlaceSearchResult` objects (may be empty).
            Returns an empty list — rather than raising — on network errors
            or when no results are returned.

        Raises:
            PlacesAPIError: If the Places API returns a 4xx/5xx error response.
        """
        if not query or not query.strip():
            return []

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
        }
        body = {
            "textQuery": query.strip(),
            "includedType": "restaurant",
        }

        try:
            response = requests.post(
                _PLACES_TEXT_SEARCH_URL,
                headers=headers,
                json=body,
                timeout=10,
            )
        except requests.exceptions.Timeout:
            logger.warning("Google Places request timed out for query=%r", query)
            return []
        except requests.exceptions.RequestException as exc:
            logger.error("Google Places HTTP error for query=%r: %s", query, exc)
            return []

        if not response.ok:
            try:
                payload = response.json() if response.content else {}
                error_message = payload.get("error", {}).get("message") or response.text
            except ValueError:
                payload = {}
                error_message = response.text
            logger.error(
                "Google Places API returned HTTP %s for query=%r: %s",
                response.status_code,
                query,
                error_message,
            )
            raise PlacesAPIError(
                f"Google Places API error (HTTP {response.status_code}): {error_message}"
            )

        payload = response.json()
        results: list[PlaceSearchResult] = []
        for item in payload.get("places", []):
            place_id = item.get("id", "")
            name = item.get("displayName", {}).get("text", "")
            address = item.get("formattedAddress", "")
            if place_id and name:
                results.append(
                    PlaceSearchResult(
                        place_id=place_id,
                        name=name,
                        address=address,
                    )
                )

        return results
