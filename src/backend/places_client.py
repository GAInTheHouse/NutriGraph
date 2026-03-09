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

_PLACES_TEXT_SEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/textsearch/json"
)


class PlacesAPIError(Exception):
    """Raised when the Google Places API returns an unexpected status."""


class GooglePlacesClient:
    """
    Thin wrapper around the Google Places Text Search API.

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

        Calls the Places Text Search endpoint with ``type=restaurant`` so
        results are filtered to food/dining establishments.

        Args:
            query: Free-text search string (e.g. "Shake Shack New York").

        Returns:
            List of :class:`PlaceSearchResult` objects (may be empty).
            Returns an empty list — rather than raising — on network errors
            or when the API signals ``ZERO_RESULTS``.

        Raises:
            PlacesAPIError: If the Places API returns a terminal error status
                (``REQUEST_DENIED``, ``INVALID_REQUEST``, ``OVER_QUERY_LIMIT``).
        """
        if not query or not query.strip():
            return []

        params = {
            "query": query.strip(),
            "type": "restaurant",
            "key": self._api_key,
        }

        try:
            response = requests.get(
                _PLACES_TEXT_SEARCH_URL,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning("Google Places request timed out for query=%r", query)
            return []
        except requests.exceptions.RequestException as exc:
            logger.error("Google Places HTTP error for query=%r: %s", query, exc)
            return []

        payload = response.json()
        status: str = payload.get("status", "")

        if status == "ZERO_RESULTS":
            return []

        if status not in ("OK",):
            error_message = payload.get("error_message", "No details provided.")
            logger.error(
                "Google Places API returned status=%r for query=%r: %s",
                status,
                query,
                error_message,
            )
            raise PlacesAPIError(
                f"Google Places API error ({status}): {error_message}"
            )

        results: list[PlaceSearchResult] = []
        for item in payload.get("results", []):
            place_id = item.get("place_id", "")
            name = item.get("name", "")
            address = item.get("formatted_address", "")
            if place_id and name:
                results.append(
                    PlaceSearchResult(
                        place_id=place_id,
                        name=name,
                        address=address,
                    )
                )

        return results
