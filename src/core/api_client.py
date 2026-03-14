"""
API client for NutriGraph backend service.

Mock methods (estimate_nutrition, builder_generate_profile) remain for the text-search
workflow. analyze_dish_image targets the real FastAPI image pipeline.
start_dish_conversation / continue_dish_conversation drive the agentic loop.
"""
from typing import Optional
import logging

import requests

from .models import Dish, NutritionEstimate, DishAnalysisResponse, ConversationState

logger = logging.getLogger(__name__)


class NutriGraphAPIError(Exception):
    """Raised when the NutriGraph backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class NutriGraphClient:
    """
    Client for communicating with the NutriGraph backend API.
    
    Currently returns mock data. In production, this will make HTTP requests
    to the FastAPI backend for RAG-powered nutrition estimation.
    """
    
    def __init__(self, base_url: str):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the NutriGraph backend API.
        """
        self.base_url = base_url.rstrip("/")
        self._mock_mode = True  # Will be False when backend is available
        logger.info(f"NutriGraphClient initialized with base_url: {self.base_url}")
    
    def estimate_nutrition(self, dish: Dish) -> NutritionEstimate:
        """
        Estimate nutrition for a dish (Diner workflow).
        
        In production, this will call the backend's RAG pipeline to estimate
        nutrition based on dish name and optional restaurant context.
        
        Args:
            dish: The dish to estimate nutrition for.
        
        Returns:
            NutritionEstimate with calorie and macro breakdown.
        """
        if self._mock_mode:
            logger.debug(f"Mock mode: generating estimate for '{dish.name}'")
            return NutritionEstimate.mock_from_dish(dish)
        
        # TODO: Implement actual API call
        # response = requests.post(
        #     f"{self.base_url}/api/v1/estimate",
        #     json=dish.model_dump()
        # )
        # response.raise_for_status()
        # return NutritionEstimate(**response.json())
        raise NotImplementedError("Backend API not yet implemented")
    
    def builder_generate_profile(self, dish: Dish) -> NutritionEstimate:
        """
        Generate a nutrition profile for a dish by sending its explicit ingredient
        list to the backend builder endpoint.

        Makes a POST to ``/api/v1/builder/generate``.  The backend looks up each
        ingredient in the ChromaDB nutritional index, scales the per-100g values by
        the ingredient's actual quantity and unit, sums to dish-level totals, and
        returns a :class:`NutritionEstimate`.  No LLM is involved — the result is
        deterministic and based on real nutritional data.

        Args:
            dish: The dish with a fully populated ingredient list (name, quantity, unit).

        Returns:
            NutritionEstimate with calculated macro totals and average confidence.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, times out, or returns
                a non-2xx status code.
        """
        url = f"{self.base_url}/api/v1/builder/generate"
        try:
            response = requests.post(url, json=dish.model_dump(), timeout=30)
            response.raise_for_status()
            return NutritionEstimate(**response.json())

        except requests.exceptions.ConnectionError as exc:
            logger.error("Backend unreachable at %s: %s", url, exc)
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend. "
                "Please verify the server is running and the URL is correct."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Builder request to %s timed out.", url)
            raise NutriGraphAPIError(
                "The builder request timed out. Please try again."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.error("Backend returned HTTP %s for %s: %s", status_code, url, exc)
            raise NutriGraphAPIError(
                f"Builder endpoint returned an error (HTTP {status_code}). Please try again later.",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error calling builder/generate endpoint.")
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc
    
    def analyze_dish_image(
        self,
        image_bytes: bytes,
        filename: str,
        restaurant_context: Optional[str] = None,
        dish_name: Optional[str] = None,
        place_id: Optional[str] = None,
    ) -> DishAnalysisResponse:
        """
        Send a dish photo to the Gemini vision pipeline and retrieve its nutritional breakdown.

        Makes a multipart POST to ``/api/v1/analyze-dish``.  When ``restaurant_context``
        is provided it is forwarded as a form field so the backend can enrich the Gemini
        prompt with establishment-specific knowledge.

        When ``dish_name`` and/or ``place_id`` are provided, the backend checks the
        database first.  A restaurant-verified record is returned immediately (fast path);
        past diner records are injected into the Gemini prompt as coaching context.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename (used to infer MIME type on the server side).
            restaurant_context: Optional restaurant name (or ``"Home Cooked"``) selected
                by the user during the upload step.
            dish_name: Optional dish-name hint typed by the user.  Enables the DB cache
                lookup on the backend.
            place_id: Google Places place_id of the selected restaurant.

        Returns:
            DishAnalysisResponse with totals, per-ingredient macros, ``is_cached``,
            and ``data_source`` fields.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, times out, or returns a
                non-2xx status code.
        """
        url = f"{self.base_url}/api/v1/analyze-dish"
        data: dict = {}
        if restaurant_context:
            data["restaurant_context"] = restaurant_context
        if dish_name:
            data["dish_name"] = dish_name
        if place_id:
            data["restaurant_place_id"] = place_id
        try:
            response = requests.post(
                url,
                files={"file": (filename, image_bytes, "image/jpeg")},
                data=data or None,
                timeout=60,
            )
            response.raise_for_status()
            return DishAnalysisResponse(**response.json())

        except requests.exceptions.ConnectionError as exc:
            logger.error("Backend unreachable at %s: %s", url, exc)
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend. "
                "Please verify the server is running and the URL is correct."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Request to %s timed out.", url)
            raise NutriGraphAPIError(
                "The request timed out. The backend may be overloaded — please try again."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.error("Backend returned HTTP %s for %s: %s", status_code, url, exc)
            raise NutriGraphAPIError(
                f"The backend returned an error (HTTP {status_code}). Please try again later.",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error calling analyze-dish endpoint.")
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def save_dish_result(
        self,
        analysis: DishAnalysisResponse,
        place_id: Optional[str] = None,
    ) -> None:
        """
        Explicitly persist a reviewed nutritional analysis as a diner record.

        POSTs to ``/api/v1/diner/save-dish``.  This is called only when the user
        clicks "Save Results" or "Save Refined Results" in the Diner UI — never
        triggered automatically.

        Args:
            analysis: The :class:`DishAnalysisResponse` to save (may be the one-shot
                result or the agent-refined result).
            place_id: Google Places place_id if the dish was tagged to a restaurant.

        Raises:
            NutriGraphAPIError: If the backend is unreachable or returns an error.
        """
        url = f"{self.base_url}/api/v1/diner/save-dish"
        payload = {
            "analysis": analysis.model_dump(),
            "place_id": place_id,
        }
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()

        except requests.exceptions.ConnectionError as exc:
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise NutriGraphAPIError("The save request timed out.") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            raise NutriGraphAPIError(
                f"Save failed (HTTP {status_code}).", status_code=status_code
            ) from exc
        except Exception as exc:
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def publish_dish(
        self,
        dish_name: str,
        place_id: str,
        calories: float,
        protein: float,
        carbs: float,
        fat: float,
        ingredients: Optional[list] = None,
        serving_size: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Publish verified dish macros to the global catalog as a restaurant owner.

        POSTs to ``/api/v1/restaurant/publish-dish``.  Once published, future diner
        requests for the same ``(dish_name, place_id)`` pair are served from this
        record without calling the LLM.

        Args:
            dish_name:    Canonical name of the dish as it appears on the menu.
            place_id:     Google Places place_id of the restaurant.
            calories:     Total calories (kcal) per serving.
            protein:      Total protein (g) per serving.
            carbs:        Total carbohydrates (g) per serving.
            fat:          Total fat (g) per serving.
            ingredients:  Optional list of per-ingredient dicts for the full breakdown.
            serving_size: Optional human-readable serving size string.
            confidence:   Optional average retrieval confidence (0–1).

        Raises:
            NutriGraphAPIError: If the backend is unreachable or returns an error.
        """
        url = f"{self.base_url}/api/v1/restaurant/publish-dish"
        payload = {
            "dish_name": dish_name,
            "place_id": place_id,
            "calories": calories,
            "protein": protein,
            "carbs": carbs,
            "fat": fat,
            "ingredients": ingredients or [],
            "serving_size": serving_size,
            "confidence": confidence,
        }
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()

        except requests.exceptions.ConnectionError as exc:
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise NutriGraphAPIError("The publish request timed out.") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            raise NutriGraphAPIError(
                f"Publish failed (HTTP {status_code}).", status_code=status_code
            ) from exc
        except Exception as exc:
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def get_restaurant_dishes(self, place_id: str) -> list[dict]:
        """
        Fetch all restaurant-verified dishes for a given place_id from the DB.

        GETs ``/api/v1/restaurant/dishes``.  Called when the Restaurant UI loads
        a profile so the catalog is populated from persisted records rather than
        starting empty every session.

        Args:
            place_id: Google Places place_id of the restaurant.

        Returns:
            List of dish dicts with keys: name, serving_size, ingredient_count,
            calories, protein_g, carbs_g, fat_g, confidence.

        Raises:
            NutriGraphAPIError: If the backend is unreachable or returns an error.
        """
        url = f"{self.base_url}/api/v1/restaurant/dishes"
        try:
            response = requests.get(url, params={"place_id": place_id}, timeout=10)
            response.raise_for_status()
            return response.json().get("dishes", [])

        except requests.exceptions.ConnectionError as exc:
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise NutriGraphAPIError("The dishes request timed out.") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            raise NutriGraphAPIError(
                f"Failed to load catalog (HTTP {status_code}).", status_code=status_code
            ) from exc
        except Exception as exc:
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def start_dish_conversation(self, initial_input: dict) -> ConversationState:
        """
        Begin an agentic clarification conversation for a dish.

        POSTs ``initial_input`` to ``/api/v1/agent/start``.  The backend creates a
        LangGraph session, runs the first retrieval + decision step, and returns a
        :class:`ConversationState` whose ``history`` will contain either an initial
        clarifying question (type ``"question"``) or an immediate final result (type
        ``"final_result"``).

        Args:
            initial_input: A dict that must include ``dish_name`` and may optionally
                include ``restaurant_name`` and ``image_analysis_id`` (the UUID
                returned by a prior ``analyze_dish_image`` call).

        Returns:
            :class:`ConversationState` with the first agent turn already appended.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, times out, or returns
                a non-2xx HTTP status code.
        """
        url = f"{self.base_url}/api/v1/agent/start"
        try:
            response = requests.post(url, json=initial_input, timeout=60)
            response.raise_for_status()
            return ConversationState(**response.json())

        except requests.exceptions.ConnectionError as exc:
            logger.error("Backend unreachable at %s: %s", url, exc)
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend. "
                "Please verify the server is running and the URL is correct."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Request to %s timed out.", url)
            raise NutriGraphAPIError(
                "The request timed out. The backend may be overloaded — please try again."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.error("Backend returned HTTP %s for %s: %s", status_code, url, exc)
            raise NutriGraphAPIError(
                f"The backend returned an error (HTTP {status_code}). Please try again later.",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error calling agent/start endpoint.")
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def continue_dish_conversation(
        self, dish_id: str, user_message: str
    ) -> ConversationState:
        """
        Send a user reply to the agent and receive the updated conversation state.

        POSTs ``{"dish_id": dish_id, "user_message": user_message}`` to
        ``/api/v1/agent/continue``.  The backend resumes the LangGraph session
        identified by ``dish_id``, appends the user turn, runs the next retrieval /
        decision cycle, and returns the updated :class:`ConversationState`.

        If the agent has gathered sufficient information the returned state will have
        ``final_result`` populated and the last history turn will be of type
        ``"final_result"``.

        Args:
            dish_id:      The opaque session identifier from the :class:`ConversationState`
                          returned by :meth:`start_dish_conversation`.
            user_message: The user's plain-text answer to the agent's most recent question.

        Returns:
            Updated :class:`ConversationState` with all new turns appended.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, times out, or returns
                a non-2xx HTTP status code.
        """
        url = f"{self.base_url}/api/v1/agent/continue"
        payload = {"dish_id": dish_id, "user_message": user_message}
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return ConversationState(**response.json())

        except requests.exceptions.ConnectionError as exc:
            logger.error("Backend unreachable at %s: %s", url, exc)
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend. "
                "Please verify the server is running and the URL is correct."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Request to %s timed out.", url)
            raise NutriGraphAPIError(
                "The request timed out. The backend may be overloaded — please try again."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.error("Backend returned HTTP %s for %s: %s", status_code, url, exc)
            raise NutriGraphAPIError(
                f"The backend returned an error (HTTP {status_code}). Please try again later.",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error calling agent/continue endpoint.")
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def search_restaurants(self, query: str) -> list[dict]:
        """
        Search for real-world restaurant establishments via the backend Places proxy.

        GETs ``/api/v1/places/search?query={query}`` and returns the ``results``
        list.  Each element is a dict with keys ``place_id``, ``name``, and
        ``address`` corresponding to :class:`~src.core.models.PlaceSearchResult`.

        Args:
            query: Restaurant name or free-text search phrase.

        Returns:
            List of place dicts.  Empty list when no matches are found.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, returns an
                unexpected HTTP error, or returns HTTP 503 because the Google
                Places API key has not been configured on the server.
        """
        url = f"{self.base_url}/api/v1/places/search"
        try:
            response = requests.get(url, params={"query": query}, timeout=15)
            response.raise_for_status()
            return response.json().get("results", [])

        except requests.exceptions.ConnectionError as exc:
            logger.error("Backend unreachable at %s: %s", url, exc)
            raise NutriGraphAPIError(
                "Could not connect to the NutriGraph backend. "
                "Please verify the server is running and the URL is correct."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Request to %s timed out.", url)
            raise NutriGraphAPIError(
                "The Places search request timed out. Please try again."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 503:
                raise NutriGraphAPIError(
                    "Restaurant search is not available: the Google Places API key "
                    "has not been configured on the server.",
                    status_code=status_code,
                ) from exc
            logger.error("Backend returned HTTP %s for %s: %s", status_code, url, exc)
            raise NutriGraphAPIError(
                f"The backend returned an error (HTTP {status_code}). Please try again later.",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected error calling places/search endpoint.")
            raise NutriGraphAPIError(f"An unexpected error occurred: {exc}") from exc

    def health_check(self) -> bool:
        """
        Check if the backend API is available.

        Returns:
            True if backend is healthy, False otherwise.
        """
        if self._mock_mode:
            return True

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
