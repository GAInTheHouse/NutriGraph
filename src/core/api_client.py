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
        Generate nutrition profile for a dish (Restaurant workflow).
        
        In production, this will use the backend to calculate precise nutrition
        based on the provided ingredient list.
        
        Args:
            dish: The dish with ingredients to calculate nutrition for.
        
        Returns:
            NutritionEstimate with calculated nutrition values.
        """
        if self._mock_mode:
            logger.debug(f"Mock mode: generating profile for '{dish.name}' with {len(dish.ingredients)} ingredients")
            # For restaurant builder, use ingredient count to influence estimate
            estimate = NutritionEstimate.mock_from_dish(dish)
            
            # Adjust based on ingredient count (more ingredients = higher calories typically)
            multiplier = 1 + (len(dish.ingredients) * 0.05)
            return NutritionEstimate(
                calories=round(estimate.calories * multiplier, 1),
                protein_g=round(estimate.protein_g * multiplier, 1),
                carbs_g=round(estimate.carbs_g * multiplier, 1),
                fat_g=round(estimate.fat_g * multiplier, 1),
                confidence=min(0.95, estimate.confidence + 0.05)  # Higher confidence with explicit ingredients
            )
        
        # TODO: Implement actual API call
        # response = requests.post(
        #     f"{self.base_url}/api/v1/builder/generate",
        #     json=dish.model_dump()
        # )
        # response.raise_for_status()
        # return NutritionEstimate(**response.json())
        raise NotImplementedError("Backend API not yet implemented")
    
    def analyze_dish_image(self, image_bytes: bytes, filename: str) -> DishAnalysisResponse:
        """
        Send a dish photo to the Gemini vision pipeline and retrieve its nutritional breakdown.

        Makes a multipart POST to ``/api/v1/analyze-dish``.  The backend is expected to
        return a JSON body that maps directly onto :class:`DishAnalysisResponse`.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename (used to infer MIME type on the server side).

        Returns:
            DishAnalysisResponse with totals and per-ingredient macros.

        Raises:
            NutriGraphAPIError: If the backend is unreachable, times out, or returns a
                non-2xx status code.
        """
        url = f"{self.base_url}/api/v1/analyze-dish"
        try:
            response = requests.post(
                url,
                files={"file": (filename, image_bytes, "image/jpeg")},
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
            List of place dicts.  Empty list when no matches are found or when
            the backend Places integration is not configured.

        Raises:
            NutriGraphAPIError: If the backend is unreachable or returns an
                unexpected HTTP error.
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
