"""
API client for NutriGraph backend service.

Mock methods (estimate_nutrition, builder_generate_profile) remain for the text-search
workflow. analyze_dish_image targets the real FastAPI image pipeline.
"""
from typing import Optional
import logging

import requests

from .models import Dish, NutritionEstimate, DishAnalysisResponse

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
