"""
API client for NutriGraph backend service.

Currently implements mock responses. Will be connected to FastAPI backend later.
"""
from typing import Optional
import logging

from .models import Dish, NutritionEstimate

logger = logging.getLogger(__name__)


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
    
    def health_check(self) -> bool:
        """
        Check if the backend API is available.
        
        Returns:
            True if backend is healthy, False otherwise.
        """
        if self._mock_mode:
            return True
        
        # TODO: Implement actual health check
        # try:
        #     response = requests.get(f"{self.base_url}/health", timeout=5)
        #     return response.status_code == 200
        # except requests.RequestException:
        #     return False
        return False
