import json
import logging
import os
from typing import Optional, Tuple

import openai

# Set up logging
logger = logging.getLogger(__name__)


class PostProcessor:
    """
    A class for post-processing extracted material science data.
    Uses GPT-5-mini to map extracted properties to standard names from properties.json.
    
    This approach uses LLM intelligence to handle:
    - Property name variations
    - Symbol-based disambiguation
    - Context-aware matching
    """

    def __init__(
        self,
        properties_file: str,
        extracted_data_file: str = None,
        llm_client: Optional[openai.OpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        gpt_model: Optional[str] = None,
    ):
        """
        Initializes the PostProcessor with GPT-based matching.

        Args:
            properties_file (str): Path to the JSON file containing allowed properties.
            extracted_data_file (str, optional): Path to the CSV file containing extracted property data.
            llm_client (openai.OpenAI, optional): OpenAI client. Created if not provided.
            api_key (str, optional): LLM API key. Required if llm_client not provided.
            base_url (str, optional): OpenAI-compatible base URL.
            gpt_model (str, optional): Model to use for matching.
        """
        self.properties_file = properties_file
        self.extracted_data_file = extracted_data_file
        self.gpt_model = gpt_model or os.getenv("LLM_MODEL", "gpt-5-mini")
        
        # Load properties and create lookup
        self.property_lookup = self.load_properties()
        
        # Initialize OpenAI client
        if llm_client is None and api_key is None:
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No LLM API key provided. Set LLM_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        if llm_client is None and base_url is None:
            base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.llm_client = llm_client or openai.OpenAI(**client_kwargs)
        
        # Statistics tracking
        self.match_stats = {
            "matched": 0,
            "no_match": 0,
            "total": 0,
        }

    def load_properties(self) -> dict:
        """
        Loads properties from the JSON file and prepares a lookup dictionary.

        Returns:
            dict: A dictionary where keys are lowercase property names, and values are
                  (domain, category, standard property).
        """
        with open(self.properties_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        lookup = {}
        for domain, categories in data.items():
            for category, properties in categories.items():
                for prop in properties:
                    lookup[prop.lower()] = (domain, category, prop)
        
        logger.info(f"Loaded {len(lookup)} standard properties from {self.properties_file}")
        return lookup

    def gpt_match(
        self, property_name: str, property_symbol: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Use GPT to match property to standard name using symbol and context.
        
        Handles:
        - Property name variations ("figure of merit ZT" → "thermoelectric figure of merit")
        - Symbol-based disambiguation (κ → thermal conductivity vs electrical)
        - Context-dependent properties (Tc → Curie temp vs critical temp)

        Args:
            property_name (str): The extracted property name.
            property_symbol (str, optional): The property symbol (e.g., "Tg", "ZT", "κ").

        Returns:
            tuple: (domain, category, standard_property_name) or (None, None, None)
        """
        if not self.llm_client:
            logger.warning("GPT fallback requested but no OpenAI client available")
            return None, None, None
        
        # Build list of all available standard properties
        all_standard_properties = list(set(
            std_prop for _, _, std_prop in self.property_lookup.values()
        ))
        
        # Create prompt for GPT
        prompt = f"""You are a materials science expert. Match the extracted property to the most appropriate standard property name from the provided list.

EXTRACTED PROPERTY:
- property_name: "{property_name}"
- property_symbol: "{property_symbol if property_symbol else 'Not provided'}"

AVAILABLE STANDARD PROPERTIES (from properties.json):
{json.dumps(all_standard_properties, indent=2)}

INSTRUCTIONS:
1. Find the best matching standard property name from the list above
2. Use the property_symbol to disambiguate when needed:
   - "κ" → thermal conductivity (NOT electrical)
   - "σ" → electrical conductivity (NOT stress, if unit is S/m)
   - "ZT" → thermoelectric figure of merit
   - "Tc" → check context (Curie temp for magnetics, critical temp for superconductors)
3. Handle variations:
   - "figure of merit ZT" → "thermoelectric figure of merit"
   - "glass transition temp" → "temperature for structural transition" or closest match
   - "Young's modulus" → "elastic stiffness coefficient" or closest match
4. Return ONLY properties that exist in the provided list
5. If no good match exists (confidence < 70%), return null for all fields

Return JSON format:
{{
    "standard_property_name": "exact name from list or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

        try:
            # Note: gpt-5-mini doesn't support custom temperature, only default (1.0)
            # For other models like gpt-4o, you can set temperature
            response = self.llm_client.chat.completions.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                # temperature=0.1,  # Commented out - gpt-5-mini only supports default
            )
            
            result = json.loads(response.choices[0].message.content)
            std_property = result.get("standard_property_name")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            logger.debug(
                f"GPT match: '{property_name}' → '{std_property}' "
                f"(confidence: {confidence:.2f}, reasoning: {reasoning})"
            )
            
            # Find domain and category for the matched property
            if std_property and confidence >= 0.7:
                # Search for this standard property in our lookup
                for prop_key, (domain, category, original_std_prop) in self.property_lookup.items():
                    if original_std_prop.lower() == std_property.lower():
                        return domain, category, original_std_prop
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"GPT matching failed for '{property_name}': {e}")
            return None, None, None

    def find_closest_property(
        self, property_name: str, property_symbol: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Find the closest matching property using GPT.
        
        This is the main method for property matching.

        Args:
            property_name (str): The extracted property name.
            property_symbol (str, optional): The property symbol for context.

        Returns:
            tuple: (domain, category, standard_property_name) or (None, None, None)
        """
        self.match_stats["total"] += 1
        
        domain, category, std_property = self.gpt_match(property_name, property_symbol)
        
        if std_property:
            self.match_stats["matched"] += 1
            logger.info(f"Matched: '{property_name}' → '{std_property}'")
            return domain, category, std_property
        else:
            self.match_stats["no_match"] += 1
            logger.warning(f"No match found for '{property_name}'")
            return None, None, None

    def update_extracted_json(self, extracted_result):
        """
        Updates the extracted JSON data by adding standardization fields to each property.
        
        Adds these fields to each property:
        - standard_property_name: Standardized name from properties.json
        - domain: Materials science domain
        - category: Property category

        Args:
            extracted_result (list): The extracted result (from JSONExtractor.extract).

        Returns:
            list: The updated extracted result with standardization fields.
        """
        # Reset statistics
        self.match_stats = {
            "matched": 0,
            "no_match": 0,
            "total": 0,
        }
        
        # Handle both dict and Pydantic model structures
        data = extracted_result[0]["data"]
        compositions = data.get("compositions") if isinstance(data, dict) else data.compositions
        
        # Process each composition
        for composition in compositions:
            # Get properties list (dict or Pydantic)
            properties = (
                composition.get("properties_of_composition") 
                if isinstance(composition, dict) 
                else composition.properties_of_composition
            )
            
            for prop in properties:
                # Get property name and symbol (works for both dict and Pydantic)
                property_name = prop.get("property_name") if isinstance(prop, dict) else prop.property_name
                property_symbol = prop.get("property_symbol") if isinstance(prop, dict) else getattr(prop, "property_symbol", None)
                
                # Apply GPT matching
                domain, category, std_property = (
                    self.find_closest_property(property_name, property_symbol)
                )
                
                # Add new fields to property object (works for both dict and Pydantic)
                if isinstance(prop, dict):
                    prop["standard_property_name"] = std_property
                    prop["domain"] = domain
                    prop["category"] = category
                else:
                    prop.standard_property_name = std_property
                    prop.domain = domain
                    prop.category = category

        # Print statistics
        self._print_match_stats()
        
        return extracted_result

    def _print_match_stats(self):
        """Print matching statistics."""
        total = self.match_stats["total"]
        if total == 0:
            return
        
        matched_pct = (self.match_stats["matched"] / total) * 100
        no_match_pct = (self.match_stats["no_match"] / total) * 100
        
        logger.info("=" * 60)
        logger.info("PROPERTY MATCHING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total properties processed: {total}")
        logger.info(
            f"Successfully matched: {self.match_stats['matched']} ({matched_pct:.1f}%)"
        )
        logger.info(
            f"No match found: {self.match_stats['no_match']} ({no_match_pct:.1f}%)"
        )
        logger.info("=" * 60)
