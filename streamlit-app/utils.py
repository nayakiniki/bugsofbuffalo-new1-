"""
Bugs of Buffalo - Utility Functions
Updated for the specific dataset structure.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import json
import os
import io
import logging
from typing import Dict, Tuple, Optional, List, Union
import cv2
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreedClassifier:
    """
    A class to handle cattle breed classification using a trained TensorFlow model.
    Updated for the specific dataset structure.
    """
    
    def __init__(self, model_path: str, mapping_path: str, breed_types_path: str):
        """
        Initialize the Breed Classifier with breed type information.
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
            if not os.path.exists(breed_types_path):
                raise FileNotFoundError(f"Breed types file not found: {breed_types_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
            logger.info(f"Loading class mapping from {mapping_path}")
            with open(mapping_path, 'r') as f:
                class_mapping = json.load(f)
            
            self.class_mapping = {int(k): v for k, v in class_mapping.items()}
            self.class_names = list(self.class_mapping.values())
            logger.info(f"Loaded {len(self.class_names)} classes: {self.class_names}")
            
            logger.info(f"Loading breed types from {breed_types_path}")
            with open(breed_types_path, 'r') as f:
                self.breed_types = json.load(f)
            logger.info("Breed types loaded successfully")
            
            self.input_shape = self.model.input_shape[1:3]
            self.img_height, self.img_width = self.input_shape
            logger.info(f"Model input shape: {self.input_shape}")
            
        except Exception as e:
            logger.error(f"Error initializing BreedClassifier: {str(e)}")
            raise
    
    def preprocess_image(self, img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for model prediction.
        """
        try:
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise FileNotFoundError(f"Image file not found: {img}")
                pil_img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                pil_img = img.convert('RGB')
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert('RGB')
            else:
                raise ValueError("Unsupported image type.")
            
            pil_img = pil_img.resize((self.img_width, self.img_height))
            img_array = np.array(pil_img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_breed(self, img: Union[str, Image.Image, np.ndarray], 
                     top_k: int = 3) -> Tuple[List[str], List[float], List[str], np.ndarray]:
        """
        Predict the breed of cattle from an image.
        Returns breed names, confidences, animal types, and all predictions.
        """
        try:
            processed_img = self.preprocess_image(img)
            predictions = self.model.predict(processed_img, verbose=0)
            predictions = predictions[0]
            
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            top_breeds = [self.class_mapping[i] for i in top_indices]
            top_confidences = [float(predictions[i]) for i in top_indices]
            top_confidences = [conf * 100 for conf in top_confidences]
            top_animal_types = [self.breed_types[breed] for breed in top_breeds]
            
            return top_breeds, top_confidences, top_animal_types, predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_breed_info(self, breed_name: str) -> Dict[str, str]:
        """
        Get additional information about a breed.
        """
        animal_type = self.breed_types.get(breed_name, 'Unknown')
        
        breed_info = {
            'name': breed_name,
            'type': animal_type,
            'description': f"{animal_type} breed from India",
            'origin': 'India',
            'characteristics': 'Dairy/Draught/Dual-purpose breed',
            'average_weight': '300-600 kg' if animal_type == 'Cattle' else '400-800 kg',
            'lifespan': '15-20 years'
        }
        
        return breed_info
    
    def get_all_breeds_by_type(self) -> Dict[str, List[str]]:
        """
        Get all breeds organized by animal type.
        """
        breeds_by_type = {'Cattle': [], 'Buffalo': []}
        
        for breed, animal_type in self.breed_types.items():
            if animal_type in breeds_by_type:
                breeds_by_type[animal_type].append(breed)
        
        return breeds_by_type

class MultilingualTranslator:
    """
    A class to handle multilingual translation using Mistral AI's Mixtral 8x7B via OpenRouter.
    Updated with breed-specific information.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        """
        Initialize the multilingual translator.
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bugs-of-buffalo.streamlit.app",
            "X-Title": "Bugs of Buffalo - Cattle Breed Identification"
        }
        
        # Language mapping with detailed information
        self.language_map = {
            'english': {'code': 'en', 'name': 'English', 'native': 'English'},
            'hindi': {'code': 'hi', 'name': 'Hindi', 'native': 'हिन्दी'},
            'punjabi': {'code': 'pa', 'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
            'bengali': {'code': 'bn', 'name': 'Bengali', 'native': 'বাংলা'},
            'marathi': {'code': 'mr', 'name': 'Marathi', 'native': 'मराठी'},
            'odia': {'code': 'or', 'name': 'Odia', 'native': 'ଓଡ଼ିଆ'},
            'tamil': {'code': 'ta', 'name': 'Tamil', 'native': 'தமிழ்'},
            'telugu': {'code': 'te', 'name': 'Telugu', 'native': 'తెలుగు'},
            'gujarati': {'code': 'gu', 'name': 'Gujarati', 'native': 'ગુજરાતી'},
            'kannada': {'code': 'kn', 'name': 'Kannada', 'native': 'ಕನ್ನಡ'},
            'malayalam': {'code': 'ml', 'name': 'Malayalam', 'native': 'മലയാളം'}
        }
        
        # Breed-specific information for better translations
        self.breed_info = {
            # Buffalo breeds
            'Banni': {
                'characteristics': 'Adapted to desert conditions, good milk yield',
                'purpose': 'Milk production, drought resistant',
                'regions': 'Gujarat, Rajasthan'
            },
            'Jaffrabadi': {
                'characteristics': 'Large size, heavy body, strong',
                'purpose': 'Milk and draught',
                'regions': 'Gujarat'
            },
            'Mehsana': {
                'characteristics': 'Medium size, good milk yield',
                'purpose': 'Milk production',
                'regions': 'Gujarat'
            },
            'Murrah': {
                'characteristics': 'Best dairy buffalo, high milk fat',
                'purpose': 'Milk production',
                'regions': 'Haryana, Punjab, Delhi'
            },
            'Nagpuri': {
                'characteristics': 'Medium size, suited for hot climate',
                'purpose': 'Draught and milk',
                'regions': 'Maharashtra, Andhra Pradesh'
            },
            # Cattle breeds
            'Gir': {
                'characteristics': 'Distinctive curved horns, good milk yield',
                'purpose': 'Milk production',
                'regions': 'Gujarat, Rajasthan'
            },
            'Hallikar': {
                'characteristics': 'Medium size, known for draught capacity',
                'purpose': 'Draught purposes',
                'regions': 'Karnataka'
            },
            'Hariana': {
                'characteristics': 'Dual purpose, white/grey color',
                'purpose': 'Milk and draught',
                'regions': 'Haryana, Punjab, UP'
            },
            'Kankrej': {
                'characteristics': 'Large size, strong, drought resistant',
                'purpose': 'Draught and milk',
                'regions': 'Gujarat, Rajasthan'
            },
            'Khillar': {
                'characteristics': 'Medium size, grey-white color',
                'purpose': 'Draught purposes',
                'regions': 'Maharashtra, Karnataka'
            },
            'Ongole': {
                'characteristics': 'Large size, muscular, white color',
                'purpose': 'Draught and milk',
                'regions': 'Andhra Pradesh'
            },
            'Rathi': {
                'characteristics': 'Medium size, brown color, good temperament',
                'purpose': 'Milk production',
                'regions': 'Rajasthan'
            },
            'Red Sindhi': {
                'characteristics': 'Red color, heat tolerant, good milk yield',
                'purpose': 'Milk production',
                'regions': 'Sindh region, Punjab'
            },
            'Sahiwal': {
                'characteristics': 'Best indigenous dairy breed, loose skin',
                'purpose': 'Milk production',
                'regions': 'Punjab, Haryana'
            },
            'Tharparkar': {
                'characteristics': 'White/grey color, drought resistant',
                'purpose': 'Milk and draught',
                'regions': 'Rajasthan, Gujarat'
            }
        }
        
        logger.info("MultilingualTranslator initialized successfully")
    
    def translate_breed_info(self, breed_name: str, target_language: str, 
                           max_retries: int = 3) -> Dict[str, str]:
        """
        Translate breed information to the target language using Mistral AI.
        """
        if target_language not in self.language_map:
            logger.warning(f"Unsupported language: {target_language}. Using English.")
            target_language = 'english'
        
        lang_info = self.language_map[target_language]
        lang_name = lang_info.get('native', lang_info['name'])
        
        # Get breed-specific information
        breed_data = self.breed_info.get(breed_name, {})
        
        prompt = f"""You are a helpful agricultural assistant for Indian farmers. 
Translate the following cattle/buffalo breed information to {lang_name}.

Breed: {breed_name}
Animal Type: {'Buffalo' if breed_name in ['Banni', 'Jaffrabadi', 'Mehsana', 'Murrah', 'Nagpuri'] else 'Cattle'}

Key Information:
- Characteristics: {breed_data.get('characteristics', 'General Indian breed')}
- Primary Purpose: {breed_data.get('purpose', 'Milk/Draught')}
- Regions: {breed_data.get('regions', 'Various regions of India')}

Provide the translation in {lang_name} with this exact JSON structure:
{{
    "name": "translated breed name in {lang_name}",
    "description": "detailed description in {lang_name} (2-3 sentences)",
    "characteristics": "key characteristics in {lang_name}",
    "purpose": "primary purpose in {lang_name}",
    "regions": "common regions in {lang_name}",
    "animal_type": "translated animal type in {lang_name}"
}}

Make the translation natural and appropriate for farmers.
Do not include any additional text or explanations.
"""

        payload = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert agricultural translator. Always respond with valid JSON only, no additional text. Provide accurate translations suitable for farmers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 600
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Translating '{breed_name}' to {lang_name}")
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                content = response_data['choices'][0]['message']['content'].strip()
                
                # Clean the response
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                # Parse JSON response
                translated_info = json.loads(content)
                
                # Validate response structure
                required_keys = ['name', 'description', 'characteristics', 'purpose', 'regions', 'animal_type']
                if all(key in translated_info for key in required_keys):
                    logger.info(f"Successfully translated '{breed_name}' to {lang_name}")
                    return translated_info
                else:
                    raise ValueError("Missing required keys in translation response")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All translation attempts failed for '{breed_name}'")
                    return self.get_fallback_translation(breed_name, target_language)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode failed (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All translation attempts failed for '{breed_name}'")
                    return self.get_fallback_translation(breed_name, target_language)
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All translation attempts failed for '{breed_name}'")
                    return self.get_fallback_translation(breed_name, target_language)
        
        return self.get_fallback_translation(breed_name, target_language)
    
    def get_fallback_translation(self, breed_name: str, target_language: str) -> Dict[str, str]:
        """
        Provide fallback translation when API calls fail.
        """
        lang_info = self.language_map.get(target_language, self.language_map['english'])
        lang_name = lang_info.get('native', lang_info['name'])
        
        # Determine animal type
        animal_type = 'Buffalo' if breed_name in ['Banni', 'Jaffrabadi', 'Mehsana', 'Murrah', 'Nagpuri'] else 'Cattle'
        
        return {
            "name": breed_name,
            "description": f"{animal_type} breed from India. Translation not available in {lang_name}.",
            "characteristics": "Characteristics information not available",
            "purpose": "Purpose information not available",
            "regions": "Region information not available",
            "animal_type": animal_type
        }
    
    def get_supported_languages(self) -> Dict[str, Dict]:
        """
        Get information about all supported languages.
        """
        return self.language_map
    
    def get_language_name(self, language_code: str, native: bool = False) -> str:
        """
        Get the display name for a language code.
        """
        lang_info = self.language_map.get(language_code)
        if not lang_info:
            return language_code
        
        if native and 'native' in lang_info:
            return lang_info['native']
        return lang_info['name']
