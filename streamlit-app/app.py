"""
Bugs of Buffalo - Streamlit Web Application
Main application for cattle breed identification with multilingual support.
"""

import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import BreedClassifier, MultilingualTranslator
except ImportError as e:
    st.error(f"Import error: {e}. Please make sure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🐃 Bugs of Buffalo - Cattle Breed Identification",
    page_icon="🐃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .breed-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1c6b8f;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'classifier_loaded' not in st.session_state:
    st.session_state.classifier_loaded = False
if 'translator_loaded' not in st.session_state:
    st.session_state.translator_loaded = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

@st.cache_resource
def load_classifier():
    """Load the breed classification model with caching."""
    try:
        # Use relative paths for Streamlit Cloud deployment
        model_path = os.path.join(os.path.dirname(__file__), 'saved_model', 'bugs_of_buffalo_model.h5')
        mapping_path = os.path.join(os.path.dirname(__file__), 'saved_model', 'class_mapping.json')
        
        # Fallback: check if files exist in parent directory (for local development)
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_model', 'bugs_of_buffalo_model.h5')
        if not os.path.exists(mapping_path):
            mapping_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_model', 'class_mapping.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")
        
        classifier = BreedClassifier(model_path, mapping_path)
        st.session_state.classifier_loaded = True
        return classifier
        
    except Exception as e:
        st.error(f"Error loading classifier: {str(e)}")
        st.session_state.classifier_loaded = False
        return None

@st.cache_resource
def load_translator():
    """Load the multilingual translator with caching."""
    try:
        # Get API key from secrets or environment variable
        api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
        
        if not api_key:
            st.warning("OpenRouter API key not found. Translation features will be limited.")
            return None
        
        translator = MultilingualTranslator(api_key)
        st.session_state.translator_loaded = True
        return translator
        
    except Exception as e:
        st.error(f"Error loading translator: {str(e)}")
        st.session_state.translator_loaded = False
        return None

def display_language_selector():
    """Display language selection sidebar."""
    st.sidebar.markdown("## 🌍 Language Settings")
    
    languages = {
        "English": "english",
        "हिन्दी (Hindi)": "hindi",
        "ਪੰਜਾਬੀ (Punjabi)": "punjabi",
        "বাংলা (Bengali)": "bengali",
        "मराठी (Marathi)": "marathi",
        "ଓଡ଼ିଆ (Odia)": "odia",
        "தமிழ் (Tamil)": "tamil",
        "తెలుగు (Telugu)": "telugu",
        "ગુજરાતી (Gujarati)": "gujarati",
        "ಕನ್ನಡ (Kannada)": "kannada",
        "മലയാളം (Malayalam)": "malayalam"
    }
    
    selected_language = st.sidebar.selectbox(
        "Choose Language",
        list(languages.keys()),
        index=0
    )
    
    return languages[selected_language], selected_language

def display_about_section():
    """Display about information in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ About")
    
    st.sidebar.markdown("""
    **Bugs of Buffalo** is an AI-powered tool designed to help farmers and field workers 
    accurately identify cattle and buffalo breeds across India.
    
    ### Features:
    - 🐄 Image-based breed identification
    - 🌍 Multilingual support
    - 📱 Mobile-friendly interface
    - 🔍 High accuracy AI model
    - 🎯 Real-time predictions
    
    ### Supported Breeds:
    - Gir, Sahiwal, Murrah, Jaffrabadi
    - Kankrej, Hallikar, Tharparkar
    - Red Sindhi, Ongole, Hariana
    - And many more...
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 Support")
    st.sidebar.markdown("""
    For support or questions:
    - 📧 Email: support@bugsofbuffalo.com
    - 🌐 Website: [bugsofbuffalo.com](https://bugsofbuffalo.com)
    - 💬 GitHub: [Report Issues](https://github.com/your-username/bugs-of-buffalo/issues)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 License")
    st.sidebar.markdown("""
    MIT License - Open source project
    [View License](https://github.com/your-username/bugs-of-buffalo/blob/main/LICENSE)
    """)

def display_upload_section():
    """Display image upload section."""
    st.markdown("### 📸 Upload Cattle/Buffalo Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload a clear image of cattle or buffalo for breed identification",
            key="file_uploader"
        )
    
    with col2:
        st.markdown("""
        **📝 Tips for best results:**
        - Use clear, well-lit images
        - Capture side view of the animal
        - Avoid blurry or distant shots
        - Ensure good contrast with background
        """)
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.uploaded_image = tmp_file.name
            
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return False
    
    return False

def display_prediction_results(predicted_breeds, confidences, target_language, language_label):
    """Display prediction results with translations."""
    st.markdown("## 🔍 Prediction Results")
    
    if not predicted_breeds or not confidences:
        st.warning("No predictions available. Please upload an image first.")
        return
    
    # Display top prediction
    top_breed = predicted_breeds[0]
    top_confidence = confidences[0]
    
    st.markdown(f"""
    <div class="success-box">
        <h3>🎯 Top Prediction: {top_breed}</h3>
        <p><strong>Confidence:</strong> {top_confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display alternative predictions
    if len(predicted_breeds) > 1:
        st.markdown("### 🔄 Alternative Predictions")
        for i, (breed, confidence) in enumerate(zip(predicted_breeds[1:], confidences[1:]), 2):
            st.markdown(f"""
            <div class="breed-card">
                <strong>#{i}: {breed}</strong> - {confidence:.2f}% confidence
            </div>
            """, unsafe_allow_html=True)
    
    # Display translated information
    if st.session_state.translator_loaded:
        with st.spinner(f"🌍 Translating information to {language_label}..."):
            try:
                translated_info = st.session_state.translator.translate_breed_info(
                    top_breed, target_language
                )
                
                st.markdown("### 📖 Breed Information")
                st.markdown(f"""
                <div class="info-box">
                    <h4>{translated_info['name']}</h4>
                    <p><strong>Description:</strong> {translated_info['description']}</p>
                    <p><strong>Characteristics:</strong> {translated_info['characteristics']}</p>
                    <p><strong>Primary Purpose:</strong> {translated_info['purpose']}</p>
                    <p><strong>Common Regions:</strong> {translated_info['regions']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Translation service temporarily unavailable. Error: {str(e)}")
    else:
        st.warning("Translation features are currently unavailable. Using English information.")
        
        # Display basic English information
        breed_info = st.session_state.classifier.get_breed_info(top_breed)
        st.markdown("### 📖 Breed Information (English)")
        st.markdown(f"""
        <div class="info-box">
            <h4>{breed_info['name']}</h4>
            <p><strong>Description:</strong> {breed_info['description']}</p>
            <p><strong>Origin:</strong> {breed_info['origin']}</p>
            <p><strong>Characteristics:</strong> {breed_info['characteristics']}</p>
            <p><strong>Average Weight:</strong> {breed_info['average_weight']}</p>
            <p><strong>Lifespan:</strong> {breed_info['lifespan']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🐃 Bugs of Buffalo</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Cattle Breed Identification System")
    st.markdown("---")
    
    # Load resources
    if st.session_state.classifier is None:
        with st.spinner("🚀 Loading AI model..."):
            st.session_state.classifier = load_classifier()
    
    if st.session_state.translator is None:
        st.session_state.translator = load_translator()
    
    # Sidebar
    with st.sidebar:
        target_language, language_label = display_language_selector()
        display_about_section()
    
    # Main content
    if not st.session_state.classifier_loaded:
        st.error("""
        ❌ AI model could not be loaded. Please ensure:
        1. The model file exists in the saved_model directory
        2. All dependencies are installed
        3. You have sufficient permissions
        """)
        return
    
    # Display upload section
    image_uploaded = display_upload_section()
    
    if image_uploaded and st.session_state.uploaded_image:
        # Make prediction when button is clicked
        if st.button("🔍 Identify Breed", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing image..."):
                try:
                    predicted_breeds, confidences, _ = st.session_state.classifier.predict_breed(
                        st.session_state.uploaded_image, top_k=3
                    )
                    
                    # Store prediction in session state
                    st.session_state.last_prediction = {
                        'breeds': predicted_breeds,
                        'confidences': confidences,
                        'language': target_language,
                        'language_label': language_label
                    }
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    return
            
            # Display results
            if st.session_state.last_prediction:
                display_prediction_results(
                    st.session_state.last_prediction['breeds'],
                    st.session_state.last_prediction['confidences'],
                    st.session_state.last_prediction['language'],
                    st.session_state.last_prediction['language_label']
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Built with ❤️ for Smart India Hackathon 2024 | 🐃 Bugs of Buffalo Team</p>
        <p>Powered by TensorFlow, Streamlit, and Mistral AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
