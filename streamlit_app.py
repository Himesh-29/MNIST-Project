import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .digit-display {
        font-size: 120px;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 20px rgba(255,255,255,0.5);
        margin: 20px 0;
    }
    .confidence-text {
        font-size: 24px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the TensorFlow model (cached for performance)"""
    try:
        model = tf.keras.models.load_model('MNIST_epic_number_reader.model')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Title and description
st.title("🎨 MNIST Digit Recognition")
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <p style='font-size: 18px; color: #666;'>
            Draw a digit from 0-9 and watch AI predict it!
        </p>
        <p style='font-size: 14px; color: #888;'>
            Model Accuracy: 97.82% | Powered by TensorFlow & Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### ✏️ Draw Your Digit")
    
    # Canvas using Streamlit's image annotation
    uploaded_file = st.file_uploader(
        "Upload a 28x28 digit image or use the drawing tool below",
        type=['png', 'jpg', 'jpeg']
    )
    
    # Create a placeholder for drawing
    if uploaded_file is None:
        # Instructions for manual drawing
        st.info("💡 **Tip**: Upload a 28x28 grayscale digit image, or use any image editor to draw a digit!")
        st.image("https://via.placeholder.com/280x280/ffffff/000000?text=Draw+a+Digit+0-9", use_container_width=True)
        
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Process button
        if st.button("🔍 Predict", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing..."):
                    # Preprocess image
                    image = image.convert('L')
                    image = image.resize((28, 28))
                    img_array = np.array(image)
                    
                    # Normalize
                    img_array = img_array.astype('float32') / 255.0
                    
                    # Reshape for model
                    img_array = img_array.reshape(1, 28, 28)
                    
                    # Predict
                    predictions = model.predict(img_array, verbose=0)
                    predicted_digit = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_digit])
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(predictions[0])[-3:][::-1]
                    top_predictions = [
                        {'digit': int(idx), 'confidence': float(predictions[0][idx])}
                        for idx in top_indices
                    ]
                    
                    # Display results in col2
                    with col2:
                        st.markdown("### 🎯 Prediction")
                        
                        # Prediction box
                        prediction_html = f"""
                        <div class="prediction-box">
                            <div class="digit-display">{predicted_digit}</div>
                            <div class="confidence-text">
                                Confidence: {confidence*100:.2f}%
                            </div>
                        </div>
                        """
                        st.markdown(prediction_html, unsafe_allow_html=True)
                        
                        # Confidence progress bar
                        st.progress(confidence)
                        
                        # Top 3 predictions
                        st.markdown("### 📊 Top 3 Predictions")
                        for i, pred in enumerate(top_predictions):
                            label = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{label} **Digit {pred['digit']}**: {pred['confidence']*100:.1f}%")
                            
                            # Progress bar for each
                            st.progress(pred['confidence'])
            else:
                st.error("Model not loaded. Please check model files.")

with col2:
    # This will be populated after prediction
    if 'predicted_digit' not in st.session_state:
        st.markdown("### 🎯 Prediction")
        st.info("Draw a digit and click Predict to see results here!")
        
        # Display example
        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. **Draw a digit** on a 28x28 pixel image
        2. **Save** as PNG or JPG
        3. **Upload** using the file uploader
        4. **Click Predict** to get results!
        
        **Suggested Tools**:
        - Windows Paint / Paint 3D
        - macOS Preview
        - Online: photopea.com
        - Mobile: Any drawing app
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #888;'>
        <p>Built with ❤️ using TensorFlow & Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown("""
    This MNIST digit recognition app uses a deep neural network 
    trained on 60,000 handwritten digits.
    
    **Model Stats**:
    - 5 Dense Layers (128 neurons each)
    - ReLU Activation
    - Adam Optimizer
    - 97.82% Accuracy
    
    **Dataset**: MNIST (Modified NIST)
    """)
    
    st.markdown("### 🔗 Deploy Your Own")
    st.markdown("""
    Host this free on Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repository
    4. Deploy instantly!
    """)


