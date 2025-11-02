import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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
        padding-top: 1rem;
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
    /* Hide canvas toolbar completely */
    .stSelectbox, .stNumberInput {
        display: none !important;
    }
    /* Canvas border styling */
    canvas {
        border: 2px solid #667eea !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with lazy import
@st.cache_resource
def load_model():
    """Load the TensorFlow model (cached for performance)"""
    try:
        # Lazy import of tensorflow for faster startup
        import tensorflow as tf
        
        # Load model in Keras native format (.keras)
        model = tf.keras.models.load_model('MNIST_model.keras')
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Title and description
st.title("🎨 MNIST Digit Recognition")
st.markdown("Draw a digit (0-9) in the canvas below and click **Predict** to see the AI's prediction!")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw Your Digit")
    
    # Initialize canvas key in session state if not present
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Create canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
        stroke_width=20,
        stroke_color="#000000",  # Black stroke
        background_color="#FFFFFF",  # White background
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=False,  # Hide the toolbar
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Buttons
    col_clear, col_predict = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            if "prediction_results" in st.session_state:
                del st.session_state.prediction_results
            st.rerun()
    
    with col_predict:
        if st.button("🔮 Predict", use_container_width=True, type="primary"):
            if canvas_result.image_data is not None and model is not None:
                # Process the canvas image
                image_data = canvas_result.image_data
                
                # Convert to PIL Image and then to grayscale
                img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
                img = img.convert('L')  # Convert to grayscale
                
                # Resize to 28x28 (MNIST format)
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img)
                
                # Invert colors (MNIST expects white digits on black background)
                img_array = 255 - img_array
                
                # Normalize to 0-1 range
                img_array = img_array.astype('float32') / 255.0
                
                # Reshape for model input (1, 28, 28)
                img_array = img_array.reshape(1, 28, 28)
                
                # Make prediction
                try:
                    import tensorflow as tf
                    predictions = model.predict(img_array, verbose=0)
                    
                    # Get prediction results
                    predicted_digit = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_digit]
                    
                    # Store results in session state
                    st.session_state.prediction_results = {
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'all_predictions': predictions[0],
                        'processed_image': img_array[0]
                    }
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please draw a digit first!")

with col2:
    st.subheader("🔮 Prediction Results")
    
    # Display prediction results if available
    if "prediction_results" in st.session_state:
        results = st.session_state.prediction_results
        
        # Main prediction display
        st.markdown(f"""
        <div class="prediction-box">
            <div class="digit-display">{results['digit']}</div>
            <div class="confidence-text">Confidence: {results['confidence']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 3 predictions
        st.subheader("📊 Top 3 Predictions")
        top_3_indices = np.argsort(results['all_predictions'])[-3:][::-1]
        
        for i, idx in enumerate(top_3_indices):
            confidence = results['all_predictions'][idx]
            st.write(f"**{i+1}. Digit {idx}:** {confidence:.1%}")
            st.progress(float(confidence))
        
        # Show processed image
        st.subheader("🖼️ Processed Image")
        st.image(results['processed_image'], caption="28x28 processed image", width=140)
        
        # All probabilities (expandable)
        with st.expander("📈 All Digit Probabilities"):
            for digit in range(10):
                prob = results['all_predictions'][digit]
                st.write(f"Digit {digit}: {prob:.3f} ({prob:.1%})")
    else:
        st.info("👆 Draw a digit and click **Predict** to see results!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a neural network trained on the MNIST dataset to recognize handwritten digits.
    
    **Model Architecture:**
    - 5 Dense layers with 128 neurons each
    - ReLU activation functions
    - Softmax output layer
    - Trained on 60,000 MNIST samples
    - Achieves ~98% accuracy
    
    **How to use:**
    1. Draw a digit (0-9) on the canvas
    2. Click "Predict" to get AI prediction
    3. View confidence scores and analysis
    4. Click "Clear" to start over
    """)
    
    st.header("🔧 Tips")
    st.write("""
    - Draw digits **large and centered**
    - Use **thick, bold strokes**
    - Make digits **clear and recognizable**
    - The model expects **white digits on black background**
    """)
