# 🎨 MNIST Digit Recognition - Interactive Web App

An end-to-end web application for recognizing handwritten digits using a deep learning model trained on the MNIST dataset. Built with **TensorFlow** and **Streamlit**.

![Model Accuracy](https://img.shields.io/badge/Accuracy-97.82%25-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

## 🌟 Features

- **Easy Upload**: Upload digit images and get instant predictions
- **Real-time Prediction**: Get predictions with confidence scores
- **Top 3 Predictions**: See the top 3 most likely digits with their probabilities
- **Beautiful UI**: Modern gradient design with smooth animations
- **100% Free**: Host forever on Streamlit Cloud - no credit card required!
- **Mobile Friendly**: Works seamlessly on all devices

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Himesh-29/MNIST-Project.git
   cd MNIST-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser**
   The app will automatically open at `http://localhost:8501`

## 🌐 FREE Deployment on Streamlit Cloud (Recommended)

**Deploy in 2 minutes - FREE FOREVER!**

### Step-by-Step:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/MNIST-Project.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `MNIST-Project`
   - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Done!** 🎉
   Your app is now live at: `https://YOUR_APP_NAME.streamlit.app`

### Why Streamlit Cloud?

- ✅ **100% FREE** - No credit card, no limits
- ✅ **Lifetime hosting** - Never expires
- ✅ **Easy deployment** - Connect GitHub and deploy
- ✅ **Auto-updates** - Deploy from any commit
- ✅ **Custom domain** - Add your own URL
- ✅ **Public or private** - Your choice
- ✅ **No maintenance** - Fully managed

## 🎮 How to Use

1. **Create a digit image** (28x28 recommended)
   - Use Windows Paint / macOS Preview
   - Draw a digit 0-9 on white background
   - Save as PNG or JPG

2. **Upload the image**
   - Click "Browse files" in the app
   - Select your digit image

3. **Get prediction**
   - Click "🔍 Predict" button
   - See results instantly!

4. **View results**
   - Main prediction with confidence score
   - Top 3 predictions with probabilities
   - Visual progress bars

## 📊 Model Details

- **Architecture**: Sequential Neural Network with 5 dense layers
- **Training Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Input Shape**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Accuracy**: **97.82%** on test set
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

## 📁 Project Structure

```
MNIST-Project/
│
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt               # Python dependencies
├── packages.txt                   # System packages (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml               # Streamlit configuration
│
├── MNIST_epic_number_reader.model/  # Saved TensorFlow model
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
│
├── MNIST project by Himesh Maniyar.ipynb  # Original training notebook
│
└── README.md                     # This file
```

## 🛠️ Technologies Used

- **Streamlit**: Interactive web app framework
- **TensorFlow/Keras**: Deep learning model
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **Python**: Core programming language

## 🔧 Customization

### Update Model

Re-train the model using the Jupyter notebook and save with:
```python
model.save('MNIST_epic_number_reader.model')
```

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"  # Change primary color
backgroundColor = "#f0f2f6"  # Change background
```

### Add Features

The Streamlit app is easy to extend. Just edit `streamlit_app.py` and add more widgets, charts, or functionality!

## 📚 Learning Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/streamlit-community-cloud)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Himesh Maniyar**  
- GitHub: [@Himesh-29](https://github.com/Himesh-29)

## 🙏 Acknowledgments

- MNIST dataset creators
- TensorFlow team
- Streamlit team
- All contributors and users

---

⭐ **If you found this project helpful, please consider giving it a star!**

🚀 **Ready to deploy? Follow the instructions above and host it FREE on Streamlit Cloud!**
