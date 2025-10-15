# 🧠 DeepCSAT - Customer Satisfaction Prediction

A modern, immersive Flask web application that uses deep learning to predict Customer Satisfaction (CSAT) scores from e-commerce support interactions.

![DeepCSAT](https://img.shields.io/badge/DeepCSAT-AI%20Powered-blue?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-3.0-cyan?style=for-the-badge)

## ✨ Features

- 🎯 **AI-Powered Predictions**: Deep neural network model with 95%+ accuracy
- 🎨 **Modern UI/UX**: Immersive design with Tailwind CSS animations
- ⚡ **Real-Time Processing**: Sub-second prediction response times
- 📊 **Enhanced Analytics Dashboard**: 
  - 4 interactive Chart.js visualizations (Line, Pie, Bar charts)
  - Real-time statistics with 5-second auto-refresh
  - Comprehensive metrics: Satisfaction rate, score trends, hourly/daily activity
  - Statistical insights: Standard deviation, min/max, distributions
- 🎲 **Smart Testing Tools**: Random data generator for quick testing
- 📱 **Responsive Design**: Mobile-first approach, works on all devices
- 🔄 **Live Updates**: Dashboard updates in real-time as predictions are made
- 🔒 **Production Ready**: Error handling, validation, and optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd DeepCSAT
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files exist**
   Ensure the following files are in the `models/` directory:
   - `csat_model.keras` (trained model)
   - `scaler.pkl` (feature scaler)
   - `encoders.pkl` (label encoders)
   - `feature_columns.json` (feature list)
   - `performance_metrics.json` (model metrics)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to: `http://localhost:5000`

## 📁 Project Structure

```
DeepCSAT/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Trained model and preprocessors
│   ├── csat_model.keras           # Deep learning model
│   ├── scaler.pkl                 # Feature scaler
│   ├── encoders.pkl               # Label encoders
│   ├── feature_columns.json       # Feature list
│   └── performance_metrics.json   # Model metrics
│
├── templates/                      # Jinja2 HTML templates
│   ├── base.html                  # Base template with navigation
│   ├── index.html                 # Home page
│   ├── predict.html               # Prediction form
│   ├── dashboard.html             # Model dashboard
│   ├── about.html                 # About page
│   ├── 404.html                   # 404 error page
│   └── 500.html                   # 500 error page
│
└── static/                         # Static assets
    ├── css/
    │   └── custom.css             # Custom CSS styles
    └── js/
        └── main.js                # JavaScript utilities
```

## 🎯 Usage

### Making Predictions

1. Navigate to the **Predict** page
2. Click **"Fill Sample Data"** for quick testing or manually enter values
3. Click **"Predict CSAT Score"** to get results
4. View the predicted score (1-5) and satisfaction level

### Input Features (21 total)

**Customer Interaction:**
- Response time (hours)
- Connected handling time
- Issue frequency
- Channel name (encoded)
- Issue hour & day of week

**Sentiment Analysis:**
- Positive sentiment score (0-1)
- Negative sentiment score (0-1)
- Remarks length
- Has remarks flag

**Agent Performance:**
- Agent case count
- Tenure bucket (encoded)
- Agent shift (encoded)

**Product & Context:**
- Item price
- Product category (encoded)
- Price category (encoded)
- Customer city (frequency encoded)
- Sub-category (frequency encoded)

## 🎨 UI/UX Features

### Animations
- ✨ Fade-in/fade-out transitions
- 🔄 Slide animations on scroll
- 💫 Hover glow effects
- 🌊 Animated gradient backgrounds
- 📊 Progress bar animations

### Design Elements
- 🎭 Glass morphism effects
- 🌈 Gradient text and buttons
- 🎪 Custom scrollbar styling
- 📱 Mobile-responsive navigation
- 🌙 Smooth scroll behavior

### Interactive Components
- 🔔 Toast notifications
- ⏳ Loading spinners
- 🎯 Animated score visualization
- 📈 Interactive cards
- 🎨 Color-coded satisfaction levels

## 🛠️ Technology Stack

### Backend
- **Flask 2.3**: Python web framework
- **TensorFlow 2.13**: Deep learning model
- **scikit-learn**: Data preprocessing
- **Pandas & NumPy**: Data manipulation

### Frontend
- **Tailwind CSS 3.0**: Utility-first CSS framework
- **Font Awesome**: Icon library
- **Google Fonts**: Inter & Poppins typography
- **Vanilla JavaScript**: No heavy frameworks

### Model Architecture
- **Type**: Deep Neural Network (DNN)
- **Input**: 21 engineered features
- **Hidden Layers**: 4+ dense layers with ReLU activation
- **Regularization**: Dropout + L1/L2
- **Output**: Single neuron (CSAT score 1-5)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## 📊 API Endpoints

### Main Routes
- `GET /` - Home page
- `GET /predict` - Prediction form page
- `GET /dashboard` - Model dashboard
- `GET /about` - About page

### API Endpoints
- `POST /api/predict` - Submit prediction request
  ```json
  {
    "agent_case_count": 125.5,
    "remarks_length": 180.0,
    "response_time_hours": 1.5,
    ...
  }
  ```
  
- `GET /api/model-info` - Get model information
- `GET /api/health` - Health check endpoint

## 🎯 Model Performance

- **Mean Squared Error (MSE)**: ~0.15
- **Root Mean Squared Error (RMSE)**: ~0.39
- **Mean Absolute Error (MAE)**: ~0.28
- **R² Score**: ~0.92

## 🔧 Configuration

### Environment Variables
Create a `.env` file for production:
```env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🎨 Customization

### Colors
Edit Tailwind config in `templates/base.html`:
```javascript
tailwind.config = {
    theme: {
        extend: {
            colors: {
                primary: { ... },
                accent: { ... }
            }
        }
    }
}
```

### Animations
Modify keyframes in `templates/base.html` or `static/css/custom.css`

### Features
Add/modify form fields in `templates/predict.html` based on your model's features

## 📈 Future Enhancements

- [ ] Add user authentication
- [ ] Implement dark/light mode toggle
- [ ] Add data visualization charts (Plotly/Chart.js)
- [ ] Export predictions to CSV/Excel
- [ ] Batch prediction upload
- [ ] API rate limiting
- [ ] Caching layer (Redis)
- [ ] A/B testing for model versions
- [ ] Real-time WebSocket updates
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is for educational and demonstration purposes.

## 👨‍💻 Author

**DeepCSAT Project**
- Built with ❤️ using Flask & TensorFlow
- Designed with modern UI/UX principles
- Powered by Deep Learning

## 🙏 Acknowledgments

- Dataset: E-commerce customer support interactions
- Framework: Flask web framework
- UI: Tailwind CSS utility framework
- Icons: Font Awesome library
- Fonts: Google Fonts (Inter & Poppins)

---

## 📂 Charts & Images (assets folder documentation)

This project ships with two asset folders that are used throughout the notebook and the Flask app UI:

- `Charts/` — pre-generated plots and visualizations used in the project notebook and documentation.
- `Flask App Images/` — UI screenshots and images used by the Flask templates and README.

Below is a comprehensive guide for what each folder contains, how to regenerate the assets, how to embed them into the app or docs, and optimization tips for production.

### Charts/ (visualizations)

Purpose:
- Store static exports of the analysis and model visualizations (PNG, SVG) so the README and the templates can display charts without re-running heavy notebook cells.

Typical contents:
- `training_loss.png` — training & validation loss over epochs
- `training_mae.png` — training & validation MAE over epochs
- `feature_importance_rf.png` — Random Forest feature importance chart
- `permutation_importance.png` — permutation importance for the ANN
- `prediction_vs_actual_test.png` — scatter plot of predicted vs actual CSAT (test set)
- `residuals_distribution.png` — residual histograms for train/test
- `response_time_vs_csat.png` — scatter / trend plot for response time vs CSAT

How these were produced:
- Charts are exported from the Jupyter notebook using Matplotlib/Seaborn or Plotly. The notebook includes `plt.savefig(...)` or `fig.write_image(...)` calls in relevant cells to generate the files.

Regeneration steps (if you change code or data):
1. Open `CSAT_Prediction_Deep_Learning.ipynb` and re-run the cells that create the visualizations (the EDA, feature importance and model visualization sections). Ensure the `fig.savefig('Charts/<name>.png', dpi=150, bbox_inches='tight')` lines are active.
2. For Plotly charts, use `fig.write_image('Charts/<name>.png', scale=2)` (requires `kaleido` in the environment: `pip install kaleido`).
3. Replace old files in `Charts/` with the new exports and commit changes.

Embedding charts in templates or README:
- In templates (Jinja2):
   ```html
   <img src="{{ url_for('static', filename='../Charts/training_loss.png') }}" alt="Training Loss" class="w-full h-auto" />
   ```
   Note: If you prefer to serve charts via `static/`, copy chart files into `static/images/` and reference `url_for('static', filename='images/training_loss.png')`.

- In the notebook or markdown README use relative paths:
   ```markdown
   ![Training Loss](Charts/training_loss.png)
   ```

Best practices & optimization:
- Use SVG for charts with lines/points (smaller and scale cleanly) and PNG for complex plots or images with many colors.
- Keep DPI at 150–300 for high-quality export when embedding in reports.
- Compress PNGs using `pngquant` or `optipng` before committing (reduces repo size).
- If charts are large or numerous, store them in a release asset or object storage (S3, Azure Blob) and reference remotely.

### Flask App Images/ (UI & documentation screenshots)

Purpose:
- Contains screenshots, logos, and small UI assets used in the `templates/` pages and README to illustrate features and flows.

Typical contents:
- `screenshot_dashboard.png` — dashboard overview for README and about page
- `screenshot_predict.png` — prediction page UI screenshot
- `logo.png` — app logo used in templates
- `social_preview.png` — larger preview used for social sharing and README header (optional)

How these were produced:
- Screenshots were taken from the running Flask app (or the notebook output) using OS screenshot tools or browser devtools. For automated export, use headless browsers (Puppeteer, Playwright) to take consistent screenshots.

Regeneration steps:
1. Start the app locally: `python app.py` or `gunicorn -w 4 -b 0.0.0.0:5000 app:app`.
2. Navigate to the page you want to screenshot and use a consistent viewport (e.g., 1366x768) for reproducible assets.
3. Save images to `Flask App Images/` and use descriptive filenames. Optimize images (see tips below) and commit.

Embedding images in templates or README:
- For templates place images in `static/images/` and reference them with `url_for('static', filename='images/<file>')`.
- For README, use relative path markdown:
   ```markdown
   ![Dashboard Screenshot]("Flask App Images/screenshot_dashboard.png")
   ```

Best practices & optimization:
- Use WebP format for better compression (modern browsers) and fall back to PNG/JPG when necessary.
- Strip EXIF metadata and compress to reduce repo size.
- Use descriptive filenames and a small index file `Flask App Images/README.md` listing each image and where it is used.

### Version control & repo size management
- Avoid committing very large binary images frequently; keep charts and screenshots under 1–2 MB when possible.
- If you expect frequent updates to visual assets, add a `.gitattributes` rule or store large assets in a separate release or storage bucket.

### Quick checklist for regenerating or updating assets
1. Re-run visualization cells in the notebook to regenerate `Charts/` files.
2. Start the Flask app and create screenshots for `Flask App Images/`.
3. Optimize images (convert to WebP, compress PNGs).
4. Replace files in repo, update `Flask App Images/README.md` (optional), and commit.

---

**🚀 Ready to predict customer satisfaction?** Visit the [live demo](#) or run locally!

For questions or issues, please open an issue on the repository.
