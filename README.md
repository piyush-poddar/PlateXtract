# PlateXtract

## 📌 Overview

PlateXtract is an Automatic License Plate Recognition (ALPR) system that detects and extracts vehicle license plate numbers from images with high accuracy. The system uses a YOLOv3 model for plate detection and Google Gemini API for OCR, achieving over **98% accuracy** in reading license plates.

🔥 **Live Demo Available!** PlateXtract is deployed on Streamlit, making it easy to use with a simple web-based interface.

## 🌐 Deployment

You can try PlateXtract live on Streamlit:

👉 **https://platextract.streamlit.app**

## ✨ Features

- 🚗 **High-Accuracy Plate Detection**: Utilizes a YOLOv3 model to detect vehicle license plates.
 
- 🔠 **OCR for Text Extraction**: Uses Google Gemini API to extract text from detected plates.

- 📊 **Real-Time Processing**: Processes images efficiently for real-time applications.

- 🖼️ **Supports Multiple Image Formats**: Works with various image types for flexibility.

## 🛠️ Tech Stack

- **Python** - Backend processing
 
- **YOLOv Model** - License plate detection
 
- **Google Gemini API** - OCR for text extraction
 
- **OpenCV** - Image processing

## 🚀 Installation

1. Clone the repository:
```Bash
git clone https://github.com/piyush-poddar/PlateXtract.git

cd PlateXtract
```

2. Install required dependencies:
```Bash
pip install -r requirements.txt
```

3. Download model weights from this [link](https://drive.google.com/uc?id=1Qlcv7vcyWn9UsKsjqHat4V_CuVh5Lggs) and put it inside `model/weights` directory. Alternatively, you can skip this step allow the code itself to do it for you.

4. Add your Gemini API key in `.streamlit/secrets.toml`.

5. Run the Streamlit App:
```Bash
streamlit run app.py
```

## 📌 Usage

- Provide an image of a vehicle with number plate.

- The model detects the license plate and extracts the number.

- The extracted text is displayed as output.

## 🔮 Future Enhancements

- 🚀 Implementing real-time detection via a webcam or video feed.

- 🛣️ Integration with traffic monitoring and law enforcement systems.

- 🌍 Support for multiple languages in OCR.

## 🤝 Contributing

1. Fork the repo and create a new branch.

2. Make improvements and test thoroughly.

3. Submit a pull request with a detailed description.

### Made with ❤️ by Piyush Poddar
