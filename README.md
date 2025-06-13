# SENTIMENT-ANALYSIS
a list of useful libraries categorized by data type and task for sentiment analysis in text, images, and videos:

🔤 Text Sentiment Analysis
These libraries are commonly used for analyzing text data:

Library	Use
NLTK (nltk)	Rule-based sentiment (e.g., VADER); NLP tasks
TextBlob	Simple sentiment analysis and text preprocessing
Transformers (transformers, by Hugging Face)	Deep learning-based sentiment models (e.g., BERT)
spaCy	Fast and scalable NLP with sentiment via extensions
scikit-learn	For traditional ML models (SVM, Naive Bayes)
TfidfVectorizer / CountVectorizer	Text vectorization tools for ML

🖼️ Image Sentiment Analysis
These are useful for emotion recognition and analysis from images:

Library	Use
DeepFace	Facial expression and emotion detection
OpenCV (cv2)	Image processing, face detection
mediapipe	Face and body landmarks for emotion clues
keras / TensorFlow / PyTorch	Deep learning frameworks to train emotion recognition models
face-recognition	High-level face detection and recognition

🎥 Video Sentiment Analysis
Sentiment from video involves frame-by-frame emotion + audio/text cues:

Library	Use
OpenCV	Capture and process video frame-by-frame
DeepFace	Analyze facial emotions in video frames
moviepy	Extract frames, audio from videos
SpeechRecognition	Convert spoken content in video to text
transformers	For analyzing transcribed video text
librosa	Audio sentiment features from video (tone, pitch)

📦 Bonus: Combined or General-Purpose Tools
Library	Use
torchvision	Pretrained image models (ResNet, etc.)
pandas / numpy	Data manipulation
matplotlib / seaborn	Visualization of sentiment results
streamlit / Flask / Gradio	Build sentiment analysis apps easily
