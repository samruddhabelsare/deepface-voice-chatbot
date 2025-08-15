# deepface-voice-chatbot
# Face & Voice Recognition Chatbot – S2Pedutech Internship Project

##  Overview
This project was developed as part of our internship at **S2Pedutech**.  
It is a **face-recognition-triggered voice chatbot** that automatically starts interacting when a **new unique face** is detected using the DeepFace library.  
The chatbot uses **Pico TTS** for text-to-speech and **Google Speech Recognition** for voice input.

---

##  Features
- **Face Recognition:** Detects and identifies unique faces in real-time using DeepFace.
- **Voice Interaction:** Speaks using Pico TTS and listens with Google Speech Recognition.
- **Automatic Trigger:** Starts the chatbot only when a new face appears.
- **Course Information:** Provides details about S2Pedutech’s courses, internships, and trainers.
- **Real-Time Display:** Shows face counts, appearances, and chatbot status via OpenCV window.

---

##  Technologies Used
- **Python 3**
- **OpenCV** – for camera input and display
- **DeepFace** – for face detection & embeddings
- **SpeechRecognition** – for voice input
- **Pico TTS (pico2wave)** – for text-to-speech
- **NumPy / SciPy** – for embedding comparison
- **Threading** – for running chatbot & face recognition simultaneously

---

## 🚀 How It Works
1. **Face Detection:**  
   - Opens camera feed using OpenCV.  
   - DeepFace checks for faces every few frames.
   - If a **new face** is detected (based on embeddings), the chatbot is triggered.

2. **Voice Chatbot:**  
   - Greets the user and asks for their name.  
   - Offers options like Courses, Internships, Certification, Trainers, or Exit.  
   - Responds with voice using Pico TTS.

3. **End Session:**  
   - User says “exit” or “bye” to end chatbot conversation.

---

## 📦 Installation

# Install dependencies
pip install opencv-python deepface SpeechRecognition scipy

# Install Pico TTS and ALSA utilities (Linux)
sudo apt install libttspico-utils alsa-utils


## 📂 Project Structure
