import cv2
import time
import os
import threading
import queue # Not directly used in current setup but good for thread communication
from deepface import DeepFace
from scipy.spatial.distance import cosine

# --- CHATBOT MODULE (Pico TTS) ---
import speech_recognition as sr
import subprocess

# --- Chatbot Configuration ---
MIC_LISTEN_TIMEOUT = 5 # Seconds to wait for speech
MIC_PHRASE_TIME_LIMIT = 5 # Seconds for a phrase to complete
PICO_LANG = 'en-US' # Language for pico2wave. Common options: 'en-US', 'en-GB', etc.
TEMP_AUDIO_FILE = '/tmp/chatbot_speech.wav' # Temporary file for generated speech

# --- Chatbot Functions ---
def speak(text):
    """
    Speaks the given text using Pico TTS (pico2wave) and aplay.
    """
    print(f"Bot: {text}")
    try:
        # Generate the WAV file using pico2wave
        subprocess.run(
            ['pico2wave', '-w', TEMP_AUDIO_FILE, '-l', PICO_LANG, text],
            check=True,
            capture_output=True # Capture output to prevent it from printing to console
        )
        
        # Play the generated WAV file using aplay
        subprocess.run(
            ['aplay', TEMP_AUDIO_FILE],
            check=True,
            capture_output=True # Capture output to prevent it from printing to console
        )
        
        # Clean up the temporary audio file
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

    except FileNotFoundError:
        print("ERROR: 'pico2wave' or 'aplay' command not found.")
        print("Please ensure 'libttspico-utils' is installed. (sudo apt install libttspico-utils)")
        print("And 'aplay' is part of 'alsa-utils'. (sudo apt install alsa-utils)")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate or play speech. Command: {e.cmd}, Error: {e.stderr.decode()}")
        print("Check if audio devices are correctly configured.")
    except Exception as e:
        print(f"An unexpected error occurred during speak: {e}")

def listen():
    """Listens for user input and returns recognized text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=MIC_LISTEN_TIMEOUT, phrase_time_limit=MIC_PHRASE_TIME_LIMIT)
            print("Processing...")
            query = r.recognize_google(audio)
            print(f"You: {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            speak("I didn't hear anything.")
            print("No speech detected within timeout.")
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand what you said.")
            print("Could not understand audio.")
        except sr.RequestError as e:
            speak("I'm having trouble connecting to the speech service. Please check your internet connection.")
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            speak("An unexpected error occurred during listening.")
            print(f"An unexpected error occurred: {e}")
    return ""

def show_courses():
    """Details available courses."""
    speak("Here are the available courses at S 2 P edutech.")
    speak("MERN Full Stack Development. Duration: 3 Months. Technologies include MongoDB, Express.js, React.js, and Node.js. Fee: 25,000.")
    speak("MEAN Full Stack Development. Duration: 3 Months. Technologies include MongoDB, Express.js, Angular, and Node.js. Fee: 24,000.")
    speak("Java Full Stack Development. Duration: 4 Months. Covers Java, Spring Boot, Hibernate, HTML, CSS, and JavaScript. Fee: 30,000.")
    speak("Data Analyst Course. Duration: 3.5 Months. Includes Python, Pandas, NumPy, Matplotlib, and SQL. Fee: 28,000.")

def chatbot_main_loop():
    """Main loop for the chatbot interaction."""
    speak("Welcome to S 2 P edutech, your gateway to tech success!")
    speak("May I know your name?")
    name = listen()
    if name:
        speak(f"Nice to meet you, {name.capitalize()}!")
    else:
        speak("No problem, we can continue without a name.")

    while True:
        speak("What would you like to know?")
        speak("You can say: Courses, Internships, Certification, Trainers, or Exit.")
        query = listen()

        if not query:
            continue

        if "course" in query:
            show_courses()
        elif "internship" in query:
            speak("We offer internships after course completion based on performance.")
        elif "certification" in query:
            speak("You will receive a certificate after successfully completing your course.")
        elif "trainer" in query or "mentor" in query:
            speak("Our trainers are industry experts with over five years of experience.")
        elif "exit" in query or "quit" in query or "bye" in query:
            speak("Thank you for visiting S2Pedutech. Goodbye!")
            break
        else:
            speak("Sorry, I didn't understand that. Please try again.")

    print("Chatbot session ended.")


# --- FACE RECOGNITION MODULE ---
# ==== CONFIG ====
MODEL_NAME = "SFace" # Lightweight model
THRESHOLD = 0.65 # Adjust as needed for unique face recognition similarity
COOLDOWN_SEC = 20 # Prevent same unique face from re-triggering within this time
CHECK_EVERY_N_FRAMES = 10 # Process DeepFace every N frames (higher = faster but less reactive)

# --- Global flags and counters for communication between threads ---
face_detected_event = threading.Event() # Set when a unique face triggers chatbot
chatbot_active_flag = threading.Event() # Set when chatbot thread is running
unique_faces_recognized_counter = 0 # Counts unique individuals recognized
embeddings_db = [] # Stores embeddings of recognized unique faces
timestamps = [] # Stores last recognition time for each unique face (for cooldown)
lock = threading.Lock() # To protect shared variables (unique_faces_recognized_counter, embeddings_db, timestamps)

def is_unique_face(new_embedding):
    """
    Checks if a new face embedding is unique or already exists in the database
    within the COOLDOWN_SEC.
    """
    now = time.time()
    for i, saved_embedding in enumerate(embeddings_db):
        if cosine(saved_embedding, new_embedding) < THRESHOLD:
            # If the same face is detected again, reset its timestamp for cooldown
            # This means a known face won't be re-recognized as unique within COOLDOWN_SEC
            if now - timestamps[i] > COOLDOWN_SEC:
                timestamps[i] = now
            return False # Not a unique face
    return True # This is a new unique face

def face_recognition_loop():
    """
    Runs face detection and unique face recognition continuously.
    Updates counters and triggers chatbot on unique face appearance.
    """
    global unique_faces_recognized_counter, embeddings_db, timestamps

    os.makedirs("unique_faces", exist_ok=True) # Ensure directory for unique face images exists

    cap = cv2.VideoCapture(0) # Initialize camera (0 is usually default webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("? Camera not found! Face recognition will not run.")
        return # Exit the thread if camera isn't found

    print("? Face Detection Thread Started. Looking for faces...")
    frame_index = 0
    
    # Counters/flags for displaying face presence events
    face_appearance_events_counter = 0 # Counts times a face (any face) appears after not being present
    was_face_present_in_prev_check = False # Flag to track face presence across intervals

    while True:
        ret, frame = cap.read()
        if not ret:
            print("? Failed to read frame from camera. Exiting face detection.")
            break

        frame_index += 1
        frame_display = frame.copy() # Create a copy for drawing overlays

        current_frame_faces_count = 0 # Number of faces detected in the current processed frame

        # Only process DeepFace every N frames to save CPU cycles
        if frame_index % CHECK_EVERY_N_FRAMES == 0:
            try:
                # DeepFace.represent returns a list of dictionaries, one for each face detected.
                # If no faces are found, it returns an empty list.
                faces = DeepFace.represent(frame, model_name=MODEL_NAME, enforce_detection=False, detector_backend='opencv')
                current_frame_faces_count = len(faces)
                
                # Logic for face_appearance_events_counter
                is_face_present_now = current_frame_faces_count > 0
                if is_face_present_now and not was_face_present_in_prev_check:
                    face_appearance_events_counter += 1
                    print(f"** Face Appeared! Total appearance events: {face_appearance_events_counter}")
                was_face_present_in_prev_check = is_face_present_now

                for face in faces:
                    emb = face["embedding"]
                    region = face["facial_area"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

                    with lock: # Acquire lock to safely access and modify shared data
                        if is_unique_face(emb):
                            unique_faces_recognized_counter += 1
                            embeddings_db.append(emb)
                            timestamps.append(time.time())

                            # Save cropped image of the new unique face
                            cropped = frame[y:y+h, x:x+w]
                            filename = f"unique_faces/face_{unique_faces_recognized_counter}.jpg"
                            cv2.imwrite(filename, cropped)
                            print(f"?? Saved NEW unique face #{unique_faces_recognized_counter} ? {filename}")
                            
                            # --- CHATBOT TRIGGER LOGIC ---
                            # Trigger chatbot ONLY on a NEW unique face AND if chatbot is not currently active
                            if not chatbot_active_flag.is_set():
                                face_detected_event.set() # Set the event to signal the main thread
                                print("! NEW Unique Face detected! Signalling chatbot to start...")

                    # Draw rectangle on the display frame (coordinates relative to the original frame)
                    x, y = max(0, x), max(0, y) # Ensure coordinates are not negative
                    w, h = max(1, w), max(1, h) # Ensure width/height are at least 1
                    cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            except ValueError as ve:
                # DeepFace.represent can raise ValueError, but with enforce_detection=False,
                # it's usually not for 'no face detected'. Catch other specific DeepFace errors.
                if "No face detected" not in str(ve): # Skip expected 'no face' errors if enforce_detection was True
                    print(f"?? DeepFace Value Error: {ve}")
            except Exception as e:
                print(f"?? General Detection Error (likely DeepFace internal): {e}")

        # Display information on the OpenCV window
        # Use 'lock' when reading shared global variables to prevent race conditions
        with lock:
            cv2.putText(frame_display, f"Faces in Frame: {current_frame_faces_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_display, f"Face Appearances: {face_appearance_events_counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_display, f"Unique Faces: {unique_faces_recognized_counter}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Indicate chatbot status
            if chatbot_active_flag.is_set():
                cv2.putText(frame_display, "Chatbot Active!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame_display, "Chatbot Ready", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Detector", frame_display) # Display the frame

        # Check for 'q' key press to quit the video feed and thread
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # Release camera resources
    cv2.destroyAllWindows() # Close OpenCV windows
    print("? Face Detection Thread Stopped.")


# --- Main Orchestrator ---
if _name_ == "_main_":
    print("Starting Main Robot Orchestrator...")

    # Start Face Recognition in a separate thread
    # daemon=True means the thread will automatically exit when the main program exits
    face_thread = threading.Thread(target=face_recognition_loop, daemon=True)
    face_thread.start()

    while True:
        # Wait for the face_detected_event to be set by the face recognition thread.
        # Use a timeout to prevent blocking indefinitely and allow for graceful exit.
        face_detected_event.wait(timeout=1) # Check every 1 second

        # If a unique face was detected AND the chatbot is not currently running
        if face_detected_event.is_set() and not chatbot_active_flag.is_set():
            print("! Unique Face detected! Launching chatbot conversation...")
            face_detected_event.clear() # Reset the event so it doesn't trigger repeatedly

            # Start Chatbot in a separate thread
            chatbot_active_flag.set() # Set flag to indicate chatbot is active
            chat_thread = threading.Thread(target=chatbot_main_loop)
            chat_thread.start()
            chat_thread.join() # Wait for the chatbot thread to complete its conversation
            chatbot_active_flag.clear() # Clear flag as chatbot has finished
            print("Chatbot finished. Waiting for new unique face to trigger again, or 'q' in OpenCV window to quit.")

        # Main loop continues to run, checking for events.
        # If the face_thread dies (e.g., camera unplugged, 'q' pressed in OpenCV window),
        # this loop will continue until manually stopped (Ctrl+C).
        # A more robust shutdown would involve a shared stop_event that both threads check.
        time.sleep(0.1) # Small sleep to prevent busy-waiting
