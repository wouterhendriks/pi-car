import cv2
import time
import face_recognition
import threading
import datetime
import os
import numpy as np
import subprocess
import platform
import sys
import yaml

# Define paths for assets and ensure they exist
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
SOUNDS_DIR = os.path.join(ASSETS_DIR, 'sounds')
MODELS_DIR = os.path.join(ASSETS_DIR, 'models')
CONFIG_DIR = os.path.join(ASSETS_DIR, 'config')

# Create asset directories if they don't exist
os.makedirs(SOUNDS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Path to the settings file
SETTINGS_FILE = os.path.join(CONFIG_DIR, 'settings.yaml')

# Create default settings if the file doesn't exist
if not os.path.exists(SETTINGS_FILE):
    default_settings = {
        'greetings': {
            'default': 'Hi {name}',
            'custom': {}
        },
        'recognition_messages': {
            'default': 'Recognized: {name}',
            'custom': {}
        }
    }
    try:
        with open(SETTINGS_FILE, 'w') as file:
            yaml.dump(default_settings, file, default_flow_style=False)
        print(f"✅ Created default settings file at: {SETTINGS_FILE}")
    except Exception as e:
        print(f"❌ Failed to create settings file: {str(e)}")
        # Create empty settings as fallback
        default_settings = {
            'greetings': {'default': 'Hi {name}', 'custom': {}},
            'recognition_messages': {'default': 'Recognized: {name}', 'custom': {}}
        }

# Load settings from file
settings = {}
try:
    with open(SETTINGS_FILE, 'r') as file:
        settings = yaml.safe_load(file)
    print(f"✅ Loaded settings from: {SETTINGS_FILE}")
except Exception as e:
    print(f"❌ Failed to load settings: {str(e)}")
    # Use default settings as fallback
    settings = {
        'greetings': {'default': 'Hi {name}', 'custom': {}},
        'recognition_messages': {'default': 'Recognized: {name}', 'custom': {}}
    }

# Download dog detector cascade file if it doesn't exist
DOG_CASCADE_FILE = os.path.join(MODELS_DIR, 'haarcascade_frontalcatface.xml')
if not os.path.exists(DOG_CASCADE_FILE):
    print(f"⚠️ Dog detection model not found at: {DOG_CASCADE_FILE}")
    print(f"For better dog detection, download a specialized model")
    print(f"and place it in: {MODELS_DIR}")

    # Try to use a cat face detector as fallback
    CAT_CASCADE_FILE = os.path.join(MODELS_DIR, 'haarcascade_frontalcatface.xml')
    if os.path.exists(CAT_CASCADE_FILE):
        DOG_CASCADE_FILE = CAT_CASCADE_FILE
        print(f"✅ Using cat face detector as fallback: {CAT_CASCADE_FILE}")
    else:
        # Try to copy the OpenCV's built-in face cascade as a last resort
        OPENCV_DIR = os.path.dirname(cv2.__file__)
        DEFAULT_CASCADE = os.path.join(OPENCV_DIR, 'data', 'haarcascade_frontalface_alt.xml')
        if os.path.exists(DEFAULT_CASCADE):
            import shutil
            shutil.copy(DEFAULT_CASCADE, DOG_CASCADE_FILE)
            print(f"✅ Copied default face cascade to: {DOG_CASCADE_FILE}")
        else:
            print(f"⚠️ Default face cascade not found at: {DEFAULT_CASCADE}")
            print(f"Dog detection may not work properly.")

def test_camera(camera_index=1, process_every_n_frames=3, scale_factor=0.5, face_folder='faces', dog_folder='dogs'):
    print(f"Testing camera at index {camera_index}...")

    # Track which faces we've already greeted
    greeted_faces = set()
    last_greeting_time = time.time()
    greeting_cooldown = 5  # Seconds between greetings

    # For tracking unknown face captures
    unknown_faces_folder = 'faces-unknown'
    last_unknown_save_time = time.time()
    unknown_save_cooldown = 10  # Seconds between saving unknown faces (increased from 5 to 10)

    # For tracking dog captures
    unknown_dogs_folder = 'dogs-unknown'
    last_dog_save_time = time.time()
    dog_save_cooldown = 10  # Seconds between saving dog faces

    # Track face detection state changes
    previous_face_detected = False
    previous_dog_detected = False

    # Track when faces completely disappear from frame
    last_face_disappearance_time = time.time()
    face_reappearance_cooldown = 10  # Wait 10 seconds after faces disappear before capturing new ones

    # For dog sound tracking
    last_bark_time = time.time()
    bark_cooldown = 5  # Seconds between playing bark sounds

    # Initialize dog detector
    dog_detector = None
    try:
        dog_detector = cv2.CascadeClassifier(DOG_CASCADE_FILE)
        print(f"✅ Loaded dog detection model from: {DOG_CASCADE_FILE}")
    except Exception as e:
        print(f"❌ Failed to load dog detection model: {str(e)}")
        dog_detector = None

    # Create unknown faces folder if it doesn't exist
    if not os.path.exists(unknown_faces_folder):
        os.makedirs(unknown_faces_folder)
        print(f"✅ Created '{unknown_faces_folder}' directory for saving unknown faces.")

    # Create dogs folder if it doesn't exist
    if not os.path.exists(dog_folder):
        os.makedirs(dog_folder)
        print(f"✅ Created '{dog_folder}' directory. Please add your dog photos there.")

    # Create unknown dogs folder if it doesn't exist
    if not os.path.exists(unknown_dogs_folder):
        os.makedirs(unknown_dogs_folder)
        print(f"✅ Created '{unknown_dogs_folder}' directory for saving unknown dogs.")

    # Function to speak greetings in a separate thread
    def speak_greeting(name):
        # Check if there's a custom greeting for this person
        if name in settings.get('greetings', {}).get('custom', {}):
            greeting = settings['greetings']['custom'][name].format(name=name)
            print(f"🔊 Speaking custom greeting for {name}: {greeting}")
        else:
            # Use default greeting if no custom one exists
            greeting = settings['greetings']['default'].format(name=name)
            print(f"🔊 Speaking default greeting: {greeting}")

        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['say', greeting])
            else:  # Raspberry Pi (Linux)
                try:
                    # Try espeak for Raspberry Pi
                    subprocess.run(['espeak', greeting])
                except FileNotFoundError:
                    pass  # If speech fails, continue silently
        except Exception:
            pass  # If all speech attempts fail, just show the message

    # Function to play camera shutter sound in a separate thread
    def play_shutter_sound():
        print("Playing shutter sound...")

        # Check for custom sound file in assets directory
        custom_sound_file = os.path.join(SOUNDS_DIR, 'camera_shutter.mp3')
        print(f"Checking for sound at: {custom_sound_file}")

        # Also check in the current directory (for backward compatibility)
        local_sound_file = 'camera_shutter.mp3'
        print(f"Checking for sound at: {local_sound_file}")

        # Determine which sound file to use
        sound_file = None
        if os.path.exists(custom_sound_file):
            sound_file = custom_sound_file
            print(f"Found sound file at: {custom_sound_file}")
        elif os.path.exists(local_sound_file):
            sound_file = local_sound_file
            print(f"Found sound file at: {local_sound_file}")
        else:
            print("No sound file found.")

        # If a sound file was found, play it
        if sound_file:
            print(f"📸 *click* - Playing sound from: {sound_file}")
            try:
                if platform.system() == 'Darwin':  # macOS
                    print(f"Using macOS afplay for: {sound_file}")
                    result = subprocess.run(['afplay', sound_file], stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        print(f"Error playing sound: {result.stderr.decode()}")
                else:  # Raspberry Pi (Linux)
                    # Try common Linux players that should be on Raspberry Pi
                    players = ['aplay', 'mpg123']
                    for player in players:
                        try:
                            print(f"Trying {player} for: {sound_file}")
                            result = subprocess.run([player, sound_file], stderr=subprocess.PIPE, timeout=1)
                            if result.returncode == 0:
                                print(f"Successfully played with {player}")
                                break
                            else:
                                print(f"Error with {player}: {result.stderr.decode()}")
                        except (subprocess.SubprocessError, FileNotFoundError) as e:
                            print(f"Failed with {player}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Exception playing sound: {str(e)}")
        else:
            # If no sound file found, just print the message
            print("📸 *click* (no sound file found)")

    # Function to play dog bark sound
    def play_bark_sound():
        print("Playing bark sound...")

        # Check for bark sound file in assets directory
        bark_sound_file = os.path.join(SOUNDS_DIR, 'bark.mp3')
        print(f"Checking for bark sound at: {bark_sound_file}")

        # Also check in the current directory (for backward compatibility)
        local_bark_file = 'bark.mp3'
        print(f"Checking for bark sound at: {local_bark_file}")

        # Determine which sound file to use
        sound_file = None
        if os.path.exists(bark_sound_file):
            sound_file = bark_sound_file
            print(f"Found bark sound at: {bark_sound_file}")
        elif os.path.exists(local_bark_file):
            sound_file = local_bark_file
            print(f"Found bark sound at: {local_bark_file}")
        else:
            print("No bark sound file found.")

        # If a sound file was found, play it
        if sound_file:
            print(f"🐕 *woof* - Playing bark from: {sound_file}")
            try:
                if platform.system() == 'Darwin':  # macOS
                    print(f"Using macOS afplay for: {sound_file}")
                    result = subprocess.run(['afplay', sound_file], stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        print(f"Error playing bark sound: {result.stderr.decode()}")
                else:  # Raspberry Pi (Linux)
                    # Try common Linux players that should be on Raspberry Pi
                    players = ['aplay', 'mpg123']
                    for player in players:
                        try:
                            print(f"Trying {player} for: {sound_file}")
                            result = subprocess.run([player, sound_file], stderr=subprocess.PIPE, timeout=1)
                            if result.returncode == 0:
                                print(f"Successfully played bark with {player}")
                                break
                            else:
                                print(f"Error with {player}: {result.stderr.decode()}")
                        except (subprocess.SubprocessError, FileNotFoundError) as e:
                            print(f"Failed with {player}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Exception playing bark sound: {str(e)}")
        else:
            # If no sound file found, just print the message
            print("🐕 *woof* (no bark sound file found)")

    # Get and print the absolute path of the faces folder
    abs_face_folder = os.path.abspath(face_folder)
    print(f"Looking for faces in: {abs_face_folder}")

    # Create faces folder if it doesn't exist
    if not os.path.exists(face_folder):
        os.makedirs(face_folder)
        print(f"✅ Created '{face_folder}' directory. Please add your photo there and restart the script.")
        print(f"   Example: Add a clear face photo named 'me.jpg' to the '{abs_face_folder}' folder")
        return False

    # Load known faces from the faces folder
    known_face_encodings = []
    known_face_names = []

    face_files = [f for f in os.listdir(face_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not face_files:
        print(f"ℹ️ No face images found in '{face_folder}' directory.")
        print(f"   Please add a clear face photo (JPG/PNG) to: {abs_face_folder}")
        return False

    print(f"Loading reference faces from '{face_folder}'...")
    for file in face_files:
        try:
            image_path = os.path.join(face_folder, file)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                face_encoding = face_encodings[0]  # Take first face in the image
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(file)[0])  # Use filename without extension as name
                print(f"✅ Loaded face: {os.path.splitext(file)[0]}")
            else:
                print(f"❌ No face found in {file}. Please use a clearer image.")
        except Exception as e:
            print(f"❌ Error loading {file}: {str(e)}")

    if not known_face_encodings:
        print("❌ Couldn't load any face encodings. Please add clear photos with faces.")
        return False

    print(f"✅ Loaded {len(known_face_encodings)} reference face(s).")

    cap = cv2.VideoCapture(camera_index)

    # Configure camera properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size to reduce latency

    # Give the camera time to initialize
    time.sleep(1)

    if not cap.isOpened():
        print(f"❌ Failed to open camera at index {camera_index}.")
        return False

    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read frame from camera at index {camera_index}.")
        cap.release()
        return False

    print(f"✅ Successfully connected to camera at index {camera_index}.")

    # Set up thread-based face detection
    face_locations = []
    face_names = []
    processing_frame = None
    detection_active = False
    detection_lock = threading.Lock()
    detection_condition = threading.Condition(detection_lock)
    running = True

    # Variables for tracking face detection changes
    previous_face_count = 0
    last_log_time = time.time()
    face_presence_logged = False

    def process_face_detection():
        nonlocal face_locations, face_names, processing_frame, detection_active
        while running:
            with detection_condition:
                # Wait for a frame to be available
                while not detection_active and running:
                    detection_condition.wait(0.1)  # Wait with timeout

                if not running:
                    break

                # Process the frame
                if processing_frame is not None:
                    # Detect faces - this is CPU intensive
                    local_face_locations = face_recognition.face_locations(processing_frame)

                    # Get face encodings for any faces in the frame
                    local_face_encodings = face_recognition.face_encodings(
                        processing_frame, local_face_locations)

                    # Initialize list for names
                    local_face_names = []

                    # Compare each face to known faces
                    for face_encoding in local_face_encodings:
                        # Compare with tolerance (lower = stricter)
                        matches = face_recognition.compare_faces(
                            known_face_encodings, face_encoding, tolerance=0.6)

                        name = "Unknown"

                        # Use the first match we find
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]

                        local_face_names.append(name)

                    # Update the shared variables
                    face_locations = local_face_locations
                    face_names = local_face_names

                    processing_frame = None
                    detection_active = False

    # Start detection thread
    detection_thread = threading.Thread(target=process_face_detection)
    detection_thread.daemon = True
    detection_thread.start()

    # Show video feed with face detection
    print("Press 'q' to quit.")
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lost connection to camera.")
            break

        # Resize frame for faster processing
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        frame_count += 1

        # Calculate FPS
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time

        # Process face detection only on every Nth frame
        if frame_count % process_every_n_frames == 0:
            with detection_lock:
                if not detection_active:
                    # Convert BGR (OpenCV) to RGB (needed for face_recognition)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processing_frame = rgb_frame
                    detection_active = True
                    detection_condition.notify()

        # Detect dogs if we have a dog detector
        dog_locations = []
        if dog_detector is not None:
            # Convert to grayscale for dog detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect dogs - returns list of rectangles
            dogs = dog_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # Convert to same format as face_locations (top, right, bottom, left)
            for (x, y, w, h) in dogs:
                dog_locations.append((y, x+w, y+h, x))

        # Draw rectangles around faces
        current_face_count = len(face_locations)
        current_dog_count = len(dog_locations)

        # Generate timestamp for logging - define it before any usage
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Log face detection events to CLI with timestamp
        current_time = time.time()
        time_diff = current_time - last_log_time

        # Only log changes at most once per second
        if time_diff >= 1.0:
            # Log when faces appear
            if current_face_count > 0 and previous_face_count == 0:
                print(f"✅ [{timestamp}] Face detected! {current_face_count} face(s) found in frame")
                face_presence_logged = True
                last_log_time = current_time

            # Log when dogs appear
            if current_dog_count > 0 and previous_dog_detected == False:
                print(f"🐕 [{timestamp}] Dog detected! {current_dog_count} dog(s) found in frame")

                # Play bark sound when dog is first detected, with cooldown
                current_time = time.time()
                if current_time - last_bark_time > bark_cooldown:
                    bark_thread = threading.Thread(target=play_bark_sound)
                    bark_thread.daemon = True
                    bark_thread.start()
                    last_bark_time = current_time
                else:
                    print(f"🐕 Dog detected but bark cooldown active ({current_time - last_bark_time:.1f}s / {bark_cooldown}s)")

                last_log_time = current_time

            # Log when faces disappear
            elif current_face_count == 0 and previous_face_count > 0:
                print(f"❌ [{timestamp}] Face(s) no longer detected")
                face_presence_logged = False
                last_log_time = current_time
                # Update the last face disappearance time when faces disappear
                last_face_disappearance_time = current_time

            # Log when dogs disappear
            elif current_dog_count == 0 and previous_dog_detected:
                print(f"🐕❌ [{timestamp}] Dog(s) no longer detected")
                last_log_time = current_time

            # Log changes in face count (but only if already detected)
            elif current_face_count != previous_face_count and face_presence_logged:
                print(f"ℹ️ [{timestamp}] Face count changed: {current_face_count} face(s) now detected")
                last_log_time = current_time

        previous_face_count = current_face_count
        previous_dog_detected = current_dog_count > 0

        # Draw rectangles around faces with names and recognition status
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Choose color based on recognition (green for recognized, red for unknown)
            if name != "Unknown":
                color = (0, 255, 0)  # Green for recognized faces (BGR format)
                # Log recognition events
                if time_diff >= 1.0 and current_face_count > 0:
                    # Check if there's a custom recognition message for this person
                    if name in settings.get('recognition_messages', {}).get('custom', {}):
                        recognition_msg = settings['recognition_messages']['custom'][name].format(name=name)
                        print(f"🔍 [{timestamp}] {recognition_msg}")
                    else:
                        # Use default recognition message
                        recognition_msg = settings['recognition_messages']['default'].format(name=name)
                        print(f"🔍 [{timestamp}] {recognition_msg}")

                    last_log_time = current_time

                    # Handle greeting with cooldown
                    current_time = time.time()
                    if (name not in greeted_faces or current_time - last_greeting_time > greeting_cooldown):
                        greeting_thread = threading.Thread(target=speak_greeting, args=(name,))
                        greeting_thread.daemon = True
                        greeting_thread.start()

                        # Update greeting tracking
                        greeted_faces.add(name)
                        last_greeting_time = current_time
            else:
                color = (0, 0, 255)  # Red for unknown faces

                # Handle saving unknown faces with cooldown
                current_time = time.time()
                save_unknown = False
                time_since_disappearance = current_time - last_face_disappearance_time

                # Save if we've passed the cooldown period since last save
                if current_time - last_unknown_save_time > unknown_save_cooldown:
                    save_unknown = True
                    print(f"ℹ️ [{timestamp}] Cooldown elapsed ({unknown_save_cooldown}s) - Ready to save unknown face")

                # Or save if state changed from no face to unknown face, but only if enough time has passed since last disappearance
                if not previous_face_detected and time_since_disappearance > face_reappearance_cooldown:
                    save_unknown = True
                    print(f"ℹ️ [{timestamp}] State changed: No face → Unknown face (after {time_since_disappearance:.1f}s)")
                elif not previous_face_detected:
                    # If a face appeared but not enough time has passed
                    print(f"ℹ️ [{timestamp}] Face appeared but waiting for cooldown ({time_since_disappearance:.1f}s / {face_reappearance_cooldown}s)")

                if save_unknown:
                    # Create a timestamp for the filename
                    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_filename = f"{unknown_faces_folder}/unknown_{timestamp_str}.jpg"

                    # Extract and save the face region
                    face_img = frame[top:bottom, left:right]
                    if face_img.size > 0:  # Make sure the face region is valid
                        cv2.imwrite(face_filename, face_img)
                        print(f"💾 [{timestamp}] Saved unknown face to: {face_filename}")
                        last_unknown_save_time = current_time

                        # Make sure to call the sound function directly first
                        play_shutter_sound()

                        # Then also try with a thread as backup
                        try:
                            sound_thread = threading.Thread(target=play_shutter_sound)
                            sound_thread.daemon = True
                            sound_thread.start()
                        except Exception as e:
                            print(f"Error starting sound thread: {str(e)}")

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Create a thicker top border to embed the name
            cv2.rectangle(frame, (left, top), (right, top + 35), color, cv2.FILLED)

            # Draw name on the top border
            font = cv2.FONT_HERSHEY_DUPLEX
            text_width = cv2.getTextSize(name, font, 0.6, 1)[0][0]
            text_x = left + (right - left - text_width) // 2  # Center text
            cv2.putText(frame, name, (text_x, top + 25), font, 0.6, (255, 255, 255), 1)

        # Display FPS and face count
        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {current_face_count} | Dogs: {current_dog_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update face detection state for next frame
        previous_face_detected = current_face_count > 0

        # Process detected dogs
        for (top, right, bottom, left) in dog_locations:
            # Use a different color for dogs (blue)
            color = (255, 0, 0)  # Blue for dogs (BGR format)

            # Handle saving unknown dogs with cooldown
            current_time = time.time()
            save_dog = False

            # Save if we've passed the cooldown period
            if current_time - last_dog_save_time > dog_save_cooldown:
                save_dog = True
                print(f"🐕 [{timestamp}] Cooldown elapsed - Ready to save unknown dog")

            if save_dog:
                # Create a timestamp for the filename
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dog_filename = f"{unknown_dogs_folder}/dog_{timestamp_str}.jpg"

                # Extract and save the dog region
                dog_img = frame[top:bottom, left:right]
                if dog_img.size > 0:  # Make sure the dog region is valid
                    cv2.imwrite(dog_filename, dog_img)
                    print(f"🐕💾 [{timestamp}] Saved unknown dog to: {dog_filename}")
                    last_dog_save_time = current_time

                    # Play camera shutter sound
                    play_shutter_sound()

            # Draw rectangle around dog
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Create a thicker top border
            cv2.rectangle(frame, (left, top), (right, top + 35), color, cv2.FILLED)

            # Draw "Dog" label
            font = cv2.FONT_HERSHEY_DUPLEX
            text = "Dog"
            text_width = cv2.getTextSize(text, font, 0.6, 1)[0][0]
            text_x = left + (right - left - text_width) // 2  # Center text
            cv2.putText(frame, text, (text_x, top + 25), font, 0.6, (255, 255, 255), 1)

        cv2.imshow(f"Face Recognition (Camera Index: {camera_index})", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    running = False
    with detection_condition:
        detection_condition.notify_all()
    detection_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

    return True

# Just use the default camera index (1) with performance optimizations
test_camera(process_every_n_frames=5, scale_factor=0.5)

# Uncomment this section if you want to try multiple camera indices
# for index in range(3):  # Try indices 0, 1, and 2
#     if test_camera(index):
#         break
#     print("Trying next camera index...")
#     time.sleep(1)
