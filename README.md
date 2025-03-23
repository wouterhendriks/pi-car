# Face & Dog Detection System

A computer vision project that detects and recognizes human faces and dogs using your webcam, with audio feedback.

## Features

- **Face Detection & Recognition**: Identifies known faces from reference photos
- **Dog Detection**: Detects dogs in the camera frame
- **Audio Feedback**:
  - Speaks greetings when recognizing known faces
  - Plays bark sounds when detecting dogs
  - Plays camera shutter sound when saving images
- **Image Capture**:
  - Automatically saves unknown faces and dogs
  - Applies cooldown periods to prevent excessive captures

## Requirements

- Python 3.6 or higher
- Webcam
- The following Python packages:
  - OpenCV (`opencv-python`)
  - face_recognition
  - NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-detection.git
   cd face-detection
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python face_recognition numpy
   ```

## Setup

1. Create a `faces` directory in the project root and add clear photos of faces you want to recognize.
   - Name each file with the person's name (e.g., `john.jpg`, `sarah.jpg`)
   - Each file should have one clear face visible

2. (Optional) Add sound files to enhance the experience:
   - `camera_shutter.mp3` - Played when saving unknown faces/dogs
   - `bark.mp3` - Played when a dog is detected

   These can be placed either in the `assets/sounds` directory or the project root.

### Sound Files

You can download free sound files from these sources:

- Camera shutter sound: [Freesound - Camera Shutter Click](https://freesound.org/people/Mafon2/sounds/128111/)
- Dog bark sound: [Freesound - Dog Bark](https://freesound.org/people/KevinVG207/sounds/331912/)

After downloading, rename them to `camera_shutter.mp3` and `bark.mp3` and place in the `sunfounder/assets/sounds` directory.

## Running the Program

Run the face detection script:

```bash
python sunfounder/face_detection_mac.py
```

Or:

```
./run_face_detection.sh
```

### Parameters

The program uses the following default settings, which can be modified in the code:

- `camera_index=1`: The webcam to use (0 is usually the built-in camera)
- `process_every_n_frames=5`: Process every 5th frame for better performance
- `scale_factor=0.5`: Resize frames to 50% for faster processing
- `face_folder='faces'`: Directory containing reference face images
- `dog_folder='dogs'`: Directory for dog reference images (if any)

The program will automatically create these directories if they don't exist:
- `faces-unknown`: For storing captured unknown faces
- `dogs-unknown`: For storing captured dogs

## Usage

- Press 'q' to quit the program
- When an unknown face is detected, it will be highlighted in red and saved to `faces-unknown`
- Known faces are highlighted in green, with the person's name displayed
- Dogs are highlighted in blue and saved to `dogs-unknown`

## Cooldown Timers

The program includes several cooldown periods to manage resources:
- 10 seconds between saving unknown faces
- 10 seconds between saving dog images
- 10 seconds after all faces disappear before saving new unknown faces
- 5 seconds between greeting the same person
- 5 seconds between playing bark sounds

## Customization

- Edit cooldown periods in the code to change behavior
- For better dog detection, you can add a specialized cascade file at `assets/models/haarcascade_frontalcatface.xml`
- Adjust detection parameters like `scale_factor` and `process_every_n_frames` for different performance/accuracy tradeoffs

## Troubleshooting

- If the camera doesn't work, try changing the `camera_index` value
- For MacOS, the script uses `afplay` for audio; for Linux/Raspberry Pi it tries `aplay` and `mpg123`
- If using a Raspberry Pi, ensure you have a suitable camera connected and configured

# Face Recognition System

This system uses face recognition to identify people and can be configured to provide custom greetings and recognition messages.

## Setup

1. Create a `faces` folder and add clear photos of people you want to recognize.
   - Name each file with the person's name (e.g., `john.jpg`, `sarah.jpg`).
   - Each file should have one clear face visible.

2. The system will automatically create configuration files in `assets/config/`.

## Custom Settings

The system uses YAML for configuration. The settings file is located at:
```
assets/config/settings.yaml
```

### Custom Greetings and Messages

You can customize what the system says when recognizing specific people:

```yaml
greetings:
  # Default greeting for people without custom messages
  default: "Hi {name}"

  # Custom greetings for specific people
  # The key should match the filename (without extension) in the faces folder
  custom:
    john: "Welcome back, {name}! How was your day?"
    sarah: "Good to see you again, {name}!"

# Custom messages when a person is recognized in the logs
recognition_messages:
  # Default message for people without custom messages
  default: "Recognized: {name}"

  # Custom messages for specific people
  custom:
    john: "John has arrived! He usually works in the office."
    sarah: "Sarah is here - she's from the marketing team."
```

Notes:
- The keys (like `john` or `sarah`) must match the filename of the person's face photo (without extension).
- You can use `{name}` in the messages to include the person's name.
- Changes to the settings file will be loaded automatically the next time you run the program.

## Why YAML for Configuration?

YAML is the recommended approach for configuration in Python because:

1. It's human-readable and easy to edit
2. Supports comments (unlike JSON)
3. Doesn't require special syntax for strings and quotes
4. Standard library support via PyYAML
5. More compact than XML
6. Widely used in Python applications and frameworks

Other options include:
- JSON: Simpler but doesn't support comments
- TOML: More structured but less common
- INI: Limited in complex structures
- Python files: Powerful but can introduce security concerns# pi-car
