import cv2
import time

def test_camera(camera_index=1):
    print(f"Testing camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

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

    # Show video feed
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lost connection to camera.")
            break

        cv2.imshow(f"Camera Test (Index: {camera_index})", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# Just use the default camera index (1)
test_camera()

# Uncomment this section if you want to try multiple camera indices
# for index in range(3):  # Try indices 0, 1, and 2
#     if test_camera(index):
#         break
#     print("Trying next camera index...")
#     time.sleep(1)
