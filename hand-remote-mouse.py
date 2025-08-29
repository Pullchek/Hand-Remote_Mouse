import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False

# Initialize MediaPipe hands with static mode for better performance
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only track one hand for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize OpenCV for camera capture
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
smoothing = 5
prev_x, prev_y = 0, 0

# Control parameters
CLICK_COOLDOWN = 0.5  # seconds between clicks
last_click_time = 0
click_hold = False

# Finger gesture tracking
pinch_threshold = 40  # pixels
scroll_sensitivity = 0.1


def get_finger_position(hand_landmarks, finger_id):
    x = hand_landmarks.landmark[finger_id].x
    y = hand_landmarks.landmark[finger_id].y
    return (x, y)


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_fist(hand_landmarks):
    # Check if fingers are bent
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    # Check if fingertips are below their PIPs (finger is bent)
    index_bent = index_tip.y > index_pip.y
    middle_bent = middle_tip.y > middle_pip.y
    thumb_bent = thumb_tip.x > thumb_ip.x  # For right hand

    return index_bent and middle_bent


def is_pinch(hand_landmarks):
    thumb_tip = get_finger_position(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP)
    index_tip = get_finger_position(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)

    distance = calculate_distance(
        (thumb_tip[0] * screen_width, thumb_tip[1] * screen_height),
        (index_tip[0] * screen_width, index_tip[1] * screen_height)
    )

    return distance < pinch_threshold


def is_peace_sign(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    # Check if index and middle are extended but ring is closed
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_closed = ring_tip.y > ring_pip.y

    return index_extended and middle_extended and ring_closed


# Frame rate calculation
prev_frame_time = 0
new_frame_time = 0

# Status HUD text color
text_color = (0, 255, 0)

# Motion box (for cursor control)
box_x, box_y = 100, 100
box_width, box_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) - 200), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 200)

try:
    while cap.isOpened():
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally and convert to RGB
        frame = cv2.flip(frame, 1)

        # Draw motion box
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

        # Process with MediaPipe (only process every other frame for performance)
        if int(fps) > 0 and int(new_frame_time * 10) % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger position for mouse movement
                index_x, index_y = get_finger_position(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)

                # Map to screen coordinates using motion box as reference
                cursor_x = np.interp(index_x, [box_x / frame.shape[1], (box_x + box_width) / frame.shape[1]],
                                     [0, screen_width])
                cursor_y = np.interp(index_y, [box_y / frame.shape[0], (box_y + box_height) / frame.shape[0]],
                                     [0, screen_height])

                # Smooth values
                smoothed_x = prev_x + (cursor_x - prev_x) / smoothing
                smoothed_y = prev_y + (cursor_y - prev_y) / smoothing
                prev_x, prev_y = smoothed_x, smoothed_y

                # Move cursor
                pyautogui.moveTo(smoothed_x, smoothed_y)

                # Show cursor position on frame
                cv2.circle(frame, (int(index_x * frame.shape[1]), int(index_y * frame.shape[0])), 10, (0, 255, 0), -1)

                # Gesture control
                current_time = time.time()

                # Left click with fist
                if is_fist(hand_landmarks) and current_time - last_click_time > CLICK_COOLDOWN:
                    if not click_hold:
                        pyautogui.click()
                        last_click_time = current_time
                        click_hold = True
                        cv2.putText(frame, "Left Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    click_hold = False

                # Right click with peace sign
                if is_peace_sign(hand_landmarks) and current_time - last_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    cv2.putText(frame, "Right Click!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Pinch for scrolling
                if is_pinch(hand_landmarks):
                    middle_y = get_finger_position(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)[1]
                    wrist_y = get_finger_position(hand_landmarks, mp_hands.HandLandmark.WRIST)[1]

                    # Determine scroll direction based on hand position
                    if middle_y < wrist_y:  # Hand up
                        pyautogui.scroll(int(10 * scroll_sensitivity))
                        cv2.putText(frame, "Scroll Up", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:  # Hand pointing down
                        pyautogui.scroll(int(-10 * scroll_sensitivity))
                        cv2.putText(frame, "Scroll Down", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Show status
        cv2.putText(frame, "Controls:", (10, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, "- Move index finger: Move cursor", (20, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    text_color, 1)
        cv2.putText(frame, "- Make a fist: Left click", (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    text_color, 1)
        cv2.putText(frame, "- Peace sign: Right click", (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    text_color, 1)

        # Show the processed frame
        cv2.imshow('Hand Tracking Mouse Control', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
