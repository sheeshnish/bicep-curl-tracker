import cv2 
import mediapipe as mp
import numpy as np  

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 1. Try changing the index to 0, 1, or -1 if your camera doesn't load
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW forces direct video show on Windows to prevent freezing

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

def calculate_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle  

def calculate_distance(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

counter = 0
rep_state = "down"
# Lowering confidence slightly to ensure it catches your body easily at first
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  
    print("MediaPipe Pose Model Initialized successfully. Starting loop...")
    
    while cap.isOpened():
        ret, frame = cap.read()  
        if not ret:
            print("Failed to grab frame.")
            continue
            
        # Flip and get dimensions
        frame = cv2.flip(frame, 1)  
        h, w, _ = frame.shape

        # MediaPipe requires RGB, OpenCV uses BGR. 
        # Making a clean copy prevents memory locking issues.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw landmarks if they exist
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                LShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                LElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                LWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                
                # Calculate angle
                angle = calculate_angle(LWrist, LElbow, LShoulder)
                distance = calculate_distance(LWrist,LShoulder)
            
                # Convert normalized coordinates back to pixel values for display
                elbow_pixels = tuple(np.multiply(LElbow, [w, h]).astype(int))
                wrist_pixels = tuple(np.multiply(LWrist, [w, h]).astype(int))
                
                print(distance)
                
                # Render text to screen
                cv2.putText(frame, f"{int(angle)} deg", elbow_pixels, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.putText(frame, f"{int(counter)} reps", wrist_pixels, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                if angle < 40.0 and distance < 0.2:
                    rep_state = "down"
                    
                if angle > 130.0 and distance > 0.4 and rep_state == "down":
                    rep_state = "up"
                    counter += 1

                print(rep_state, counter)

            except Exception as e:
                # Print the error to your console so you know exactly what failed inside the math
                print(f"Tracking error: {e}")
        else:
            # If the model doesn't see a human body, it will print this continuously
            cv2.putText(frame, "No Pose Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Pushup Tracker Feed", frame)  
        
        # Press 'q' to quit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

cap.release()
cv2.destroyAllWindows()
print("Program closed cleanly.")