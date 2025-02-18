import numpy as np
import pygame, cv2
import mediapipe as mp

pygame.init()

# screen size and title
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SURVIWALL")

run = True

def point_in_poly(x, y, poly):
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def poseDetection():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)  # Open webcam

    # Define the hole region in screen coordinates (adjust as needed)
    hole_polygon = [(150, 150), (350, 150), (350, 450), (150, 450)]
    # Landmarks to check: shoulders and hips
    required_indices = [11, 12, 23, 24]

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert image to RGB for MediaPipe
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks on the image for visualization
        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check if required keypoints are within the hole (after converting coordinates)
            correct_count = 0
            for idx in required_indices:
                landmark = results.pose_landmarks.landmark[idx]
                # Convert normalized coordinates to pixel coordinates (assuming a 640x480 capture)
                # Adjust these values if your capture resolution is different.
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                # Simple check: draw a circle where the landmark is (for debugging)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                if point_in_poly(x, y, hole_polygon):
                    correct_count += 1

            if correct_count == len(required_indices):
                cv2.putText(image, "Correct Pose!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optionally, draw the hole on the OpenCV window for reference
        cv2.polylines(image, [np.array(hole_polygon, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit pose detection
            break

    cap.release()
    cv2.destroyAllWindows()
    del pose







# game loop
while run:
    # main menu
    font = pygame.font.Font(None, 36)
    title = font.render("SURVIWALL", False, (255, 255, 255))
    titleRect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
    screen.blit(title, titleRect)

    # start button
    font = pygame.font.Font(None, 24)
    startBtn = font.render("START", False, (255, 255, 255))
    startRect = startBtn.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    screen.blit(startBtn, startRect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        # pose detection
        if startRect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            poseDetection()
    
    pygame.display.update()

pygame.quit()
