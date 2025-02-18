import numpy as np
import pygame
import cv2
import mediapipe as mp

pygame.init()

# screen size and title
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SURVIWALL")

run = True

def poseDetection():
    # MediaPipe Pose Setup
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 1280)  # Set the width
    cap.set(4, 720)  # Set the height

    # Define the range (rectangle) for checking poses
    region_x_min = SCREEN_WIDTH // 2 - 100
    region_x_max = SCREEN_WIDTH // 2 + 200
    region_y_min = SCREEN_HEIGHT // 2 - 200
    region_y_max = SCREEN_HEIGHT // 2 + 200

    # Load an image (replace 'image.png' with the actual image file path)
    image_overlay = pygame.image.load("E:\CVlize\SURVIWALL\hole.png")
    image_overlay = pygame.transform.scale(image_overlay, (700, 700))  # Resize the image to fit on screen
    image_overlay.set_alpha(128)  # Set the transparency level (0-255)
 

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert image to RGB for MediaPipe
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            correct_count = 0
            required_indices = [11, 12, 23, 24]  # Shoulders and hips
            for idx in required_indices:
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

                # Check if the landmark is inside the defined range
                if region_x_min <= x <= region_x_max and region_y_min <= y <= region_y_max:
                    correct_count += 1

            if correct_count == len(required_indices):
                cv2.putText(image, "Correct Pose!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the rectangular range (region)
        cv2.rectangle(image, (region_x_min, region_y_min), (region_x_max, region_y_max), (0, 255, 0), 2)

        # Convert the OpenCV frame to Pygame surface
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))

        # Display the webcam frame on the Pygame window
        screen.blit(frame_surface, (0, 0))

        # Overlay the image on top of the webcam feed (you can change the position as needed)
        x = (SCREEN_WIDTH - image_overlay.get_width()) // 2
        y = (SCREEN_HEIGHT - image_overlay.get_height()) // 2
        # วางรูปภาพที่ตำแหน่งกลางของหน้าจอ
        screen.blit(image_overlay, (x, y))

        pygame.display.update()

        # Break if ESC is pressed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                exit()

# Game loop (pygame)
while run:
    font = pygame.font.Font(None, 36)
    title = font.render("SURVIWALL", False, (255, 255, 255))
    titleRect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
    screen.blit(title, titleRect)

    font = pygame.font.Font(None, 24)
    startBtn = font.render("START", False, (255, 255, 255))
    startRect = startBtn.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    screen.blit(startBtn, startRect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if startRect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            poseDetection()

    pygame.display.update()

pygame.quit()
