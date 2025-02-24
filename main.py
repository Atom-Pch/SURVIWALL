import numpy as np
import pygame
import cv2
import mediapipe as mp
import time  # Import time for countdown

# Initialize Pygame
pygame.init()

# Screen size and title
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SURVIWALL")

# Constants
LINE_THICKNESS = 30
HOLE_WIDTH = 400
HOLE_HEIGHT = 600
hole_x_min = SCREEN_WIDTH // 2 - HOLE_WIDTH // 2
hole_x_max = SCREEN_WIDTH // 2 + HOLE_WIDTH // 2
hole_y_min = SCREEN_HEIGHT // 2 - HOLE_HEIGHT // 2
hole_y_max = SCREEN_HEIGHT // 2 + HOLE_HEIGHT // 2

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)


def draw_hole(image):
    """Draw the bounding box (hole) on the image."""
    cv2.rectangle(
        image,
        (hole_x_min, hole_y_min),
        (hole_x_max, hole_y_max),
        (255, 255, 0),
        3,
    )

def process_pose(image, results):
    """Process pose landmarks and check if the skeleton is within the hole."""
    h, w, _ = image.shape
    pose_ready = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Check if all landmarks are within the hole
        skeleton_in_hole = True
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if not (hole_x_min <= x <= hole_x_max and hole_y_min <= y <= hole_y_max):
                skeleton_in_hole = False
                break

        # Divide the hole into 5 rows
        row_height = (hole_y_max - hole_y_min) // 5
        row_1_min = hole_y_min
        row_1_max = hole_y_min + row_height
        row_5_min = hole_y_max - row_height
        row_5_max = hole_y_max

        # Get the y-coordinates of the head and feet landmarks
        head_y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * h)
        left_foot_y = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)
        right_foot_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h)

        # Check if the head is in the first row and the feet are in the last row
        head_in_first_row = row_1_min <= head_y <= row_1_max
        feet_in_last_row = (
            row_5_min <= left_foot_y <= row_5_max and row_5_min <= right_foot_y <= row_5_max
        )

        # Combine all conditions
        pose_ready = skeleton_in_hole and head_in_first_row and feet_in_last_row

        # Draw skeleton
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            color = (0, 255, 0) if pose_ready else (0, 0, 255)
            cv2.line(image, (x1, y1), (x2, y2), color, LINE_THICKNESS)

    return pose_ready

def display_message(image, pose_ready):
    """Display a message on the image based on pose readiness."""
    if pose_ready:
        cv2.putText(
            image,
            "Hold still!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
    else:
        cv2.putText(
            image,
            "Stand in the box!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

def pose_detection():
    """Run the pose detection loop."""
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    countdown_start_time = None  # Variable to track the start of the countdown

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert image to RGB for MediaPipe
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Process pose and check if it's ready
        pose_ready = process_pose(image, results)

        # Handle countdown logic
        if pose_ready:
            if countdown_start_time is None:
                countdown_start_time = time.time()  # Start the countdown
            elapsed_time = time.time() - countdown_start_time
            remaining_time = max(0, 5 - int(elapsed_time))  # Calculate remaining time

            if remaining_time == 0:
                # Pose held for 5 seconds, display success message
                cv2.putText(
                    image,
                    "Game Start!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 255),
                    5,
                )
        else:
            countdown_start_time = None  # Reset the countdown if pose is incorrect
            remaining_time = 5  # Reset the countdown to 5 seconds

        # Draw the hole and display messages
        draw_hole(image)
        display_message(image, pose_ready)

        # Display countdown timer on the screen
        cv2.putText(
            image,
            f"Countdown: {remaining_time} !",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )

        # Convert the OpenCV frame to Pygame surface
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))

        # Display the webcam frame on the Pygame window
        screen.blit(frame_surface, (0, 0))
        pygame.display.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                exit()

def draw_menu():
    """Draw the main menu."""
    screen.fill((0, 0, 0))  # Clear the screen with black

    # Title
    font = pygame.font.Font(None, 36)
    title = font.render("SURVIWALL", False, (255, 255, 255))
    title_rect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
    screen.blit(title, title_rect)

    # Start button
    font = pygame.font.Font(None, 24)
    start_btn = font.render("START", False, (255, 255, 255))
    start_rect = start_btn.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

    # Give the button a visible box and border
    pygame.draw.rect(screen, (0, 128, 0), start_rect.inflate(40, 40), 2)

    # Hover over and click effects
    if start_rect.collidepoint(pygame.mouse.get_pos()):
        if pygame.mouse.get_pressed()[0]:
            pygame.draw.rect(screen, (0, 64, 0), start_rect.inflate(30, 30), 0)
        else:
            pygame.draw.rect(screen, (0, 192, 0), start_rect.inflate(30, 30), 0)

    screen.blit(start_btn, start_rect)

    return start_rect

"""Main game loop."""
run = True

while run:
    start_rect = draw_menu()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if start_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            pose_detection()

    pygame.display.update()

pygame.quit()
