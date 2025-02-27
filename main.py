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
LINE_THICKNESS = 20

HOLE_WIDTH_LIMIT = 200
HOLE_HEIGHT_LIMIT = 400

hole_limit_x_min = SCREEN_WIDTH // 2 - HOLE_WIDTH_LIMIT // 2
hole_limit_x_max = SCREEN_WIDTH // 2 + HOLE_WIDTH_LIMIT // 2
hole_limit_y_min = SCREEN_HEIGHT // 2 - HOLE_HEIGHT_LIMIT // 2
hole_limit_y_max = SCREEN_HEIGHT // 2 + HOLE_HEIGHT_LIMIT // 2

HOLE_WIDTH_REC = 400
HOLE_HEIGHT_REC = 600

hole_rec_x_min = SCREEN_WIDTH // 2 - HOLE_WIDTH_REC // 2
hole_rec_x_max = SCREEN_WIDTH // 2 + HOLE_WIDTH_REC // 2
hole_rec_y_min = SCREEN_HEIGHT // 2 - HOLE_HEIGHT_REC // 2
hole_rec_y_max = SCREEN_HEIGHT // 2 + HOLE_HEIGHT_REC // 2

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def draw_hole(image, started):
    """Draw the bounding box (hole) on the image."""
    if not started:
        cv2.rectangle(
            image,
            (hole_limit_x_min, hole_limit_y_min),
            (hole_limit_x_max, hole_limit_y_max),
            (0, 255, 255),
            5,
        )
        cv2.rectangle(
            image,
            (hole_rec_x_min, hole_rec_y_min),
            (hole_rec_x_max, hole_rec_y_max),
            (255, 255, 0),
            5,
        )

def check_pose(image, results):
    """Process pose landmarks and check if the skeleton is not too far."""
    h, w, _ = image.shape
    pose_valid = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        all_landmarks_inside_hole = True  # Assume all landmarks are inside

        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if not (hole_limit_x_min <= x <= hole_limit_x_max and hole_limit_y_min <= y <= hole_limit_y_max):
                all_landmarks_inside_hole = False  # At least one landmark is outside
                break  # No need to check further, countdown continues

        pose_valid = not all_landmarks_inside_hole  # Countdown resets if ALL are inside


        # Draw skeleton
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            color = (0, 255, 0) if pose_valid else (0, 0, 255)
            cv2.line(image, (x1, y1), (x2, y2), color, LINE_THICKNESS)

    return pose_valid

def display_message(image, pose_ready):
    """Display a message on the image based on pose readiness."""
    if pose_ready:
        cv2.putText(
            image,
            "Get ready!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
    else:
        cv2.putText(
            image,
            "Too far, come closer!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

playing_countdown = None
def display_playing_content(image):
    """Display playing content on the image."""
    global playing_countdown

    if playing_countdown is None:
        playing_countdown = time.time()  # Start the countdown

    elapsed_time = time.time() - playing_countdown
    remaining_time = max(0, 5 - int(elapsed_time))

    if remaining_time == 0:
        text_size = cv2.getTextSize("Game Over", cv2.FONT_HERSHEY_SIMPLEX, 5, 5)[0]
        text_x = (SCREEN_WIDTH - text_size[0]) // 2
        text_y = (SCREEN_HEIGHT - text_size[1]) // 2
        cv2.putText(
            image,
            "Game Over",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 255),
            20,
        )
    
    # Display countdown timer
    cv2.rectangle(
        image,
        (0, 0),
        (300, 100),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        f"Time: {remaining_time}",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        5,
    )

def ready_to_play():
    """Run the pose detection loop."""
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    countdown_start_time = None
    game_started = False  # Track if game has officially started

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
        pose_ready = check_pose(image, results)

        if not game_started:
            # Handle countdown logic
            if pose_ready:
                if countdown_start_time is None:
                    countdown_start_time = time.time()  # Start the countdown
                elapsed_time = time.time() - countdown_start_time
                remaining_time = max(0, 5 - int(elapsed_time))

                if remaining_time == 0:
                    game_started = True  # Mark game as started

            else:
                countdown_start_time = None  # Reset countdown
                remaining_time = 5  # Reset to 5 seconds

            # Draw hole and display messages
            draw_hole(image, game_started)
            display_message(image, pose_ready)

            # Display countdown timer
            cv2.putText(
                image,
                f"Countdown: {remaining_time}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )

        else:
            draw_hole(image, game_started)
            # start the game
            display_playing_content(image)

        # Convert the OpenCV frame to Pygame surface
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))

        # Display the frame in Pygame
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
            ready_to_play()

    pygame.display.update()

pygame.quit()
