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

run = True

def poseDetection():
    # MediaPipe Pose Setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 1280)  # Set the width
    cap.set(4, 720)  # Set the height

    # ขนาดของเส้นโครงกระดูกที่ขยายขึ้น 100 เท่า
    LINE_THICKNESS = 30     

    # โหลดภาพ Hole และตรวจสอบว่าภาพโหลดสำเร็จ
    image = cv2.imread(r"assets/test2.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load test.png")
        return

    # แปลงภาพเป็น Binary และหา Contours
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ตรวจสอบว่ามี Contours หรือไม่
    if not contours:
        print("Error: No contours found in test.png")
        return

    hole_polygon = [cnt.reshape(-1, 2).tolist() for cnt in contours][0]
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

# Global variables for game state
playing_countdown = None
lives = 3

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
    cv2.rectangle(
        image,
        (0, 0),
        (400, 150),
        (0, 0, 0),
        -1,
    )
    if pose_ready:
        cv2.putText(
            image,
            "Get ready!",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
    else:
        cv2.putText(
            image,
            "Come closer!",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

def display_playing_content(image):
    """Display playing content on the image, handle timer expiry, and deduct a heart."""
    global playing_countdown, lives

    # If no lives remain, show GAME OVER and return
    if lives <= 0:
        cv2.putText(
            image,
            "GAME OVER",
            (SCREEN_WIDTH // 2 - 450, SCREEN_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 255),
            25,
        )
        return

    # Start or continue the countdown timer
    if playing_countdown is None:
        playing_countdown = time.time()  # Start the countdown for the current hole

    elapsed_time = time.time() - playing_countdown

    # Check if the timer has reached 5 seconds
    if elapsed_time >= 5:
        lives -= 1  # Deduct one heart
        print("next hole")  # Debug message for now
        playing_countdown = time.time()  # Reset the timer for the next hole
        elapsed_time = 0  # Reset elapsed time

    remaining_time = max(0, 5 - int(elapsed_time))

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

    # Draw the player's remaining hearts
    for i in range(lives):
        cv2.rectangle(
            image,
            (SCREEN_WIDTH - 100 - i * 100, 20),
            (SCREEN_WIDTH - 40 - i * 100, 80),
            (0, 0, 225),
            -1,
        )
        cv2.rectangle(
            image,
            (SCREEN_WIDTH - 100 - i * 100, 20),
            (SCREEN_WIDTH - 40 - i * 100, 80),
            (0, 0, 0),
            5,
        )

def ready_to_play():
    """Run the pose detection and game loop after starting."""

    # Reset the game state each time the game starts
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

        pose_correct = False  # ตัวแปรเก็บสถานะว่าท่าถูกต้องหรือไม่

        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark

            skeleton_in_hole = True  # สมมติว่าโครงกระดูกอยู่ใน Hole แล้วตรวจสอบ
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if cv2.pointPolygonTest(np.array(hole_polygon, np.int32), (x, y), False) < 0:
                    skeleton_in_hole = False
                    break  # ออกจากลูปทันทีถ้าพบว่ามีจุดออกนอกขอบเขต

            if skeleton_in_hole:
                pose_correct = True  # ท่าถูกต้อง

            # วาด Skeleton และเส้นโครงกระดูก
            skeleton_connections = mp_pose.POSE_CONNECTIONS
            for connection in skeleton_connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                color = (0, 255, 0) if pose_correct else (0, 0, 255)  # สีเขียวถ้าถูก, แดงถ้าผิด
                cv2.line(image, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        # วาดกรอบ Hole บนภาพ
        cv2.polylines(image, [np.array(hole_polygon, np.int32)], isClosed=True, color=(255, 255, 0), thickness=3)

        # แสดงข้อความแจ้งเตือนว่าท่าถูกต้องหรือไม่
        if pose_correct:
            cv2.putText(image, "Correct Pose!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            draw_hole(image, game_started)
            # In game: update playing content (timer, hearts, etc.)
            display_playing_content(image)

        # Convert the OpenCV frame to a Pygame surface and display it
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))
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
    font = pygame.font.Font(None, 108)
    title = font.render("S U R V I W A L L", False, (255, 255, 255))
    title_rect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
    screen.blit(title, title_rect)

    # Start button
    font = pygame.font.Font(None, 48)
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
