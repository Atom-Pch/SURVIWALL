import numpy as np
import pygame
import cv2
import mediapipe as mp
import time
import os

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

# Global variables for game state
playing_countdown = None
lives = 3
current_pose = 1
total_poses = 10
game_over = False
victory = False

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def load_pose_contour(pose_number):
    """Load a pose image and extract its contour."""
    image_path = f"assets/pose{pose_number}.png"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Unable to load {image_path}")
        return None
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load {image_path}")
        return None

    # Convert image to binary and find contours
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours were found
    if not contours:
        print(f"Error: No contours found in {image_path}")
        return None

    return [cnt.reshape(-1, 2).tolist() for cnt in contours][0]

def draw_box(image, started):
    """Draw the bounding box on the image."""
    if not started:
        # Draw recommended area (cyan)
        cv2.rectangle(
            image,
            (hole_rec_x_min, hole_rec_y_min),
            (hole_rec_x_max, hole_rec_y_max),
            (255, 255, 0),
            5,
        )
        # Draw minimum area (yellow)
        cv2.rectangle(
            image,
            (hole_limit_x_min, hole_limit_y_min),
            (hole_limit_x_max, hole_limit_y_max),
            (0, 255, 255),
            5,
        )

def check_pose_with_rectangle(image, results):
    """Check if any pose landmarks are outside the minimum area."""
    h, w, _ = image.shape
    pose_valid = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        all_landmarks_inside_hole = True  # Assume all landmarks are inside

        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if not (hole_limit_x_min <= x <= hole_limit_x_max and 
                    hole_limit_y_min <= y <= hole_limit_y_max):
                all_landmarks_inside_hole = False  # At least one landmark is outside
                break  # No need to check further

        pose_valid = not all_landmarks_inside_hole  # Valid if NOT all are inside

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

def check_pose_with_contour(image, results, contour):
    """Check if pose matches the contour."""
    h, w, _ = image.shape
    pose_valid = False

    if results.pose_landmarks and contour is not None:
        landmarks = results.pose_landmarks.landmark

        # Convert contour to numpy array for pointPolygonTest
        contour_np = np.array(contour, np.int32)
        
        skeleton_in_hole = True  # Assume skeleton is in hole
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if cv2.pointPolygonTest(contour_np, (x, y), False) < 0:
                skeleton_in_hole = False
                break  # Exit loop if any point is outside

        pose_valid = skeleton_in_hole  # Pose is valid if all points are inside

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
            "Can't see you!",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

def display_playing_content(image, contour, results):
    """Display playing content on the image, handle timer expiry, and deduct a heart."""
    global playing_countdown, lives, current_pose, game_over, victory

    # If game is over, show appropriate message and return
    if game_over:
        (GO_text_width, GO_text_height), _ = cv2.getTextSize("GAME OVER", cv2.FONT_HERSHEY_SIMPLEX, 5, 25)
        cv2.putText(
            image,
            "GAME OVER",
            ((SCREEN_WIDTH // 2) - (GO_text_width // 2), (SCREEN_HEIGHT // 2) + (GO_text_height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 255),
            25,
        )
        return
    
    if victory:
        (VIC_text_width, VIC_text_height), _ = cv2.getTextSize("VICTORY", cv2.FONT_HERSHEY_SIMPLEX, 5, 25)
        cv2.putText(
            image,
            "VICTORY",
            ((SCREEN_WIDTH // 2) - (VIC_text_width // 2), (SCREEN_HEIGHT // 2) + (VIC_text_height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 255, 0),
            25,
        )
        return

    # Start or continue the countdown timer
    if playing_countdown is None:
        playing_countdown = time.time()  # Start the countdown for the current hole

    elapsed_time = time.time() - playing_countdown
    
    # Always draw the skeleton regardless of timer
    # This is the key change - draw skeleton continuously
    pose_valid = check_pose_with_contour(image, results, contour)

    # Check if the timer has reached 5 seconds
    if elapsed_time >= 5:
        # Use the already calculated pose_valid result
        if pose_valid:
            current_pose += 1
            if current_pose > total_poses:
                victory = True
                return
        else:
            lives -= 1  # Deduct one heart
            if lives <= 0:
                game_over = True
                return
                
        playing_countdown = time.time()  # Reset the timer for the next hole
        elapsed_time = 0  # Reset elapsed time

    remaining_time = max(0, 5 - int(elapsed_time))

    # Draw the current pose contour
    if contour is not None:
        cv2.polylines(
            image, 
            [np.array(contour, np.int32)], 
            isClosed=True, 
            color=(255, 255, 0), 
            thickness=10
        )

    # Display countdown timer and pose count
    cv2.rectangle(
        image,
        (0, 0),
        (340, 180),
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
    
    # Display current pose number
    cv2.putText(
        image,
        f"Pose: {current_pose}/{total_poses}",
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
    )

    # Draw the player's remaining hearts
    # Load the heart icon
    heart_icon = cv2.imread("assets/heart.png", cv2.IMREAD_UNCHANGED)

    # Check if the heart icon was loaded correctly
    if heart_icon is None:
        print("Failed to load heart icon.")
    else:
        # Resize the heart icon if necessary
        heart_icon = cv2.resize(heart_icon, (100, 100))  # Adjust size as needed

        # Extract the alpha channel from the heart icon for transparency
        if heart_icon.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_channel = heart_icon[:, :, 3] / 255.0
            heart_icon = heart_icon[:, :, :3]  # Remove alpha channel for BGR overlay
        else:
            alpha_channel = None

        # Draw the player's remaining hearts
        for i in range(lives):
            # Calculate position for each heart
            x_offset = SCREEN_WIDTH - 125 - i * 125
            y_offset = 25

            # Define the region of interest (ROI) on the main image
            roi = image[y_offset:y_offset + 100, x_offset:x_offset + 100]

            # Overlay heart icon using alpha blending if alpha channel exists
            if alpha_channel is not None:
                for c in range(3):  # Iterate over BGR channels
                    roi[:, :, c] = (
                        alpha_channel * heart_icon[:, :, c] + (1 - alpha_channel) * roi[:, :, c]
                    )
            else:
                # Directly copy the heart icon if no alpha channel
                roi[:, :, :] = heart_icon

            # Place the modified ROI back into the main image
            image[y_offset:y_offset + 100, x_offset:x_offset + 100] = roi

def ready_to_play():
    """Run the pose detection and game loop after starting."""
    global playing_countdown, lives, current_pose, game_over, victory
    
    # Reset the game state each time the game starts
    playing_countdown = None
    lives = 3
    current_pose = 1
    game_over = False
    victory = False
    
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    countdown_start_time = None
    game_started = False  # Track if game has officially started
    loaded_pose_number = current_pose  # Track which pose is currently loaded
    
    # Load the first pose
    current_contour = load_pose_contour(current_pose)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert image to RGB for MediaPipe
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Check if we're in the preparation phase or game phase
        if not game_started:
            # Preparation phase - check if player is in position
            pose_ready = check_pose_with_rectangle(image, results)
            draw_box(image, game_started)
            display_message(image, pose_ready)
            
            # Start countdown if pose is ready
            if pose_ready and countdown_start_time is None:
                countdown_start_time = time.time()
            
            # Reset countdown if pose becomes not ready
            if not pose_ready and countdown_start_time is not None:
                countdown_start_time = None
            
            # If countdown is active, display it
            if countdown_start_time is not None:
                elapsed = time.time() - countdown_start_time
                if elapsed < 10:  # 10-second countdown
                    count = 10 - int(elapsed)
                    cv2.putText(
                        image,
                        f"Countdown: {str(count)}",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                    )
                else:
                    game_started = True  # Start the game after countdown
                    playing_countdown = time.time()  # Start the game timer
        else:
            # Game phase - check if pose matches the contour
            # If we need a new contour (after advancing to next pose)
            if current_pose != loaded_pose_number:
                current_contour = load_pose_contour(current_pose)
                loaded_pose_number = current_pose

            # Display game content (timer, hearts, current pose)
            display_playing_content(image, current_contour, results)

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
                return  # Return to main menu instead of quitting

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
def main():
    run = True

    while run:
        start_rect = draw_menu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_rect.collidepoint(pygame.mouse.get_pos()):
                    ready_to_play()

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
