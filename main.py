import pygame, cv2
import mediapipe as mp

pygame.init()

# screen size and title
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SURVIWALL")

run = True

def poseDetection():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and convert image to RGB for Mediapipe
        image = cv2.flip(image, 1)
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(new_image)

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', image)

        # Exit if ESC is pressed
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    del pose  # Properly release resources






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
