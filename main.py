import pygame, cv2
import mediapipe as mp

pygame.init()

# screen size and title
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SURVIWALL")

run = True

while run:
    # main menu
    font = pygame.font.Font(None, 36)
    text = font.render("SURVIWALL", False, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))
    screen.blit(text, text_rect)

    # start button
    font = pygame.font.Font(None, 24)
    text = font.render("START", False, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
    screen.blit(text, text_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        # started
        if text_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            mp_pose = mp.solutions.pose
            mp_draw = mp.solutions.drawing_utils
            pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_draw.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            pose.close()
            cap.release()
    
    pygame.display.update()

pygame.quit()
