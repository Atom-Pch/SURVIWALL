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
            cv2.putText(image, "Incorrect Pose!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Convert the OpenCV frame to Pygame surface
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(image_rgb, (1, 0, 2)))

        # Display the webcam frame on the Pygame window
        screen.blit(frame_surface, (0, 0))

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
