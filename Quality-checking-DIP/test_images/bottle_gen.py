import cv2
import numpy as np

# Create blank image
img = np.zeros((600, 800, 3), dtype=np.uint8) + 255  # White background

# Draw conveyor belt
cv2.rectangle(img, (0, 450), (800, 600), (200, 200, 200), -1)

# Bottle drawing function
def draw_bottle(img, x, fill_ratio=0.9, color=(200, 100, 0), has_cap=True):
    # Bottle outline
    cv2.rectangle(img, (x, 300), (x+60, 450), (0, 0, 0), 2)
    
    # Liquid fill
    fill_height = int(300 + (450-300) * (1 - fill_ratio))
    cv2.rectangle(img, (x+5, fill_height), (x+55, 445), color, -1)
    
    # Neck
    cv2.rectangle(img, (x+20, 250), (x+40, 300), (0, 0, 0), 2)
    
    # Cap
    if has_cap:
        cv2.rectangle(img, (x+15, 230), (x+45, 250), (0, 0, 255), -1)

# Draw bottles
draw_bottle(img, 100, 0.95)  # Good
draw_bottle(img, 200, 0.65)  # Underfilled
draw_bottle(img, 300, 1.1, (150, 50, 0))  # Overfilled + color variation
draw_bottle(img, 400, 0.9, (200, 100, 0), False)  # Missing cap
draw_bottle(img, 500, 0.92)  # Good
draw_bottle(img, 600, 0.88)  # Good

# Save
cv2.imwrite("bottling_test_image.jpg", img)
print("Test image generated: bottling_test_image.jpg")