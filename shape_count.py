import numpy as np
import cv2

# Load Image
img = cv2.imread('data\\industrial-die-casting-components-648.jpg')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.THRESH_BINARY)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Dictionary to Store Counts
shape_count = {
    'Triangle': 0,
    'Square': 0,
    'Rectangle': 0,
    'Pentagon': 0,
    'Circle': 0
}

# Loop over contours
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
    
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    
    sides = len(approx)

    if sides == 3:
        shape_name = "Triangle"
    elif sides == 4:
        x_, y_, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if 0.95 <= aspectRatio < 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif sides == 5:
        shape_name = "Pentagon"
    else:
        shape_name = "Circle"

    shape_count[shape_name] += 1
    cv2.putText(img, shape_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

# Display Count on Image
y0 = 20
for shape, count in shape_count.items():
    text = f"{shape} : {count}"
    cv2.putText(img, text, (10, y0), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
    y0 += 20

# Show Result
cv2.imshow("Shape Detection & Counting", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
