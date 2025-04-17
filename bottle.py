import cv2
import numpy as np

class BottleQualityAnalyzer:
    def __init__(self):
        self.min_fill_height_ratio = 0.85
        self.color_variation_threshold = 0.15
        self.expected_hue_range = (10, 100)  # Wider hue range
        self.cap_detection_area = 300  # Lower for top-view caps

    def process_frame(self, frame):
        bottles = self._detect_bottles(frame)
        results = []

        for i, bottle in enumerate(bottles):
            fill_ratio = self._check_fill_level(frame, bottle)
            color_score = self._check_color_consistency(frame, bottle)
            has_cap = self._check_cap_presence(frame, bottle)

            result = {
                'fill_ratio': fill_ratio,
                'color_score': color_score,
                'has_cap': has_cap,
                'passed': (fill_ratio >= self.min_fill_height_ratio and 
                          color_score >= (1 - self.color_variation_threshold) and 
                          has_cap)
            }

            results.append(result)
            self._draw_results(frame, bottle, result, index=i+1)

        return frame, results

    def _detect_bottles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        cv2.imshow("edged:",edged)

        # Apply morphological closing to reduce small gaps/reflections
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)

            if area > 10000 and aspect_ratio > 1.5:  # Only large, tall objects
                bottles.append({
                    'bounding_box': (x, y, w, h),
                    'neck_position': (x + w // 2, y + int(h * 0.08))
                })

        bottles = sorted(bottles, key=lambda b: b['bounding_box'][0])
        return bottles

    def _check_fill_level(self, frame, bottle):
        x, y, w, h = bottle['bounding_box']
        roi = frame[y:y + h, x:x + w]

        # Ignore top/bottom 10% for more accurate detection
        trim_top = int(h * 0.1)
        trim_bottom = int(h * 0.9)
        roi = roi[trim_top:trim_bottom]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Adjust expected hue range per liquid color
        lower = np.array([self.expected_hue_range[0], 50, 50])
        upper = np.array([self.expected_hue_range[1], 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean small blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Vertical histogram of filled pixels
        row_sums = np.sum(mask > 0, axis=1).astype(np.float32) / w  # Normalize

        # Smooth to reduce noise
        row_sums_smooth = np.convolve(row_sums, np.ones(5)/5, mode='same')

        # Determine fill threshold dynamically
        threshold = 0.3  # row must be at least 30% filled
        filled_rows = np.where(row_sums_smooth > threshold)[0]

        if len(filled_rows) == 0:
            return 0.0

        # Fill height is from bottom to top of last filled region
        fill_height = len(row_sums_smooth) - filled_rows[0]
        fill_ratio = fill_height / len(row_sums_smooth)

        return round(fill_ratio, 2)


    def _check_color_consistency(self, frame, bottle):
        x, y, w, h = bottle['bounding_box']
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, 
                           np.array([self.expected_hue_range[0], 50, 50]), 
                           np.array([self.expected_hue_range[1], 255, 255]))
        hue_values = hsv[:, :, 0][mask > 0]
        if len(hue_values) > 0:
            std_dev = np.std(hue_values) / 180.0
            return 1.0 - std_dev
        return 0.0

    def _check_cap_presence(self, frame, bottle):
        x, y = bottle['neck_position']
        h, w = frame.shape[:2]

        if y - 20 < 0 or x - 20 < 0 or x + 20 > w or y + 20 > h:
            return False  # Skip if neck region is out of bounds

        neck_roi = frame[y - 20:y + 20, x - 20:x + 20]
        gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        cap_area = cv2.countNonZero(thresh)

        return cap_area > self.cap_detection_area

    def _draw_results(self, frame, bottle, result, index):
        x, y, w, h = bottle['bounding_box']
        color = (0, 255, 0) if result['passed'] else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        fill_height = int(y + h * (1 - result['fill_ratio']))
        cv2.line(frame, (x, fill_height), (x+w, fill_height), (255, 0, 0), 2)

        info = f"Bottle {index}: {'PASS' if result['passed'] else 'FAIL'}"
        cv2.putText(frame, info, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        fill_info = f"Fill: {result['fill_ratio']:.0%} Color: {result['color_score']:.2f} Cap: {'Yes' if result['has_cap'] else 'No'}"
        cv2.putText(frame, fill_info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# === MAIN ===
if __name__ == "__main__":
    analyzer = BottleQualityAnalyzer()

    # Use generated image
    image_path = "test_bottle\\bottle 8.png"
    img = cv2.imread(image_path)

    processed_img, results = analyzer.process_frame(img)

    for i, result in enumerate(results):
        print(f"Bottle {i+1}: {'PASS' if result['passed'] else 'FAIL'}")
       
        print(f"  Fill Ratio: {result['fill_ratio']:.2f}")
        print(f"  Color Consistency: {result['color_score']:.2f}")
        print(f"  Cap Present: {result['has_cap']}\n")

    cv2.imshow("Inspection Result", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()