import cv2
import numpy as np
from datetime import datetime

class BiscuitQualityAnalyzer:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.color_range = None  # Will be set via calibration
        self.size_range = None   # Will be auto-calculated
        self.expected_count = 10
        self.quality_log = []

    def calibrate(self, calibration_img):
        """Run this first with an image containing good biscuits"""
        print("Click on several GOOD biscuits to calibrate color range")
        hsv = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2HSV)
        samples = []
        
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                samples.append(hsv[y,x])
                cv2.circle(calibration_img, (x,y), 5, (0,255,0), -1)
                cv2.imshow("Calibration", calibration_img)
        
        cv2.imshow("Calibration", calibration_img)
        cv2.setMouseCallback("Calibration", on_mouse_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if samples:
            samples = np.array(samples)
            mean = np.mean(samples, axis=0)
            std = np.std(samples, axis=0)
            self.color_range = {
                'lower': np.maximum(0, mean - 1.5*std),
                'upper': np.minimum(255, mean + 1.5*std)
            }
            print(f"Calibrated color range: Lower {self.color_range['lower']}, Upper {self.color_range['upper']}")

    def detect_biscuits(self, frame):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Use calibrated color range or fallback to defaults
        if self.color_range:
            mask = cv2.inRange(hsv, self.color_range['lower'], self.color_range['upper'])
        else:
            # Default range for light brown biscuits
            mask = cv2.inRange(hsv, np.array([10,50,50]), np.array([30,255,255]))
        
        # Enhance mask quality
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Auto-calculate size thresholds if not set
        if not self.size_range:
            self._auto_set_size_thresholds(frame)
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.size_range['min'] < area < self.size_range['max']:
                # Additional circularity check
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter**2)
                    if circularity > 0.5:  # Adjust for your biscuits
                        valid_contours.append(cnt)
        
        return valid_contours
    
    def _auto_set_size_thresholds(self, frame):
        """Estimate biscuit size based on package dimensions"""
        height, width = frame.shape[:2]
        
        # Assumptions (adjust these!)
        package_width_mm = 180       # Physical width of package in mm
        biscuit_diameter_mm = 45     # Typical biscuit size
        
        # Calculate pixels per mm
        px_per_mm = width / package_width_mm
        biscuit_radius_px = (biscuit_diameter_mm / 2) * px_per_mm
        expected_area = np.pi * (biscuit_radius_px ** 2)
        
        self.size_range = {
            'min': max(100, 0.3 * expected_area),  # Lower bound for broken pieces
            'max': 1.7 * expected_area             # Upper bound for overlapping
        }
        print(f"Auto-set size range: {self.size_range}")

    def process_frame(self, frame):
        biscuits = self.detect_biscuits(frame)
        
        # Count and quality check
        biscuit_count = len(biscuits)
        quality_score = self._check_quality(frame, biscuits)
        
        result = {
            'timestamp': datetime.now(),
            'count': biscuit_count,
            'quality': quality_score,
            'passed': biscuit_count == self.expected_count and quality_score > 0.7
        }
        self.quality_log.append(result)
        
        # Visual feedback
        self._draw_results(frame, biscuits, result)
        return frame, result
    
    def _check_quality(self, frame, contours):
        if not contours:
            return 1.0  # Default to perfect if no biscuits (edge case)
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        quality_scores = []
    
        for cnt in contours:
            # 1. Create biscuit mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
        
            # 2. Focus on Hue (color) and Saturation (vibrancy)
            hue = hsv[:,:,0][mask == 255]
            sat = hsv[:,:,1][mask == 255]
        
            # 3. Calculate "goodness" based on calibrated color
            if self.color_range:
                # Percentage of pixels within ideal Hue range
                hue_min, hue_max = self.color_range['lower'][0], self.color_range['upper'][0]
                hue_score = np.mean((hue >= hue_min) & (hue <= hue_max))
                
                # Reward high saturation (vibrant color)
                sat_score = np.mean(sat) / 255.0
            else:
                # Fallback: brightness consistency
                val = hsv[:,:,2][mask == 255]
                hue_score = 1.0  # Assume correct hue
                sat_score = np.std(val) / 255.0  # Lower std = more consistent
            
            # 4. Combined score (weighted)
            quality_scores.append(0.7 * hue_score + 0.3 * sat_score)
        
        # Return worst-case score (a single bad biscuit fails the package)
        return np.min(quality_scores) if quality_scores else 1.0
    
    def _draw_results(self, frame, contours, result):
        # Draw contours
        cv2.drawContours(frame, contours, -1, (0,255,0), 2)
        
        # Display info
        y_offset = 30
        for key, val in result.items():
            if key == 'timestamp':
                text = f"{key}: {val.strftime('%H:%M:%S')}"
            elif isinstance(val, float):
                text = f"{key}: {val:.2f}"
            else:
                text = f"{key}: {val}"
                
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0,255,0) if result['passed'] else (0,0,255), 2)
            y_offset += 30

    def run_test(self, test_img_path):
        """Test with static images"""
        img = cv2.imread(test_img_path)
        if img is None:
            print(f"Error: Could not load image {test_img_path}")
            return
            
        # First-time calibration if needed
        if self.color_range is None:
            self.calibrate(img.copy())
            
        processed, result = self.process_frame(img)
        
        print("\nTest Results:")
        for k, v in result.items():
            print(f"{k:15}: {str(v):20}")
            
        cv2.imshow("Result", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==================== USAGE EXAMPLES ==================== 
if __name__ == "__main__":
    analyzer = BiscuitQualityAnalyzer()
    
    # Option 1: Calibrate with a sample image
    analyzer.calibrate(cv2.imread("test_images\\Biscuit5.png"))
    
    #Option 2: Test with individual images
    analyzer.run_test("test_images\\Biscuit5.png")
    #analyzer.run_test("real_biscuits_2.jpg")
    
    # Option 3: Process live camera feed
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     processed, _ = analyzer.process_frame(frame)
    #     cv2.imshow("Live QA", processed)
    #     if cv2.waitKey(1) == ord('q'): break
    # cap.release()
    # cv2.destroyAllWindows()