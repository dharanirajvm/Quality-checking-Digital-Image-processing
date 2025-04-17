import streamlit as st
import cv2
import numpy as np
from biscuit2 import BiscuitQualityAnalyzer  # your class file

# Create or retrieve the analyzer object from session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = BiscuitQualityAnalyzer()

analyzer = st.session_state.analyzer

st.title("🍪 Biscuit Quality Analyzer")

# Upload image
uploaded_image = st.file_uploader("Upload an image of biscuits", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    if st.button("🔧 Calibrate"):
        st.info("Please click on the calibration window and select good biscuits")
        analyzer.calibrate(image.copy())
        st.success("Calibration done!")

    if st.button("🧪 Run Analysis"):
        if analyzer.color_range is None:
            st.warning("Please calibrate first!")
        else:
            processed_frame, result = analyzer.process_frame(image.copy())
            st.image(processed_frame, caption="Analysis Result", channels="BGR")

            st.write("### 📋 Result")
            st.json({
                "Timestamp": str(result['timestamp']),
                "Count": result['count'],
                "Quality Score": f"{result['quality']:.2f}",
                "Passed": result['passed']
            })
