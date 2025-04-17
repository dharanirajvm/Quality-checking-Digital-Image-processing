import streamlit as st
import cv2
import numpy as np
from biscuit2 import BiscuitQualityAnalyzer
from bottle import BottleQualityAnalyzer  # Save your code in this file

# Initialize analyzers in session state
if 'biscuit_analyzer' not in st.session_state:
    st.session_state.biscuit_analyzer = BiscuitQualityAnalyzer()

if 'bottle_analyzer' not in st.session_state:
    st.session_state.bottle_analyzer = BottleQualityAnalyzer()

st.set_page_config(page_title="Quality Analyzer", layout="centered")

st.title("🧪 Smart Quality Analyzer")
analyzer_type = st.selectbox("Select Product Type", ["Biscuits", "Bottles"])

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Uploaded Image", channels="BGR", use_column_width=True)

    if analyzer_type == "Biscuits":
        analyzer = st.session_state.biscuit_analyzer

        col1, col2 = st.columns(2)
        if col1.button("🔧 Calibrate"):
            st.info("Click on the calibration window and select good biscuits")
            analyzer.calibrate(image.copy())
            st.success("Calibration done!")

        if col2.button("🧪 Run Biscuit Analysis"):
            if analyzer.color_range is None:
                st.warning("Please calibrate first!")
            else:
                processed_img, result = analyzer.process_frame(image.copy())
                st.image(processed_img, caption="Analyzed Biscuits", channels="BGR")
                st.markdown(f"""
                **🕒 Timestamp:** `{result['timestamp']}`  
                **🔢 Count:** `{result['count']}`  
                **🎯 Quality Score:** `{result['quality']:.2f}`  
                **✅ Passed:** {'Yes ✅' if result['passed'] else 'No ❌'}
                """)

    else:
        analyzer = st.session_state.bottle_analyzer

        if st.button("🧪 Run Bottle Analysis"):
            processed_img, results = analyzer.process_frame(image.copy())
            st.image(processed_img, caption="Analyzed Bottles", channels="BGR")

            for i, result in enumerate(results):
                st.subheader(f"Bottle {i+1}: {'✅ PASS' if result['passed'] else '❌ FAIL'}")
                st.markdown(f"""
                - **Fill Ratio**: `{result['fill_ratio']:.2f}`
                - **Color Score**: `{result['color_score']:.2f}`
                - **Cap Present**: `{result['has_cap']}`
                """)

