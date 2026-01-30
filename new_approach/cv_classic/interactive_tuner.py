# interactive_tuner.py
"""
Interactive parameter tuning UI for curve extraction.
Run with: streamlit run interactive_tuner.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from curve_extraction_cv import (
    ExtractionParams,
    full_extraction_pipeline,
    visualize_curves,
    remove_grid_lines,
    preprocess_for_skeletonization
)


st.set_page_config(page_title="Curve Extraction Tuner", layout="wide")

st.title("🔧 Curve Extraction Parameter Tuner")

# Sidebar for parameters
st.sidebar.header("Parameters")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Chart Image", 
    type=['png', 'jpg', 'jpeg', 'bmp', 'tif']
)

# Parameters
st.sidebar.subheader("Grid Removal")
grid_h = st.sidebar.slider("Horizontal Kernel", 10, 100, 40, 5)
grid_v = st.sidebar.slider("Vertical Kernel", 10, 100, 40, 5)

st.sidebar.subheader("Dashed Line Connection")
conn_kernel = st.sidebar.slider("Connection Kernel Size", 1, 15, 5, 2)
conn_iter = st.sidebar.slider("Connection Iterations", 1, 5, 2)

st.sidebar.subheader("Noise Removal")
min_area = st.sidebar.slider("Min Area Threshold", 50, 2000, 500, 50)

st.sidebar.subheader("Display")
show_debug = st.sidebar.checkbox("Show Debug Images", value=True)


def process_image(image: np.ndarray, params: ExtractionParams):
    """Process image and return results."""
    curves, debug_imgs = full_extraction_pipeline(image, params, debug=True)
    vis = visualize_curves(image, curves)
    return curves, debug_imgs, vis


if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Create parameters
    params = ExtractionParams(
        grid_h_kernel=grid_h,
        grid_v_kernel=grid_v,
        connection_kernel_size=conn_kernel,
        connection_iterations=conn_iter,
        min_area_threshold=min_area
    )
    
    # Process
    with st.spinner("Processing..."):
        curves, debug_imgs, vis = process_image(image, params)
    
    # Results
    st.success(f"✅ Found {len(curves)} curves")
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("Extracted Curves")
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Debug images
    if show_debug:
        st.subheader("Debug Images")
        
        debug_cols = st.columns(len(debug_imgs))
        for i, (name, img) in enumerate(debug_imgs.items()):
            with debug_cols[i]:
                st.caption(name)
                if len(img.shape) == 2:
                    st.image(img, use_column_width=True)
                else:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Download buttons
    st.subheader("Download Results")
    
    # Prepare download
    _, buffer = cv2.imencode('.png', vis)
    st.download_button(
        "📥 Download Visualization",
        data=buffer.tobytes(),
        file_name="extracted_curves.png",
        mime="image/png"
    )
    
    # Download curves as CSV
    if curves:
        csv_data = ""
        for i, curve in enumerate(curves):
            csv_data += f"# Curve {i}\n"
            for point in curve:
                csv_data += f"{point[0]},{point[1]}\n"
            csv_data += "\n"
        
        st.download_button(
            "📥 Download Curve Coordinates (CSV)",
            data=csv_data,
            file_name="curves.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Upload a chart image to start")
    
    # Show example
    st.subheader("Example Usage")
    st.markdown("""
    1. Upload your engineering chart image
    2. Adjust parameters using the sliders:
       - **Grid Removal**: Larger values = remove longer grid lines
       - **Connection Kernel**: Larger = connect bigger gaps in dashed lines
       - **Min Area**: Larger = remove more small components (text, noise)
    3. Download the extracted curves
    """)
