import streamlit as st
from PIL import Image
import os
from app_utils.prediction_pipeline import run_pipeline
import plotly.graph_objects as go
import numpy as np
import base64

def set_bg_from_local(image_file_path):
    # Open the image file in binary mode
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Get correct MIME type
    file_ext = image_file_path.split('.')[-1].lower()
    mime_type = f"image/{'png' if file_ext in ['png', 'jpeg'] else file_ext}"

    # Inject background CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{mime_type};base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            backdrop-filter: blur(3px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_from_local("static/background.jpg")
st.title("🌀 Cyclone Early Alert System")
st.write("Upload a satellite image to detect cyclone presence, visualize detection, predict windspeed, and estimate intensity.")

# Plot animated IMD scale
import plotly.graph_objects as go

def plot_static_intensity_scale(windspeed_knots):
    # Define IMD categories
    categories = [
        ("D", 17, 27),
        ("DD", 28, 33),
        ("CS", 34, 47),
        ("SCS", 48, 63),
        ("VSCS", 64, 89),
        ("ESCS", 90, 119),
        ("SuCS", 120, 150)
    ]
    colors = ['#b0c4de', '#87ceeb', '#00bfff', '#1e90ff', '#ff8c00', '#ff4500', '#8b0000']

    shapes, annotations = [], []
    x_max = categories[-1][2] + 10

    # Draw the colored bands and labels
    for i, (label, start, end) in enumerate(categories):
        shapes.append(dict(
            type='rect',
            x0=start, x1=end, y0=0.5, y1=1,
            fillcolor=colors[i], line=dict(width=0)
        ))
        annotations.append(dict(
            x=(start + end) / 2,
            y=0.75,
            text=label,
            showarrow=False,
            font=dict(size=14, color='white', family="Arial Black")
        ))

    # Determine where the arrow should point
    arrow_annotation = dict(
        x=windspeed_knots,
        y=1.05,
        ax=windspeed_knots,
        ay=1.3,
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="black"
    )

    # Windspeed text
    windspeed_label = dict(
        x=windspeed_knots,
        y=1.35,
        text=f"{windspeed_knots:.1f} knots",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    # Final figure
    fig = go.Figure(
        layout=go.Layout(
            shapes=shapes,
            annotations=annotations + [arrow_annotation, windspeed_label],
            xaxis=dict(range=[15, x_max], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(range=[0, 2], visible=False),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white'
        )
    )

    return fig



# Upload UI
uploaded_file = st.file_uploader("Upload a Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save image temporarily
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Run the prediction pipeline
    with st.spinner("Analyzing image..."):
        results = run_pipeline(image_path)

    # Show results
    if results["cyclone_present"]:
        st.success("🌪️ Cyclone Detected!")
        # Show annotated image
        if results.get("annotated_image") and os.path.exists(results["annotated_image"]):
            st.image(results["annotated_image"], caption="Detected Cyclone (YOLO)", use_column_width=True)
        st.markdown(f"**Predicted Windspeed:** `{results['windspeed']} knots`")
        st.markdown(f"**Estimated Intensity Category:** `{results['intensity']}`")

        # Plot intensity scale
        st.markdown("### 📊 Intensity Classification Scale")
        fig = plot_static_intensity_scale(results['windspeed'])
        st.plotly_chart(fig, use_container_width=True)
        

        

        
    else:
        st.info("✅ No cyclone detected in the image.")

    # Clean up uploaded image
    os.remove(image_path)
