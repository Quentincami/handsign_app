import streamlit as st
import cv2
import av  # The library for video frames
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer  # Only import webrtc_streamer

# --- 1. Load Your Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the YOLO model from the specified path."""
    model = YOLO('weights/best.pt')
    return model

model = load_model()

# --- 2. Define the Frame Processing Function ---
# This is the new, simpler callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This method is called for every frame from the webcam.
    """
    # Convert the av.VideoFrame to a NumPy array (OpenCV format)
    img = frame.to_ndarray(format="bgr24")

    # Run the YOLO model on the frame
    results = model(img, stream=True)

    # Process results and draw boxes
    for r in results:
        img = r.plot()  # r.plot() returns a NumPy array with boxes drawn

    # Convert the annotated NumPy array back to an av.VideoFrame
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 3. Set Up the Streamlit Interface ---
st.set_page_config(page_title="Hand Sign Detection", layout="wide")
st.title("Hand Sign Detection")
st.write("This app uses a YOLOv8 model to detect hand signs in real-time.")
st.write("This model has been trained on 10 different hand signs : 🤚👌🤞🖖☝️🤟👍✌️🤙🤏 ")
st.write("Click 'Start' below to enable your webcam and see the detection.")

# --- 4. Add the WebRTC Streamer ---
webrtc_streamer(
    key="handsign-detection",
    video_frame_callback=video_frame_callback,  # <-- Pass the FUNCTION here
    rtc_configuration={  # This is needed for deployment on Streamlit Cloud
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

st.sidebar.header("About")
st.sidebar.info("This is a demo of a custom-trained YOLOv8 model for hand sign recognition, "
                "deployed as a Streamlit web app.")