# Adapted from: https://www.gradio.app/guides/object-detection-from-webcam-with-webrtc
import gradio as gr
from fastrtc import Stream
from huggingface_hub import hf_hub_download

# from inference.models import YOLOv10ObjectDetection
from ultralytics import ASSETS, YOLO
import PIL.Image as Image

css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""


model = YOLO("yolo11n.pt")


def detection(image, conf_threshold=0.3):
    # import pdb; pdb.set_trace()
    results = model.predict(image, conf=conf_threshold, show_labels=True, show_conf=True, device="0")
    im = None
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
    return im_array


stream = Stream(
    detection,
    modality='video',
    mode='send-receive',
    additional_inputs=[
        gr.Slider(
            label="Confidence Threshold",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=0.30,
        )
    ],
    concurrency_limit=10
)


if __name__ == "__main__":
    stream.ui.launch(share=True,debug=True)
    
