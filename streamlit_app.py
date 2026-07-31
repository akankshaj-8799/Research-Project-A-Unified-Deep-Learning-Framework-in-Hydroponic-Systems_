from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from crop_cnn import CropGroupCNN, MODEL_PATH, image_to_tensor


def confidence_text(value):
    return f"{value * 100:.0f}% confidence"


@st.cache_resource
def load_crop_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run: python train_light_model.py --epochs 6")
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = CropGroupCNN(len(checkpoint["class_labels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


def predict(image, model):
    tensor = image_to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)

    return torch.softmax(logits, dim=1)[0]


def image_cues(image):
    arr = np.array(image.convert("RGB").resize((128, 128)), dtype=np.float32)
    brightness = arr.mean()
    green = arr[:, :, 1].mean()
    red = arr[:, :, 0].mean()
    blue = arr[:, :, 2].mean()
    yellow_strength = max(0.0, min(red, green) - blue)
    contrast = arr.std()
    return {
        "brightness": brightness,
        "green": green,
        "red": red,
        "blue": blue,
        "yellow_strength": yellow_strength,
        "contrast": contrast,
    }


def explain_prediction(crop, stage, health, crop_conf, stage_conf, health_conf, cues):
    reasons = [
        f"The model compared the whole image against the crop patterns it learned during training and matched it most strongly with {crop}.",
        f"The growth stage is {stage} because the flower/leaf shape, color distribution, and texture patterns were closest to that stage class.",
    ]
    if health.lower() == "unhealthy":
        reasons.append(
            "It is marked unhealthy because the learned visual pattern is closer to wilted or unhealthy training images than to healthy examples."
        )
    else:
        reasons.append(
            "It is marked healthy because the learned visual pattern is closer to healthy training images than to wilted or unhealthy examples."
        )

    reasons.append(
        f"Confidence: crop {crop_conf * 100:.0f}%, stage {stage_conf * 100:.0f}%, health {health_conf * 100:.0f}%."
    )

    cue_text = (
        f"Image cues: brightness {cues['brightness']:.1f}, contrast {cues['contrast']:.1f}, "
        f"red/green/blue averages {cues['red']:.1f}/{cues['green']:.1f}/{cues['blue']:.1f}, "
        f"yellow-green strength {cues['yellow_strength']:.1f}."
    )
    return reasons, cue_text


st.set_page_config(page_title="Crop Prediction", layout="centered")
st.title("Crop Type, Stage, and Health Prediction")

st.markdown(
    """
    <style>
    .uploaded-preview img {
        max-width: 320px;
        border-radius: 8px;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

checkpoint, model = load_crop_model()
class_labels = checkpoint["class_labels"]

image_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if image_file:
    original_image = Image.open(image_file).convert("RGB")
    class_probs = predict(original_image, model)
    class_idx = int(torch.argmax(class_probs).item())
    crop, stage, health = class_labels[class_idx]
    class_conf = float(class_probs[class_idx].item())
    crop_conf = float(sum(prob for prob, label in zip(class_probs.tolist(), class_labels) if label[0] == crop))
    stage_conf = float(sum(prob for prob, label in zip(class_probs.tolist(), class_labels) if label[1] == stage))
    health_conf = float(sum(prob for prob, label in zip(class_probs.tolist(), class_labels) if label[2] == health))

    cues = image_cues(original_image)
    reasons, cue_text = explain_prediction(crop, stage, health, crop_conf, stage_conf, health_conf, cues)

    left, right = st.columns([0.9, 1.1], vertical_alignment="top")
    with left:
        st.markdown('<div class="uploaded-preview">', unsafe_allow_html=True)
        st.image(original_image, caption="Uploaded image", width=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.metric("Crop type", crop, confidence_text(crop_conf))
        st.metric("Stage", stage, confidence_text(stage_conf))
        st.metric("Health", health, confidence_text(health_conf))
        st.caption(f"Exact class confidence: {class_conf * 100:.0f}%")
        if class_conf < 0.65 or health_conf < 0.65:
            st.warning("This prediction is close between classes, so review the image manually before relying on it.")

        st.subheader("Why this result")
        for reason in reasons:
            st.write(reason)
        st.caption(cue_text)
else:
    st.info("Upload a crop image to get a prediction.")
