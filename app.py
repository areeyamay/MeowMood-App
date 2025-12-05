import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64
from io import BytesIO

st.set_page_config(
    page_title="MeowMood - Cat Emotion Detector",
    page_icon="😺",
    layout="wide"
)

# ---------- load CSS ----------
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")


# ---------- utils ----------
def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()


# ---------- paw icon (images/pawprint.png) ----------
PAW_ICON_PATH = Path("images/pawprint.png")
paw_icon_html = "🐾"
paw_b64 = None

if PAW_ICON_PATH.exists():
    try:
        paw_img = Image.open(PAW_ICON_PATH)
        paw_b64 = pil_to_base64(paw_img)
        paw_icon_html = f'<img src="data:image/png;base64,{paw_b64}" class="paw-icon"/>'
    except Exception:
        paw_icon_html = "🐾"

# ---------- upload cat icon (images/icon_cat.png) ----------
ICON_CAT_PATH = Path("images/icon_cat.png")
upload_icon_html = "🐱"

if ICON_CAT_PATH.exists():
    try:
        icon_img = Image.open(ICON_CAT_PATH)
        icon_b64 = pil_to_base64(icon_img)
        upload_icon_html = f'<img src="data:image/png;base64,{icon_b64}" class="upload-icon"/>'
    except Exception:
        upload_icon_html = "🐱"

# ---------- BG ลายเท้าแมว ----------
if paw_b64 is not None:
    paw_bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: #FFF7EC;
        position: relative;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;

        background-image:
          url("data:image/png;base64,{paw_b64}"),
          url("data:image/png;base64,{paw_b64}"),
          url("data:image/png;base64,{paw_b64}"),
          url("data:image/png;base64,{paw_b64}"),
          url("data:image/png;base64,{paw_b64}"),
          url("data:image/png;base64,{paw_b64}");

        background-position:
          5% 18%,
          70% 10%,
          96% 35%,
          6% 62%,
          44% 92%,
          95% 82%;

        background-repeat: no-repeat;
        background-size: 32px 32px;
        opacity: 0.22;
    }}
    </style>
    """
    st.markdown(paw_bg_css, unsafe_allow_html=True)


# ---------- hero cats (images/hero_cats.png) ----------
HERO_CATS_PATH = Path("images/hero_cats.png")
hero_cats_html = ""
if HERO_CATS_PATH.exists():
    try:
        hero_img = Image.open(HERO_CATS_PATH)
        hero_b64 = pil_to_base64(hero_img)
        hero_cats_html = f'<img src="data:image/png;base64,{hero_b64}" class="hero-cats"/>'
    except Exception:
        hero_cats_html = ""


# ---------- load model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("meowmood_model.h5")

model = load_model()

CLASS_NAMES = ["angry", "neutral", "relaxed", "scared"]

EXPLANATIONS = {
    "angry": "Your cat may be annoyed or irritated – tail flicking, ears back, not in the mood to cuddle.",
    "neutral": "Your cat looks calm and normal – just observing the world, not too excited, not too scared.",
    "relaxed": "Your cat seems comfy and relaxed – lying down, eyes half-closed, body loose and soft.",
    "scared": "Your cat might be feeling scared or anxious – wide eyes, tense body, ears down, ready to hide.",
}


# ---------- preprocess / predict ----------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_mood(img: Image.Image):
    x = preprocess_image(img)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])
    scores = {name: float(p) for name, p in zip(CLASS_NAMES, preds)}
    return label, confidence, scores


def load_pil_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)
    return img


# ---------- HERO ----------
st.markdown(
    f"""
<div class="hero-card">

  <div class="hero-badge">
    {paw_icon_html}
    Smart Cat Care · MeowMood
  </div>

  <div class="hero-title">
    Caring for your cat's<br>mood like our own.
  </div>

  <div class="hero-sub">
    Upload a photo of your cat and let MeowMood guess <br>whether
    your cat is...
    <div class="hero-moods">
      <span class="emo-tag emo-angry">ANGRY</span>
      <span class="emo-tag emo-neutral">NEUTRAL</span>
      <span class="emo-tag emo-relaxed">RELAXED</span>
      <span class="emo-tag emo-scared">SCARED</span>
    </div>
  </div>

  <div class="hero-cats-wrapper">
    {hero_cats_html}
  </div>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# ---------- layout ----------
col_left, col_right = st.columns([1, 1])

uploaded_file = None
image = None
label = None
confidence = None
scores = None

with col_left:
    uploaded_file = st.file_uploader(
        "Upload your cat photo (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = load_pil_image(uploaded_file)
        img_b64 = pil_to_base64(image)

        st.markdown(
            f"""
<div class="image-card">
  <img src="data:image/png;base64,{img_b64}" class="image-fit"/>
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption("Tip: A clear view of your cat's face and eyes helps the model.")
    else:
        st.markdown(
            f"""
<div class="upload-hint">
  <span>Drop a cat photo here to start</span>
  {upload_icon_html}
</div>
""",
            unsafe_allow_html=True,
        )

    analyze = st.button("Analyze mood")

# run prediction
if analyze and uploaded_file is not None and image is not None:
    with st.spinner("Thinking about your cat's mood..."):
        label, confidence, scores = predict_mood(image)

with col_right:
    if label is None:
        st.markdown(
            """
<div class="emotion-box">
  <div class="emotion-title">Mood result</div>
  <p>Upload a photo on the left and click <b>Analyze mood</b>.</p>
  <p>We will show the predicted emotion and explanation here.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        emotion_class = f"emotion-label emotion-label-{label}"

        st.markdown(
            f"""
<div class="emotion-box">
  <div class="emotion-title">Predicted mood</div>
  <div class="{emotion_class}">✨ {label.upper()} ✨</div>
  <p>Confidence: <b>{confidence * 100:.2f}%</b></p>
  <p><b>What it means:</b> {EXPLANATIONS[label]}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="prob-box"><div class="emotion-title">Emotion probabilities</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="prob-section">', unsafe_allow_html=True)

        for emo in CLASS_NAMES:
            p = scores[emo]
            st.markdown(
                f"""
<div class="prob-item">
  <p class="prob-label"><b>{emo.upper()}</b>: {p * 100:.2f}%</p>
  <div class="emo-bar-wrapper">
    <div class="emo-bar emo-bar-{emo}" style="width:{p * 100:.2f}%"></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
