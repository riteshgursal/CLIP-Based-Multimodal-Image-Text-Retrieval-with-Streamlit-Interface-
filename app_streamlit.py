import torch
import clip
from PIL import Image
import os
import streamlit as st

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load all images
image_folder = "sample_data"
images = []
image_names = []

for img_file in os.listdir(image_folder):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = preprocess(Image.open(os.path.join(image_folder, img_file))).unsqueeze(0).to(device)
        images.append(image)
        image_names.append(img_file)

# Encode all image features once
image_features = torch.cat([model.encode_image(img) for img in images])
image_features /= image_features.norm(dim=-1, keepdim=True)

# Streamlit UI
st.title("🖼️ CLIP-Based Multimodal Image–Text Retrieval")
st.write("Enter a text query to find the most semantically similar image using OpenAI’s CLIP model.")

text_query = st.text_input("Enter your search text:", "")

if text_query:
    text_tokens = clip.tokenize([text_query]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).squeeze(1)
    best_idx = similarity.argmax().item()
    best_match = image_names[best_idx]
    best_score = similarity[best_idx].item()

    st.image(os.path.join(image_folder, best_match),
         caption=f"Best match: {best_match} (Score: {best_score:.2f})",
         use_container_width=True)

    # Optional: show all ranked results
    st.subheader("All Image Similarities:")
    for idx, name in enumerate(image_names):
        st.write(f"{name}: {similarity[idx]:.2f}")
