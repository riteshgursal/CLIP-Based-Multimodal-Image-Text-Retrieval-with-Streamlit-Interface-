import torch
import clip
from PIL import Image
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load images
image_folder = "sample_data"
images = []
image_names = []

for img_file in os.listdir(image_folder):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        image = preprocess(Image.open(os.path.join(image_folder, img_file))).unsqueeze(0).to(device)
        images.append(image)
        image_names.append(img_file)

image_features = torch.cat([model.encode_image(img) for img in images])
image_features /= image_features.norm(dim=-1, keepdim=True)

# Run text query loop
while True:
    text_query = input("\nEnter a text query (or 'exit' to stop): ")
    if text_query.lower() == 'exit':
        break

    text_tokens = clip.tokenize([text_query]).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (100.0 * image_features @ text_features.T).squeeze(1)
    best_match_idx = similarity.argmax().item()
    print(f"🔍 Best match: {image_names[best_match_idx]} (Score: {similarity[best_match_idx]:.2f})")
