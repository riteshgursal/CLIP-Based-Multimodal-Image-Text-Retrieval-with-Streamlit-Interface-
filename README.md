# 🖼️ CLIP-Based Multimodal Image–Text Retrieval (with Streamlit Interface)

A multimodal AI system that connects **text and image understanding** using **OpenAI’s CLIP (Contrastive Language–Image Pretraining)** model.  
This project retrieves the most semantically relevant image for a given text query, demonstrating cross-modal representation learning — a key concept in **Multimodal AI**, **Computer Vision**, and **Information Retrieval**.

---

## 🎯 Project Overview
Multimodal retrieval is the process of integrating and understanding multiple data modalities such as text and images.  
Using OpenAI’s CLIP model (`ViT-B/32`), this project embeds both text and images into a shared vector space and measures cosine similarity to find the best match.  

The repository includes:
- 🧠 **Command-line version (`app.py`)** – for fast testing and demonstration.  
- 🌐 **Streamlit Web App (`app_streamlit.py`)** – for an interactive user interface.

---

## 🚀 Key Features
- Text-to-image semantic search using OpenAI CLIP (ViT-B/32)
- Interactive **Streamlit** interface with real-time image retrieval
-  lightweight, Runs seamlessly on both CPU and GPU
- Demonstrates real-world **multimodal AI** capabilities used in:
  - Smart Manufacturing
  - Digital Twins
  - Autonomous Driving
  - Generative AI Systems
    

---


**🧠 Learning Outcomes**

Understanding of CLIP and multimodal embeddings

Application of cosine similarity for semantic matching

Experience with AI deployment using Streamlit

Integration of machine learning into real-world retrieval systems

---


## 🧩 Tech Stack
- **Python**
- **PyTorch**
- **OpenAI CLIP (ViT-B/32)**
- **Streamlit**
- **Pillow (PIL)**
- **NumPy / TQDM / Regex**

---

## 🗂️ Folder Structure
```
CLIP-Multimodal-Retrieval/
│
├── app.py # CLI version
├── app_streamlit.py # Streamlit Web Interface
├── requirements.txt
├── sample_data/ # Folder containing sample images
│ ├── cat.jpg
│ ├── car.jpg
│ ├── laptop.jpg
│ └── beach.jpg
├── results/ # Folder for screenshots / outputs
└── README.md
```

---

## ⚙️ Installation and Execution

### 1️⃣ Create Environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run CLI Version
python app.py


Enter queries such as:

a cat sitting on a floor
a car driving on the road


The system will output the best matching image.

4️⃣ Run Streamlit Web Interface
streamlit run app_streamlit.py


The app will open automatically at http://localhost:8501


| Text Query             | Retrieved Image | Score |
| ---------------------- | --------------- | ----- |
| "a cute kitten"        | cat.jpg         | 87.12 |
| "a car on the highway" | car.jpg         | 85.47 |
| "a laptop on a desk"   | laptop.jpg      | 83.25 |






