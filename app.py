import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from huggingface_hub import hf_hub_download

# ── IMPORTANT: replace with YOUR actual Hugging Face username ──
MODEL_REPO = "Iffath222/street-food-quality-model"

# ── Model + class names download from HF Hub ──
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="street_food_best_model.pth")
class_names_path = hf_hub_download(repo_id=MODEL_REPO, filename="class_names.txt")

with open(class_names_path) as f:
    class_names = f.read().splitlines()
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_layer = [model.features[-1]]


# ── Core analysis function ──
def analyze_food_quality(image):
    img_resized = image.convert("RGB").resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    input_tensor = val_test_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx].item() * 100

    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    food_type, quality = pred_class.rsplit("_", 1)
    food_type_display = food_type.replace("-", " ").title()

    badge = {"fresh": "🟢", "acceptable": "🟡", "poor": "🔴"}.get(quality, "⚪")

    explanations = {
        "fresh": f"This {food_type_display} appears fresh and well-prepared. The color and texture look normal, and it seems safe to eat.",
        "acceptable": f"This {food_type_display} is edible but shows minor signs of aging or imperfection. Consider it with some caution.",
        "poor": f"This {food_type_display} shows signs of being overcooked, stale, burnt, or of poor quality. Exercise caution before eating."
    }
    explanation = explanations.get(quality, "Quality could not be determined.")

    label_text = f"{badge} {food_type_display} — {quality.title()}"

    top3_idx = probs.argsort(descending=True)[:3]
    confidence_dict = {class_names[i]: float(probs[i]) for i in top3_idx}

    return label_text, confidence_dict, heatmap, explanation


# ── Gradio Interface ──
demo = gr.Interface(
    fn=analyze_food_quality,
    inputs=gr.Image(type="pil", label="Upload Street Food Image"),
    outputs=[
        gr.Textbox(label="Quality Result"),
        gr.Label(label="Confidence Scores"),
        gr.Image(label="Grad-CAM Explanation"),
        gr.Textbox(label="Explanation", lines=3)
    ],
    title="🍢 Street Food Quality Inspector",
    description="Upload a photo of Bangladeshi street food (Beguni, Chotpoti, Fuska, Jalmuri, Kulfi, Murobba, Papar, Piaju, Puri, Singhara, Tiler Khaja, Chanachur) to check its quality.",
)

demo.launch()
