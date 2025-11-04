#!/usr/bin/env python3
"""
Simple Gradio UI for image classification using DeiT and TinyVIT models.
Uses checkpoints from Dataset_11880_64c_186f.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from typing import Dict, List
import gradio as gr
from torchvision import transforms
import timm

# Constants
ROOT = "/home/saksham/coding/dl"
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
DATASET_TAG = "Dataset_11880_64c_186f"
CSV_PATH = os.path.join(ROOT, "Dataset.csv")
IMG_SIZE = 224
PROJ_DIM = 384

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Unified Attribute Model
class UnifiedAttributeModel(nn.Module):
    def __init__(self, backbone_name: str, facet_to_id: Dict[str, Dict[str, int]], proj_dim: int = 384):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Create backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        embed_dim = getattr(self.backbone, 'num_features', 192)
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # Attribute heads
        self.attr_heads = nn.ModuleDict({})
        for facet, mapping in facet_to_id.items():
            self.attr_heads[facet] = nn.Linear(embed_dim, len(mapping))

        # Image projection for retrieval
        self.img_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        logits_attrs = {facet: head(feats) for facet, head in self.attr_heads.items()}
        img_emb = self.img_proj(feats)
        img_emb = F.normalize(img_emb, dim=-1)
        return feats, logits_attrs, img_emb


def parse_attributes_dynamic(attr_str: str) -> Dict[str, str]:
    """Dynamically parse all attributes from string."""
    values = {}
    if isinstance(attr_str, str) and attr_str.strip():
        parts = [p.strip() for p in attr_str.split(";") if p.strip()]
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                k = k.strip().lower()
                v = v.strip().lower()
                values[k] = v if v else "unknown"
    return values


def extract_coarse_class(label: str) -> str:
    """Extract high-level category."""
    if not isinstance(label, str):
        return "unknown"
    label_clean = label.strip().lower().replace("-", "_")
    parts = [p for p in label_clean.split("_") if p]
    if not parts:
        return label_clean or "unknown"
    if len(parts) >= 2 and parts[0] in {"sports", "home", "daily"} and parts[1] in {"item", "items", "needs"}:
        return f"{parts[0]} {parts[1]}".strip()
    return parts[0]


def build_label_mappings():
    """Build label mappings from CSV dataset. Matches training preprocessing."""
    print("Loading dataset and building label mappings...")
    df = pd.read_csv(CSV_PATH)
    
    # Extract coarse and fine classes
    df["target_class"] = df["class_label"].apply(extract_coarse_class)
    df["fine_label"] = df["class_label"].str.strip().str.lower().str.replace("-", "_")
    
    # Filter classes based on minimum sample requirement (matches training)
    MIN_SAMPLES_PER_CLASS = 2
    USE_ALL_CLASSES = True
    
    class_counts = df["target_class"].value_counts()
    eligible_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    
    if USE_ALL_CLASSES:
        selected_classes = eligible_classes
        print(f"✅ Using ALL {len(selected_classes)} eligible classes (>= {MIN_SAMPLES_PER_CLASS} samples each)")
    else:
        selected_classes = eligible_classes[:10]  # Top 10 for backward compatibility
    
    # Filter dataframe to only include selected classes
    df = df[df["target_class"].isin(selected_classes)].reset_index(drop=True)
    print(f"📈 Filtered dataset size: {len(df)} rows")
    
    # Parse attributes
    attr_rows = df["attributes"].apply(parse_attributes_dynamic)
    
    # Discover all attribute keys
    all_facet_keys = set()
    for row in attr_rows:
        all_facet_keys.update(row.keys())
    
    FACETS = sorted(all_facet_keys)
    
    # Normalize attributes
    def normalize_attributes(attr_dict: Dict[str, str]) -> Dict[str, str]:
        return {facet: attr_dict.get(facet, "unknown") for facet in FACETS}
    
    attr_rows = attr_rows.apply(normalize_attributes)
    
    # Build facet vocabularies
    facet_to_values: Dict[str, List[str]] = {}
    for facet in FACETS:
        vals = sorted(set([row[facet] for row in attr_rows]))
        if "unknown" not in vals:
            vals = ["unknown"] + vals
        facet_to_values[facet] = vals
    
    # Class label maps (only from filtered data)
    coarse_classes = sorted(set(df["target_class"]))
    fine_classes = sorted(set(df["fine_label"]))
    
    # Unified facets
    FACETS_UNIFIED = ["coarse_class", "fine_class", "color", "material", "condition", "size"]
    
    facet_to_values_unified: Dict[str, List[str]] = {
        "coarse_class": coarse_classes,
        "fine_class": fine_classes,
        "color": facet_to_values["color"],
        "material": facet_to_values["material"],
        "condition": facet_to_values["condition"],
        "size": facet_to_values["size"]
    }
    
    facet_to_id_unified: Dict[str, Dict[str, int]] = {
        facet: {v: i for i, v in enumerate(values)} 
        for facet, values in facet_to_values_unified.items()
    }
    
    id_to_facet_value_unified: Dict[str, Dict[int, str]] = {
        facet: {i: v for v, i in mapping.items()} 
        for facet, mapping in facet_to_id_unified.items()
    }
    
    print(f"Built mappings: {len(coarse_classes)} coarse classes, {len(fine_classes)} fine classes")
    return facet_to_id_unified, id_to_facet_value_unified


def load_checkpoint(model, model_name: str, dataset_tag: str):
    """Load model checkpoint."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, dataset_tag, f"{model_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded {model_name} (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


# Build label mappings
facet_to_id_unified, id_to_facet_value_unified = build_label_mappings()

# Image transforms
val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load models
print("\nLoading models...")
model_deit = UnifiedAttributeModel(
    backbone_name='deit_tiny_patch16_224',
    facet_to_id=facet_to_id_unified,
    proj_dim=PROJ_DIM
).to(device)
model_deit = load_checkpoint(model_deit, "deit_unified_attr", DATASET_TAG)

model_tinyvit = UnifiedAttributeModel(
    backbone_name='tiny_vit_5m_224.dist_in22k_ft_in1k',
    facet_to_id=facet_to_id_unified,
    proj_dim=PROJ_DIM
).to(device)
model_tinyvit = load_checkpoint(model_tinyvit, "tinyvit_unified_attr", DATASET_TAG)

print("✅ All models loaded successfully!\n")


@torch.no_grad()
def predict_image(image: Image.Image, model_name: str) -> Dict[str, str]:
    """Run inference on an image."""
    if image is None:
        return {"error": "Please upload an image"}
    
    # Select model
    model = model_deit if model_name == "DeiT-Tiny" else model_tinyvit
    
    # Transform image
    img_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        _, logits_attrs, _ = model(img_tensor)
    
    # Get predictions
    predictions = {}
    for facet in ["coarse_class", "fine_class", "color", "material", "condition", "size"]:
        logits = logits_attrs[facet]
        probs = F.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()
        confidence = probs[0, pred_id].item()
        pred_value = id_to_facet_value_unified[facet][pred_id]
        
        # Get top 3 predictions
        top3_probs, top3_ids = torch.topk(probs[0], min(3, len(probs[0])))
        top3_values = [id_to_facet_value_unified[facet][idx.item()] for idx in top3_ids]
        top3_confidences = [prob.item() for prob in top3_probs]
        
        predictions[facet] = {
            "predicted": pred_value,
            "confidence": f"{confidence:.2%}",
            "top3": list(zip(top3_values, [f"{c:.2%}" for c in top3_confidences]))
        }
    
    return predictions


def format_predictions(predictions: Dict[str, Dict]) -> str:
    """Format predictions for display."""
    if "error" in predictions:
        return predictions["error"]
    
    output = []
    output.append("## Classification Results\n")
    
    # Class predictions
    output.append("### 🏷️ Class Predictions")
    output.append(f"**Coarse Class:** {predictions['coarse_class']['predicted']} ({predictions['coarse_class']['confidence']})")
    output.append(f"**Fine Class:** {predictions['fine_class']['predicted']} ({predictions['fine_class']['confidence']})")
    
    # Attributes
    output.append("\n### 🎨 Attributes")
    for attr in ["color", "material", "condition", "size"]:
        pred = predictions[attr]
        output.append(f"**{attr.capitalize()}:** {pred['predicted']} ({pred['confidence']})")
        if len(pred['top3']) > 1:
            alternatives = ", ".join([f"{val} ({conf})" for val, conf in pred['top3'][1:]])
            output.append(f"  *Alternatives: {alternatives}*")
    
    return "\n".join(output)


def classify_image(image: Image.Image, model_choice: str) -> str:
    """Main classification function for Gradio."""
    try:
        predictions = predict_image(image, model_choice)
        return format_predictions(predictions)
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Image Classification - DeiT & TinyVIT") as demo:
    gr.Markdown("# 🖼️ Image Classification Interface")
    gr.Markdown("Upload an image to classify it using DeiT-Tiny or TinyVIT models trained on Dataset_11880.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            model_choice = gr.Radio(
                choices=["DeiT-Tiny", "TinyVIT"],
                value="DeiT-Tiny",
                label="Select Model"
            )
            classify_btn = gr.Button("Classify Image", variant="primary")
        
        with gr.Column():
            output = gr.Markdown(label="Predictions")
    
    classify_btn.click(
        fn=classify_image,
        inputs=[image_input, model_choice],
        outputs=output
    )
    
    gr.Markdown("### 📊 Model Information")
    gr.Markdown(f"- **Dataset:** {DATASET_TAG}")
    gr.Markdown("- **Models:** DeiT-Tiny (deit_tiny_patch16_224) and TinyVIT (tiny_vit_5m_224.dist_in22k_ft_in1k)")
    gr.Markdown("- **Classes:** 64 coarse classes, 186 fine classes")
    gr.Markdown("- **Attributes:** Color, Material, Condition, Size")

if __name__ == "__main__":
    demo.launch(share=False)
