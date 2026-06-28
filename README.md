# Street Food Quality Inspector
### AI-Powered Bangladeshi Street Food Quality Detection System

---

| Field | Details |
|---|---|
| **Project Title** | Street Food Quality Inspector |
| **Version** | 1.0 |
| **Date** | June 2026 |
| **Author** | Iffath Sanjida (232-115-222), Wasima Sultana (232-115-230) |
| **Institution** | Metropolitan University, Sylhet |
| **Department** | Computer Science & Engineering (59th Batch, Section F) |
| **Project Type** | Final Year AI/ML Project |
| **Status** | In Development |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [System Architecture](#5-system-architecture)
6. [Use Cases](#6-use-cases)
7. [Model Performance Summary](#7-model-performance-summary)
8. [Known Limitations](#8-known-limitations)
9. [Future Work](#9-future-work)
10. [Appendix: Supported Food Categories](#appendix-supported-food-categories)

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) describes the functional and non-functional requirements for the **Street Food Quality Inspector** — an AI-powered computer vision system designed to assess the quality of Bangladeshi street food items. The system uses deep learning (EfficientNet B0 with Transfer Learning) to classify street food images into three quality categories: **Fresh**, **Acceptable**, and **Poor**.

### 1.2 Scope
The Street Food Quality Inspector is a web-based AI application that:
- Accepts images of Bangladeshi street food as input via a Gradio web interface
- Classifies food into one of **36 categories** (12 food types × 3 quality levels)
- Provides **Grad-CAM heatmap** visualizations explaining the AI's decision
- Displays confidence scores for each quality prediction
- Is accessible via **Hugging Face Spaces** (public URL, no installation required)

The system covers **12 Bangladeshi street food types**: Beguni, Chanachur, Chotpoti, Fuska, Jalmuri, Kulfi, Murobba, Papar, Piaju, Puri, Singhara, and Tiler Khaja.

### 1.3 Definitions and Acronyms

| Term | Definition |
|---|---|
| **AI** | Artificial Intelligence |
| **CNN** | Convolutional Neural Network — deep learning model for image processing |
| **EfficientNet B0** | Pretrained CNN model by Google, used via Transfer Learning |
| **Transfer Learning** | Using a pretrained model and fine-tuning it for a specific task |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — visualizes AI decision regions |
| **Gradio** | Python library for building ML web interfaces |
| **Hugging Face** | Platform for hosting and sharing AI models and apps |
| **Fresh** | Food is newly prepared, visually appealing, and safe to eat |
| **Acceptable** | Food is edible but shows minor signs of aging or imperfection |
| **Poor** | Food is overcooked, stale, burnt, or unsafe to eat |

### 1.4 References
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV.
- PyTorch Documentation: https://pytorch.org/docs
- Hugging Face Spaces: https://huggingface.co/spaces
- Roboflow Bangladeshi Street Food Dataset: https://universe.roboflow.com/bengali-street-food-dataset

---

## 2. Overall Description

### 2.1 Product Perspective
The Street Food Quality Inspector is a **standalone AI application** with no dependency on external food safety databases or inspection services. It operates as a self-contained system using a pre-trained deep learning model deployed on the cloud via Hugging Face Spaces. Users interact through a Gradio web interface accessible from any modern web browser on desktop or mobile.

### 2.2 Product Functions
| Function | Description |
|---|---|
| Image Input | Accept JPEG/PNG/WEBP images uploaded by the user |
| Food Type Recognition | Identify which of the 12 street food types is in the image |
| Quality Classification | Classify food into Fresh, Acceptable, or Poor |
| Confidence Display | Show prediction confidence scores as percentages |
| Grad-CAM Visualization | Generate heatmap showing which regions influenced the decision |
| Text Explanation | Provide natural language explanation of the quality score |

### 2.3 User Classes

| User Class | Description | Technical Level |
|---|---|---|
| General Public | Street food buyers assessing quality before purchasing | Non-technical |
| Street Food Vendors | Vendors self-assessing food presentation and quality | Non-technical |
| Public Health Inspectors | Officials doing informal food safety spot checks | Moderate |
| Researchers / Developers | AI/ML researchers studying model outputs | Technical |

### 2.4 Operating Environment
- **Frontend:** Any modern web browser (Chrome, Firefox, Edge, Safari)
- **Backend:** Python 3.10+, PyTorch 2.0+, torchvision, Gradio 4.0+
- **Model:** EfficientNet B0 pretrained on ImageNet, fine-tuned on custom dataset
- **Deployment:** Hugging Face Spaces (free tier)
- **Training:** Google Colab with T4 GPU

### 2.5 Assumptions and Dependencies
- Users have access to a device with a camera or image files of street food
- Internet connection is required to access the Hugging Face Spaces deployment
- The system assumes images contain Bangladeshi street food from the 12 supported categories
- Images should be reasonably clear and well-lit for optimal accuracy
- The system does **not** perform microbiological analysis — quality is assessed **visually only**

---

## 3. Functional Requirements

### 3.1 Image Input Module

| ID | Requirement |
|---|---|
| FR-01 | The system shall accept image uploads in JPEG, PNG, and WEBP formats with a maximum file size of 10MB |
| FR-02 | The system shall display a preview of the uploaded image before processing |
| FR-03 | The system shall validate uploaded files and display an error message if the format is unsupported or the file is corrupted |

### 3.2 Classification Module

| ID | Requirement |
|---|---|
| FR-04 | The system shall identify which of the 12 supported food types appears in the input image |
| FR-05 | The system shall classify the detected food into one of three quality levels: Fresh, Acceptable, or Poor |
| FR-06 | The system shall display confidence percentage scores for all three quality levels |
| FR-07 | The system shall return a classification result within **5 seconds** of image submission |

### 3.3 Explainability Module

| ID | Requirement |
|---|---|
| FR-08 | The system shall generate and display a Grad-CAM heatmap overlaid on the original image |
| FR-09 | The heatmap shall use a red-to-blue color scale (red = most influential, blue = least influential) |
| FR-10 | The system shall display a brief natural language explanation of the quality rating |

### 3.4 Output Display Module

| ID | Requirement |
|---|---|
| FR-11 | The system shall present results in a dashboard: food type, quality label, confidence chart, heatmap, and text explanation |
| FR-12 | The system shall display a color-coded quality badge: 🟢 Green (Fresh), 🟡 Yellow (Acceptable), 🔴 Red (Poor) |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Requirement | Target |
|---|---|
| Model Test Accuracy | ≥ 75% (achieved: **77.17%**) |
| Inference Time | < 5 seconds per image on Hugging Face Spaces |
| Max Image Size | 10MB |
| Concurrent Users | Up to 10 (free Hugging Face tier) |

### 4.2 Reliability
- The system shall be available 99% of the time during normal Hugging Face Spaces operation
- The system shall handle invalid inputs gracefully without crashing
- The model shall be stored permanently on Hugging Face Hub

### 4.3 Usability
- The interface shall be operable by a non-technical user with no instructions
- The interface shall be responsive on mobile devices (minimum screen width: 320px)
- All labels shall be in simple, jargon-free English

### 4.4 Security
- Uploaded images shall **not** be stored permanently on the server
- No user authentication or personal data collection is required
- User-uploaded images shall not be retained after the session ends

### 4.5 Maintainability
- All source code shall be maintained in a public GitHub repository with a clear README
- The model training notebook shall be documented with comments for reproducibility
- The system shall support model version updates without changing the UI

---

## 5. System Architecture

### 5.1 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Collection | Roboflow Universe, Manual Photography | Source dataset (12 food types) |
| Data Processing | Python, PIL, NumPy | Image conversion, augmentation, splitting |
| Training Framework | PyTorch 2.0, torchvision | Deep learning model training |
| Base Model | EfficientNet B0 (ImageNet pretrained) | Transfer learning backbone |
| Explainability | pytorch-grad-cam | Grad-CAM heatmap generation |
| Web Interface | Gradio 4.0 | User-facing web application |
| Deployment | Hugging Face Spaces | Cloud hosting |
| Version Control | GitHub | Code and documentation |
| Training Compute | Google Colab (T4 GPU) | Model training environment |

### 5.2 Model Architecture

```
Input Image (224×224×3)
        ↓
EfficientNet B0 Backbone (pretrained on ImageNet)
   Phase 1: Backbone frozen → only classifier trained (20 epochs, lr=0.0001)
   Phase 2: Last 3 blocks unfrozen → fine-tuning (15 epochs, lr=0.00001)
        ↓
Final Classifier Layer (1280 → 36 classes)
        ↓
Softmax → Confidence Scores (Fresh / Acceptable / Poor per food type)
        ↓
Grad-CAM → Heatmap Visualization
```

### 5.3 Dataset Summary

| Property | Value |
|---|---|
| Food Types | 12 Bangladeshi street food categories |
| Quality Levels | 3 (Fresh, Acceptable, Poor) |
| Total Classes | 36 (12 × 3) |
| Original Images | ~2,065 |
| After Augmentation | ~5,400 (min. 150 per class) |
| Train Split | 70% (~3,780 images) |
| Validation Split | 15% (~792 images) |
| Test Split | 15% (~828 images) |
| Augmentation Techniques | Horizontal flip, rotation (±25°), brightness, contrast, color jitter, Gaussian blur, random crop |
| Class Imbalance Handling | Weighted CrossEntropyLoss |

---

## 6. Use Cases

### UC-01: Assess Street Food Quality

| Field | Description |
|---|---|
| **Actor** | General Public / Street Food Buyer |
| **Precondition** | User has the app URL and an image of street food |
| **Main Flow** | 1. User opens app URL → 2. Uploads food photo → 3. Clicks 'Analyze' → 4. System processes image → 5. System displays: food type, quality badge, confidence scores, Grad-CAM heatmap, text explanation |
| **Postcondition** | User receives quality assessment with visual explanation |
| **Alternative Flow** | If image format is invalid → system shows error and prompts re-upload |
| **Exception** | If food is not recognized → system displays "Unable to identify food item" |

### UC-02: Vendor Self-Assessment

| Field | Description |
|---|---|
| **Actor** | Street Food Vendor |
| **Precondition** | Vendor has a smartphone and prepared food |
| **Main Flow** | 1. Vendor opens app on mobile → 2. Takes photo of food → 3. Uploads and analyzes → 4. Reviews quality feedback → 5. Adjusts food preparation accordingly |
| **Postcondition** | Vendor has actionable feedback on food quality |
| **Alternative Flow** | If 'Poor' detected → vendor is advised to prepare fresh food |

---

## 7. Model Performance Summary

Results on held-out test set (828 images, 36 classes):

| Metric | Value |
|---|---|
| **Test Accuracy** | **77.17%** |
| Macro Precision | 78% |
| Macro Recall | 77% |
| Macro F1-Score | 76% |
| Best Performing Class | `papar_fresh` (F1 = 1.00) |
| Weakest Performing Class | `jalmuri_acceptable` (F1 = 0.40) |
| Training Phase 1 | 20 epochs (frozen backbone) |
| Training Phase 2 | 15 epochs (fine-tuning) |
| Base Model | EfficientNet B0 (ImageNet pretrained) |

> **Note:** The `acceptable` quality class consistently shows lower F1 scores. This is expected — "acceptable" is a subjective middle category that is visually ambiguous for both human annotators and the model.

---

## 8. Known Limitations

- The system is limited to **12 Bangladeshi street food categories** only
- Visual quality assessment **cannot detect microbiological contamination** or chemical hazards
- Performance may degrade for blurred, poorly lit, or partially obscured food images
- The `acceptable` quality category shows lower accuracy due to its subjective nature
- Limited dataset size per class (~57 original images average) restricts generalization
- **Not a replacement** for certified food safety inspection

---

## 9. Future Work

- Expand to more Bangladeshi street food categories (target: 36 types)
- Integrate real-time camera feed for live quality assessment
- Add **Bengali language** support for text explanations
- Develop a **mobile app** (Android/iOS) with offline inference using TFLite
- Incorporate hygiene factor detection (contamination, packaging damage)
- Implement user feedback loop for continuous model improvement
- Explore **YOLOv8** integration for multi-food detection in a single image

---

## Appendix: Supported Food Categories

| # | Food Type | বাংলা নাম | Quality Classes |
|---|---|---|---|
| 1 | Beguni | বেগুনি | Fresh, Acceptable, Poor |
| 2 | Chanachur | চানাচুর | Fresh, Acceptable, Poor |
| 3 | Chotpoti | চটপটি | Fresh, Acceptable, Poor |
| 4 | Fuska | ফুসকা | Fresh, Acceptable, Poor |
| 5 | Jalmuri | ঝালমুড়ি | Fresh, Acceptable, Poor |
| 6 | Kulfi | কুলফি | Fresh, Acceptable, Poor |
| 7 | Murobba | মুরব্বা | Fresh, Acceptable, Poor |
| 8 | Papar | পাপড় | Fresh, Acceptable, Poor |
| 9 | Piaju | পিয়াজু | Fresh, Acceptable, Poor |
| 10 | Puri | পুরি | Fresh, Acceptable, Poor |
| 11 | Singhara | শিঙাড়া | Fresh, Acceptable, Poor |
| 12 | Tiler Khaja | তিলের খাজা | Fresh, Acceptable, Poor |

---

*Street Food Quality Inspector SRS v1.0 | Metropolitan University, Sylhet | CSE 59th Batch*
