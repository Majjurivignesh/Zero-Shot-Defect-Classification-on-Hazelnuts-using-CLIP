# Zero-Shot-Defect-Classification-on-Hazelnuts-using-CLIP
Akridata-ML-Intern-Sep25-Assignment1 is a machine learning assignment focused on zero-shot defect classification using OpenAI’s CLIP model. The goal is to classify surface defects in the hazelnut subset of the MVTec-AD dataset without explicit training, by leveraging vision–language alignment.
# CLIP Zero-Shot Defect Classification

This project implements zero-shot defect classification using CLIP on the hazelnut subset of the MVTec-AD dataset.

## Overview

The project performs the following tasks:
1. Downloads and sets up the MVTec-AD hazelnut dataset
2. Implements a Pydantic configuration class for defect classification
3. Creates a CLIP-based zero-shot classifier
4. Runs classification on the test set
5. Generates confusion matrix and analysis
6. Provides observations and recommendations

## Setup

1. Install required packages:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the complete workflow:
\`\`\`python
python scripts/complete_workflow.py
\`\`\`

## Project Structure

- `spec.py`: Pydantic configuration class defining class names, prompts, and model
- `clip_ac.py`: Main CLIP classifier implementation
- `scripts/`: Utility scripts for dataset setup, git workflow, and analysis
- `data/`: Directory for MVTec-AD dataset (hazelnut subset)

## Hazelnut Defect Classes

The hazelnut subset includes the following defect types:
- **good**: Normal hazelnuts without defects
- **crack**: Hazelnuts with crack defects
- **cut**: Hazelnuts with cut defects  
- **hole**: Hazelnuts with hole defects
- **print**: Hazelnuts with print defects

## Usage

### Quick Start
\`\`\`python
from clip_ac import run_classification
from scripts.run_classification_and_analysis import generate_confusion_matrix, analyze_results

# Run classification
results = run_classification("data/hazelnut/test")

# Generate confusion matrix
cm = generate_confusion_matrix(results)

# Analyze results
analysis = analyze_results(results)
\`\`\`

### Custom Configuration
\`\`\`python
from spec import DefectClassificationSpec
from clip_ac import CLIPDefectClassifier

# Create custom config
config = DefectClassificationSpec(
    class_names=["good", "crack", "cut", "hole", "print"],
    prompts=[
        "a photo of a perfect hazelnut",
        "a photo of a cracked hazelnut", 
        "a photo of a cut hazelnut",
        "a photo of a hazelnut with holes",
        "a photo of a hazelnut with print defects"
    ],
    model_name="ViT-L/14"
)

# Initialize classifier
classifier = CLIPDefectClassifier(config)
\`\`\`

## Results and Analysis

The system provides:
- Overall accuracy metrics
- Per-class performance analysis
- Confidence score analysis
- Common misclassification patterns
- Recommendations for improvement

## Git Workflow

The project follows this Git workflow:
1. Clone the ZS-CLIP-AC-naive repository
2. Checkout/create feature branch
3. Implement changes
4. Commit modifications
5. Merge back to main
6. Verify with git log and git show

## Notes

- The MVTec-AD dataset must be manually downloaded from the official source
- CLIP model weights are downloaded automatically on first run
- GPU acceleration is used when available
- Results include detailed analysis and improvement suggestions
