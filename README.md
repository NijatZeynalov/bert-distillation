# BERT Knowledge Distillation for Medical Note Classification

How to compress a BERT model by 96% and speed it up by 46x using knowledge distillation, while retaining approximately 89% of its original performance.

## Problem Statement

The task is to classify clinical doctor's notes into one of five medical specialties:

- Dermatology  
- Gastroenterology  
- Endocrinology  
- Oncology  
- Pulmonology  

## Models

| Model                      | Parameters | Macro F1 Score | Size Reduction | Performance Retained | Relative Speed |
|---------------------------|------------|----------------|----------------|----------------------|----------------|
| bert-base-uncased         | 109M       | 0.9056         | -              | 100%                 | 1x             |
| boltuix/bert-micro (KD)   | 4.4M       | 0.8065         | 96%            | 89%                  | 46x            |

## Dataset

The dataset consists of clinical notes labeled with one of five specialties. The dataset is stored using Hugging Face's `datasets` library in `data/med_dr_notes/`.


## Pipeline Overview

### Step 1: Train the Teacher Model

Fine-tune `bert-base-uncased` using standard supervised learning.

### Step 2: Distill the Student Model

Use `boltuix/bert-micro` as the student and distill knowledge from the teacher using soft logits.

## Method

* Shared tokenizer: Both teacher and student use the same BERT tokenizer.
* Custom `DistillationTrainer`: Combines student loss and Kullback-Leibler divergence with teacher outputs.
* Hyperparameters:

  * `alpha = 0.3`: 30% student loss, 70% distillation loss
  * `temperature = 3.0`: Softens logits for better gradient signal
