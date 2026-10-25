# CardioPredict: Multimodal Cardiovascular Disease Assessment

An automated multimodal diagnostic framework combining tabular clinical indicators, 1D convolutional signal processing, and 2D computer vision for comprehensive cardiovascular risk assessment.

## Research Publication
* **Journal:** Journal of Emerging Technologies and Innovative Research (JETIR)
* **Author Tracking ID:** 557335
* **Paper / Author Portal:** [JETIR Publication Tracking (ID: 557335)](https://www.jetir.org/trackauthorhome.php?a_rid=557335)

---

## Problem & Solution

### Problem
Cardiovascular diseases (CVDs) are the leading cause of global mortality. Conventional clinical assessment frequently relies on fragmented, single-modality evaluations:
* **Tabular clinical data** (demographics, blood pressure, cholesterol, glucose) identify longitudinal risk factors but cannot capture acute electrical cardiac events.
* **ECG readings** (waveform signals or paper printouts) detect arrhythmias and infarctions but lack patient baseline context, and manual interpretation remains prone to inter-observer variability and delays.
* Relying on isolated data streams limits early detection sensitivity and fails to cross-validate physiological indicators against electrophysiological abnormalities.

### Solution
**CardioPredict** unifies heterogeneous diagnostic modalities into a single automated pipeline:
1. **Clinical Risk Modeling:** Random Forest classifiers trained on the Cleveland Heart Disease and Framingham Heart Study cohorts to evaluate systemic cardiovascular risk.
2. **ECG Signal Classification (1D-CNN):** Deep convolutional feature extraction across 5 arrhythmia classes using normalized heartbeat time-series from the MIT-BIH database.
3. **ECG Image Classification (2D-CNN):** Vision-based convolutional network diagnosing 12-lead ECG printouts into 4 clinical categories: *Normal*, *Myocardial Infarction (MI)*, *Abnormal Heartbeat*, and *History of MI*.
4. **Multimodal Risk Fusion:** Decision-level fusion combining clinical probabilities with electrophysiological findings into an aggregated risk tier (*Low*, *Medium*, *High*) alongside actionable clinical recommendations.
5. **Interactive Dashboard:** A lightweight Gradio clinical web application for rapid patient evaluation and real-time ECG image diagnosis.

---
