# Advanced Audio Signal Processing and Recognition Using Wavelet and Modulation Spectrum Features  

## 📖 Overview  
This repository presents an **advanced audio recognition framework** that integrates **digital signal processing (DSP)** and **machine learning** to classify musical instruments from short recordings. Traditional **STFT/MFCC-based approaches** suffer from fixed time–frequency resolution, which limits their ability to capture non-stationary signals with overlapping harmonics.  

To overcome these challenges, this project introduces:  
- **Continuous Wavelet Transform (CWT)** for adaptive multi-resolution analysis.  
- **Modulation Spectrum Features** to capture slower envelope and rhythm-related variations.  
- **Data augmentation and FM synthesis** for improved robustness.  

The system targets instrument recognition (e.g., **flute, guitar, vocal**) and aims to achieve **high accuracy, robustness under noise, and real-time applicability**.  

---

## 🎯 Research Objectives  
- Develop a **reproducible end-to-end pipeline** for audio recognition.  
- Incorporate **CWT** and **modulation spectrum descriptors** to overcome STFT’s limitations.  
- Enrich feature sets with **low-level, time–frequency, and auditory-inspired descriptors**.  
- Employ **FM synthesis and augmentation** (pitch shifting, tempo perturbation, reverberation, noise injection) to improve robustness.  
- Compare **classical ML models** (SVM, Random Forest) with **deep learning architectures** (CNN, CRNN).  
- Deliver a **lightweight real-time prototype** demonstrating interactive instrument recognition.  

---

## 🧩 Methodology  

### 1. Dataset  
- **NSynth**, **UrbanSound8K**, and other open-source corpora.  
- FM-synthesized timbres for augmentation.  

### 2. Preprocessing  
- Resampling to 16 kHz, normalization, mono conversion.  
- Fixed-length cropping (~2.5s) with SpecAugment (time/frequency masking + noise).  

### 3. Time–Frequency Analysis  
- **STFT spectrograms** as baseline.  
- **Wavelet scalograms (CWT)** for adaptive resolution.  
- **Modulation spectra** via 2D FFT on overlapping spectrogram patches.  

### 4. Feature Extraction  
- **Low-level descriptors**: RMS energy, ZCR, spectral centroid, rolloff, flux.  
- **Time–frequency features**: MFCCs, Mel-spectrogram, Constant-Q Transform (CQT).  
- **Advanced features**: Wavelet coefficients, modulation descriptors, ERB filter banks, LPC coefficients.  

### 5. Modeling and Classification  
- **Classical ML**: SVM, Random Forests.  
- **Deep Learning**:  
  - 1D CNNs on raw waveforms.  
  - 2D CNNs (ResNet, VGG) on spectrograms/scalograms.  
  - CRNNs (CNN + GRU/LSTM) for temporal aggregation.  

### 6. Evaluation  
- Metrics: Accuracy, F1-score, confusion matrix.  
- Robustness under additive noise and augmentation.  

### 7. Prototype  
- Real-time recognition demo via **MATLAB App Designer** or **Python GUI (PyQt/Streamlit)**.  

---

## 🔬 Current Progress  
- **Dataset**: NSynth (10 classes, stratified splits, 16 kHz).  
- **Models**:  
  - CNN (ResNet-style) achieves ~80% accuracy.  
  - CRNN with bidirectional GRU enhances temporal modeling.  
  - SVM baseline is reliable for 3-class categorization (music/other/speech).  
- **Challenges**:  
  - Misclassification among **guitar–keyboard–mallet** and **bass–brass** due to overlapping spectral patterns.  
  - Uneven per-class performance (some classes >0.9 accuracy, others ~0.5).  

---

## 🔮 Future Work  
- **Data**: Multi-crop ensembling, hard-example mining, MixUp/CutMix.  
- **Features**: Multi-view fusion (STFT+CQT+Modulation), transient-aware branches.  
- **Models**: Hierarchical classification (family → instrument), Transformer-based CRNN.  
- **Training**: Focal Loss, Cosine LR schedule, Stochastic Weight Averaging (SWA).  

---

## 📊 Expected Outcomes  
- A **fully documented instrument recognition framework** integrating STFT, wavelet, and modulation-based features.  
- **Demonstrable improvements** over conventional MFCC-only baselines.  
- Comparative insights into the trade-offs between **classical ML** and **deep learning**.  
- A **lightweight real-time prototype** showcasing practical audio classification under noisy and real-world conditions.  

---

## 📅 Timeline (Weeks 6–13)  
- **Weeks 6–7**: Literature review, dataset collection.  
- **Weeks 8–9**: Preprocessing & implementation of STFT, CWT, modulation spectrum.  
- **Weeks 10–11**: Feature extraction, model training (SVM, CNN, CRNN), baseline evaluation.  
- **Week 12**: Optimization, robustness testing.  
- **Week 13**: Final evaluation, prototype development, documentation.  

---

## 📜 License  
This project is released under the **MIT License**.  

