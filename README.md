# dvanced Audio Signal Processing and Recognition  

## Overview  
This project explores advanced methods for audio signal processing and recognition. It builds upon baseline experiments using STFT and MFCC features with an ECOC-SVM classifier and extends the framework with **wavelet transform (CWT)** and **modulation spectrum analysis** to overcome STFT’s time-frequency resolution limitations.  

The goal is to design a reproducible pipeline that integrates **classical DSP** and **modern machine learning**, improving recognition accuracy, robustness under noisy conditions, and enabling a lightweight real-time prototype.  

---

## Objectives  
- Develop a reproducible audio recognition pipeline.  
- Introduce **wavelet transform (CWT)** and **modulation spectrum features** for non-stationary signal analysis.  
- Enrich feature sets with time-domain, frequency-domain, and advanced time-frequency descriptors.  
- Apply **FM synthesis and data augmentation** to improve robustness and class balance.  
- Compare **classical ML methods (SVM, Random Forests)** with **deep learning models (CNNs, CRNNs)**.  
- Deliver a **real-time recognition demo**.  

---

## Methodology  
1. **Dataset**: Public datasets (e.g., NSynth, UrbanSound8K) + FM-synthesized signals.  
2. **Preprocessing**: Resampling, pre-emphasis, framing, windowing.  
3. **Time-Frequency Analysis**: STFT spectrograms, wavelet scalograms, modulation spectrum analysis.  
4. **Feature Extraction**:  
   - Time-domain: RMS, energy entropy, ZCR  
   - Frequency-domain: centroid, rolloff, flatness, flux  
   - Time-frequency: MFCCs, Mel-spectrogram, CQT  
   - Advanced: wavelet coefficients, modulation descriptors, F0, harmonic ratio  
5. **Modeling**:  
   - Classical: SVM, Random Forest, Gradient Boosting  
   - Deep Learning: 1D CNNs (raw waveforms), 2D CNNs (spectrograms/scalograms), CRNNs  
6. **Evaluation**: Accuracy, F1-score, confusion matrix, robustness under noise.  
7. **Prototype**: Real-time recognition demo (MATLAB App Designer / Python GUI).  

---

## Expected Outcomes  
- A complete and reproducible **end-to-end audio recognition framework**.  
- Improved recognition accuracy and robustness compared to MFCC-only baselines.  
- Comparative insights between **STFT vs CWT**, **traditional vs modulation-based features**.  
- A real-time prototype demonstrating practical audio classification.  

---

## Timeline (Weeks 6–13)  
- **Weeks 6–7**: Literature review, dataset collection  
- **Weeks 8–9**: Preprocessing & implementation of STFT, wavelet, modulation analysis  
- **Weeks 10–11**: Feature extraction, model training, baseline evaluation  
- **Week 12**: Optimization, robustness testing  
- **Week 13**: Final evaluation, real-time prototype, GitHub documentation  

