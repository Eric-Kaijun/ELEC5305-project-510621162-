# Advanced Audio Signal Processing and Recognition  

---

## Overview  
This project extends initial lab experiments into a comprehensive research-oriented system for audio signal processing and recognition. It integrates advanced time-frequency analysis, richer feature sets, and modern machine learning methods, along with synthesis and augmentation for robustness. The goal is to transform a classroom-level exercise into a scalable project with practical and research value.  

---

## Objectives  
- Build a reproducible pipeline for audio recognition, integrating preprocessing, feature extraction, and classification.  
- Introduce **wavelet transform (CWT)** and **modulation spectrum analysis** to overcome the resolution limitations of STFT.  
- Enrich features with **time-domain, frequency-domain, and time-frequency representations**.  
- Apply **FM synthesis and data augmentation** to mitigate dataset imbalance and improve robustness.  
- Compare **classical machine learning models** with **deep learning architectures**.  
- Develop a **real-time recognition prototype** for interactive demonstration.  

---

## Methodology  
1. **Dataset Preparation & Augmentation**  
   - Extend to a broader range of instruments and voice samples.  
   - Use data augmentation (noise addition, pitch shifting, time stretching) and **FM synthesis** to generate additional training samples.  

2. **Preprocessing & Time-Frequency Analysis**  
   - Perform resampling, pre-emphasis filtering, framing, and windowing.  
   - Compare window functions (Hamming, Hanning, Blackman) and FFT lengths.  
   - Apply **STFT** to obtain spectrograms, and use **CWT** to analyze non-stationary signals, comparing scalograms with spectrograms.  
   - Incorporate **modulation spectrum analysis** (via 2D Welch method) to capture rhythmic and envelope-related patterns.  

3. **Feature Extraction**  
   - **Time-domain**: RMS energy, short-term energy entropy, zero-crossing rate.  
   - **Frequency-domain**: spectral centroid, spectral spread, spectral flatness, spectral rolloff, spectral flux.  
   - **Time-frequency**: Mel-Spectrogram, MFCCs and delta coefficients.  
   - **High-level**: modulation spectrum descriptors, harmonic ratio, fundamental frequency (F0).  

4. **Modeling & Classification**  
   - **Classical methods**: Support Vector Machines (SVM), Random Forests, Gradient Boosting.  
   - **Deep learning methods**:  
     - 1D CNNs for raw waveform processing.  
     - 2D CNNs (e.g., ResNet, VGG) for spectrogram classification.  
     - CRNNs to capture sequential dependencies in audio data.  

5. **Evaluation & Prototype Development**  
   - Evaluate with cross-validation, confusion matrices, and accuracy metrics.  
   - Compare classical and deep models in terms of accuracy, robustness, and computational cost.  
   - Build a **real-time recognition prototype** (MATLAB App Designer or Python GUI) for live audio input and classification.  

---

## Expected Outcomes  
- A complete and reproducible audio classification framework.  
- **Improved recognition accuracy** and robustness under noisy and augmented conditions.  
- Comparative insights into **STFT vs CWT** and **traditional vs modulation-based features**.  
- A working **real-time demonstration system** showcasing the potential of audio signal processing in real-world scenarios.  

---

## Future Work  
- Extend applications to **speech emotion recognition** or **speaker identification**.  
- Apply **transfer learning** with pre-trained audio models (e.g., YAMNet, OpenL3).  
- Explore **explainability methods** to highlight frequency regions that influence model predictions.  
- Optimize models for **embedded or edge device deployment**, enabling low-latency real-time recognition.  

---

## Significance  
This project demonstrates how fundamental audio signal processing concepts can be integrated with advanced feature engineering and modern machine learning to create practical recognition systems. By incorporating **wavelet analysis, modulation spectrum, synthesis-based augmentation, and real-time prototyping**, it advances from **theoretical exercises → engineering solutions → real-world applications**, contributing both academic and practical value.  

