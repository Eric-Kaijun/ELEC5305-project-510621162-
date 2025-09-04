# ELEC5305 Project: Audio Signal Processing and Instrument Recognition

## Objective  
Design a reproducible pipeline for instrument recognition using the NSynth dataset (flute, guitar, vocal). The project explores preprocessing, feature extraction, and classification to evaluate the impact of signal representations on recognition accuracy.  

## Background  
Audio signals are non-stationary and require short-time analysis. Techniques such as pre-emphasis, framing, FFT, and STFT help extract meaningful features. This project demonstrates how classical methods link to modern classification tasks in audio processing.  

## Methodology  
1. **Dataset** – Select flute, guitar, vocal samples from NSynth.  
2. **Preprocessing** – Apply pre-emphasis, framing, and windowing.  
3. **Time-Frequency Analysis** – FFT and STFT for spectral insights.  
4. **Feature Extraction** – ZCR, Energy, Entropy, MFCCs, Harmonic Ratio, F0.  
5. **Classification** – ECOC-SVM for instrument recognition (~80% accuracy).  
6. **FM Synthesis** – Generate synthetic woodwind-like signals for augmentation.  

## Expected Outcomes  
- End-to-end pipeline for audio instrument recognition.  
- Evaluation of feature contributions and classification results.  
- Visualizations (spectrograms, t-SNE plots) showing class separability.  

## Future Work  
- Expand dataset with more instruments.  
- Apply CNNs on spectrograms for higher accuracy.  
- Real-time demo via MATLAB App Designer or Python GUI.  
