# Multi-View Audio Feature Fusion and Robust Instrument Recognition

## Overview
This project develops an advanced **instrument recognition system** that integrates **multi-view feature representations** with both **statistical** and **deep learning** models.  
The system targets instruments such as **flute, guitar, and vocal**, and is designed to perform robustly under noisy and reverberant conditions.

Key highlights:
- **Multi-view features**: STFT, CQT, wavelet scalograms, cochlear-inspired filterbanks, and modulation spectra.
- **Hybrid modeling**: Statistical baselines (GMM, HMM) fused with deep models (CNN, CRNN, Transformer).
- **Robust augmentation**: SpecAugment++, pitch/tempo perturbation, noise/reverberation injection, MixUp/CutMix.
- **Hierarchical classification**: Family-level → instrument-level.
- **Interactive prototype**: Real-time demo with visualization.

---

## Motivation
Traditional STFT/MFCC methods face challenges in:
- Limited time–frequency resolution, leading to confusion between similar instruments.
- Lack of perceptual modeling of rhythm, modulation, and resonance.
- Poor robustness to noise, reverberation, and spatial variability.

This project addresses these gaps with **advanced signal processing, perceptually aligned features, and robust learning frameworks**.

---

## Methodology

### Dataset & Augmentation
- Dataset: NSynth and extended corpora, resampled at 16 kHz.
- Augmentations:
  - Pitch-shift ±1–2 semitones, tempo ±5–8%.
  - SpecAugment++ with multiple masks.
  - MixUp / CutMix (targeting confusing pairs).
  - Additive noise (20 dB / 10 dB SNR), BRIR/HRIR convolution for reverberation.

### Feature Extraction
- **Time–frequency**: STFT spectrograms, CQT, wavelet scalograms.
- **Auditory-inspired**: Cochlear ERB filterbanks + Δ/ΔΔ.
- **Modulation**: 2D modulation spectra, tempograms.
- **Parametric**: LPC coefficients, harmonic envelope descriptors.
- **Spatial**: ILD/ITD features derived from HRIR/BRIR.

### Modeling
- **Statistical baselines**: GMM, HMM.
- **Deep learning**:
  - CNNs (ResNet variants).
  - CRNN (CNN + Bi-GRU/LSTM).
  - CNN + Transformer encoder.
- **Fusion**:
  - Multi-channel feature concatenation (STFT+CQT+ERB+Modulation).
  - Late-fusion with statistical models.
- **Hierarchical classification**:
  - Stage 1: Instrument family.
  - Stage 2: Fine-grained instrument type.

### Evaluation
- **Metrics**: Accuracy, macro F1, per-class recall, hierarchical confusion matrices.
- **Robustness**: Tested at different SNRs, reverberation times (T60 = 0.2–0.6s), and spatial directions.
- **Perceptual**: segSNR, Harmonics-to-Noise Ratio (HNR), cochlear correlation.
- **Efficiency**: Real-time factor (RTF) comparison for lightweight vs. deep models.

---

## Expected Outcomes
- Overall accuracy ≥ **88–90%** (from ~80% baseline).
- +8–12pt macro-F1 improvement for confusable pairs (e.g., guitar–keyboard–mallet).
- ≥85% accuracy retained under 10 dB SNR and moderate reverberation.
- Comparative insights between statistical and deep models, single-view vs multi-view features.
- Real-time prototype demo showcasing system performance.

---

## Timeline
- **Week 6–7**: Data preparation; implement feature extraction (CQT, wavelet, cochlear, modulation, HRIR).
- **Week 8–9**: Train baselines (SVM, GMM, CNN); test augmentations.
- **Week 10–11**: Implement CRNN, Transformer; hierarchical classification; fusion experiments.
- **Week 12**: Robustness and perceptual metric evaluation.
- **Week 13**: Prototype app, final report, GitHub release.

---

## References
1. Quatieri, T. F. *Discrete-Time Speech Signal Processing*. Pearson, 2002.  
2. Phan, D. T. “Reduce computational complexity for continuous wavelet transform in acoustic recognition using hop size.” *ISETC 2024*. IEEE.  
3. Mirzaei, S., Jazani, I. K. “Acoustic scene classification with multi-temporal modulation spectrogram features.” *Multimedia Tools and Applications*, 2023.  
4. Hossan, M. A., Memon, S., Gregory, M. A. “A novel approach for MFCC feature extraction.” *ICSPCS 2010*.  
5. Chang, C. C., Lin, C. J. “LIBSVM: A library for support vector machines.” *ACM TIST*, 2011.  





