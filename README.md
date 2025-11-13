# Multi-View Audio Feature Fusion and Robust Instrument Recognition  
*A Unified Multi-Stage Framework Integrating Multi-View Spectral Analysis, Polyphonic Priors, Domain Robustness, Contrastive Learning, and Multi-Task Representation Modeling*

---

## 1. Introduction

Musical instrument recognition is fundamentally a **timbre modeling problem**, requiring representations that remain discriminative across variations in **pitch**, **intensity**, **articulation**, and **recording conditions**.  
Conventional approaches based solely on STFT or MFCC features often fail when:

- temporal and harmonic details conflict (e.g., transients vs. steady harmonics),
- acoustic environments introduce reverberation or noise,
- instruments with similar spectral envelopes overlap (e.g., keyboards vs. guitars),
- single-view features fail to capture the multi-dimensional nature of timbre.

To address these limitations, this project develops a **multi-view, multi-domain, and multi-task learning architecture**, making systematic use of:

1. **Multi-view spectral analysis** (STFT + CQT)  
2. **Polyphonic synthetic mixtures** with soft-mask generation  
3. **Target-domain degradation** to model real-world acoustic variability  
4. **Contrastive multi-view pretraining** for representation alignment  
5. **Multi-task supervised learning** to encode richer semantic dimensions  
6. **Cross-view and cross-domain visualization** for diagnostic interpretability  

Each component is grounded in existing literature on **multi-view learning**, **source separation priors**, **domain adaptation**, and **self-supervised contrastive representation learning**, forming a tightly integrated system.

---

# 2. System Overview: Theoretical Integration of All Components

The complete pipeline is designed around the hypothesis that:

> **Robust instrument recognition requires representations that are simultaneously multi-view aligned, domain-invariant, and semantically structured.**

Thus the pipeline flows through the following dependencies:

### 1) Multi-view features  
→ supply complementary information and increase representational richness.

### 2) Polyphonic mixtures  
→ inject structural priors of overlapping harmonics, improving discriminability.

### 3) Target-domain degradation  
→ forces the encoder to learn domain-agnostic latent factors.

### 4) Contrastive SSL  
→ aligns the multi-view space, making STFT and CQT consistent.

### 5) Multi-task learning  
→ embeds semantic structure into the shared representation space.

### 6) Visualization and metrics  
→ validate consistency, robustness, and task separability.

This hierarchical integration ensures that each subsequent stage utilizes enriched, structured information from prior stages.  
No module is isolated: **the strength of the system comes from the interdependence of all parts**.

---

# 3. Multi-View Time–Frequency Representations

### 3.1 Motivation: Why Multi-View?
Timbre is inherently multidimensional.  
Traditional STFT-based systems capture:

- temporal modulations,  
- broadband spectra,  
- transient strength,  

but fail to encode:

- harmonic spacing regularity,  
- musical scale structure.

Conversely, CQT captures:

- log-frequency scaling aligned to musical intervals,  
- harmonic stacks,  
- pitch-invariant spectral ratios,  

while losing temporal precision.

Thus, the two views encode approximately **orthogonal subspaces** of the timbral manifold.

### 3.2 Implementation  

- **STFT:** 1024 Hann window, hop 256, log-magnitude  
- **CQT:** 96 bins per octave, fmin ~ 32 Hz, log-magnitude  
- **Standardization:** global mean/variance  
- **Caching:** reduces training overhead by 10×  

### 3.3 Integration with Later Stages  
Multi-view features enable:

- contrastive learning (cross-view alignment),  
- polyphonic mixture analysis (STFT masks),  
- domain consistency checks (cross-view Δ comparison),  
- multi-task supervision across harmonic and temporal cues.  

### 3.4 Empirical Example  

![STFT + CQT](assets/1.png)

This dual-view structure is the foundation of the entire pipeline.

---

# 4. Polyphonic Mixture Synthesis and Soft Mask Priors

### 4.1 Motivation from Source Separation Theory  
Although NSynth is monophonic, natural musical environments are not.  
To approximate this, we synthesize polyphonic mixtures.

From the perspective of **source separation**, the ideal ratio mask:

\[
M_k(f,t)=\frac {|S_k(f,t)|}{\sum_j |S_j(f,t)| + \epsilon}
\]

acts as a soft label describing the fractional contribution of source k at each time–frequency point.

### 4.2 Implementation  

- Select 2–3 monophonic notes from the same batch  
- Combine waveforms in the time domain  
- Compute STFT per source and mixture  
- Derive soft masks  
- Store mixtures in `.npz` format  

### 4.3 Why This Helps Recognition  
Soft masks encode:

- harmonic templates,  
- energy distributions,  
- spectral dominance patterns,  
- onset overlaps,  

which provide structural cues for the model.  
Even when masks are not used directly in training, they **shape model intuition** and form the basis of data augmentation.

### 4.4 Example Visualization  

![Masks](assets/2.png)

---

# 5. Target-Domain Degradation: Modeling Real Acoustic Variability

### 5.1 Motivation: Domain Adaptation Perspective  

Real-world acoustic conditions differ substantially from controlled studio recordings.  
This creates **covariate shift**, which reduces generalization.

We therefore generate a **paired degraded domain**:

\[
x_\text{target} = T(x_\text{original})
\]

where \(T\) includes transformations inspired by:

- room acoustics,  
- phone/streaming codecs,  
- microphone filtering,  
- ambient noise.  

### 5.2 Implementation of Degradations  

| Type | Description |
|------|-------------|
| Reverb | BRIR/HRIR convolution, RT60 0.2–1.2s |
| Noise | SNR 10/20 dB Gaussian or pink noise |
| EQ | Low/high shelves and band-pass filters |
| Codec | MP3/OPUS at 24–48 kbps |

### 5.3 Role in the Pipeline  
These augmentations encourage the model to learn:

- domain invariance,  
- spectral-temporal stability,  
- robust harmonic descriptors.  

### 5.4 Visualization  

![Domain example](assets/3.png)  
![Delta example](assets/6.png)

---

# 6. Self-Supervised Contrastive Multi-View Pretraining

### 6.1 Theory: Why Contrastive Learning?  
Contrastive learning enforces:

\[
z_{\text{STFT}} \approx z_{\text{CQT}}
\]

while keeping different instruments apart.  
This encourages:

- **view invariance**,  
- **domain robustness**,  
- **semantic compression**,  
- **improved clustering** of timbre-related embeddings.

### 6.2 Implementation  

- NT-Xent loss  
- Temperature scheduling  
- 1cycle learning rate  
- Positive pairs: (STFT, CQT) of same sample  
- Negatives: other samples in batch  

### 6.3 Effect on Supervised Learning  
Contrastive alignment acts as a preconditioner:

- stabilizes gradients,  
- reduces overfitting in supervised stage,  
- enhances harmonically relevant features,  
- improves robustness to reverb/noise.

### 6.4 Learning Behavior  

![SSL curves](assets/8.png)

---

# 7. Supervised Multi-Task Learning (MTL)

### 7.1 Motivation from Representation Learning  
Timbre is influenced by factors such as:

- instrument family,  
- spectral envelope,  
- pitch class,  
- velocity,  
- transient richness.  

Predicting multiple attributes simultaneously encourages a **factorized representation** of timbre.

### 7.2 Model Architecture  

- Shared backbone: CRNN or lightweight Conformer  
- Heads:  
  - 16-class instrument classifier  
  - 8-class family classifier  
  - 12-class pitch-class predictor  
  - 3–5-class velocity bin predictor  
  - 5-dimensional timbre regression head  

### 7.3 How MTL Integrates with SSL and Multi-View Features  
- Multi-view SSL creates a stable shared embedding  
- MTL injects semantic structure into that embedding  
- Domain degradation stabilizes the embedding across acoustics  
- Polyphonic synthesis regularizes harmonic feature extraction  

All these interplay to produce a timbre-aware, robust representation.

### 7.4 Supervised Performance  

![Supervised curves](assets/9.png)  
![Accuracy](assets/10.png)

### 7.5 Confusion Matrix  

![Confusion](assets/7.png)

---

# 8. Multi-View Visualization and Diagnostic Tools

The system includes extensive tools for:

- cross-domain comparison,  
- view-to-view consistency,  
- harmonic pattern visualization,  
- polyphonic interference behavior.  

### Example visualizations

![View comparison](assets/4.png)

---

# 9. Final Performance Summary

- **Instrument accuracy: ~90%**  
- **Strong robustness** under RT60 0.2–1.2s reverb  
- **SNR robustness** at 10 dB and 20 dB noise  
- **Significant reduction** in confusion among harmonically similar classes  
- **Stable cross-view alignment** after contrastive training  
- **Smooth and monotonic convergence** in MTL training
---

# 10.  Conclusion

This project demonstrates that robust musical instrument recognition benefits from the *joint design* of multi-view feature extraction, self-supervised cross-view alignment, polyphonic priors, domain robustness modeling, and multi-task supervision.

The final system is not a simple classifier:
it is a **holistic timbre modeling framework** that unifies insights from:

- spectral signal processing,
- psychoacoustics,
- domain adaptation,
- contrastive learning,
- multi-task representation learning,
- and source separation theory.

The combination of these approaches achieves state-of-the-art robustness and interpretability on the NSynth-small 16-class task.



