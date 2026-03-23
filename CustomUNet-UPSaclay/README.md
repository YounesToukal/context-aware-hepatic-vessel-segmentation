

# Context-Aware Hepatic Vessel Segmentation with 2.5D EfficientNet U-Net

## Abstract

This repository presents a research-grade implementation of a 2.5D EfficientNet U-Net for automatic segmentation of hepatic vessels in CT scans. The model leverages context-aware slice stacking, deep supervision, and topology-preserving clDice loss to address the challenges of thin, tubular vessel structures in medical imaging.

## Methods

- **2.5D Input**: Each input sample consists of three adjacent CT slices stacked as channels, providing 3D anatomical context while maintaining 2D computational efficiency.
- **Encoder**: EfficientNetB2 pretrained on ImageNet, enabling strong feature extraction from limited medical data.
- **Decoder**: U-Net style with skip connections, residual blocks, and deep supervision for robust multi-scale learning.
- **Loss**: Combination of Binary Cross-Entropy, Dice, and clDice (centerline Dice) to optimize both overlap and vessel connectivity.
- **Data Pipeline**: Custom Keras Sequence for NIfTI/NumPy volumes, patch sampling, and on-the-fly normalization.

## Dataset

- **Source**: Medical Segmentation Decathlon - Task 08: Hepatic Vessel ([link](http://medicaldecathlon.com/))
- **Size**: 303 annotated CT volumes (liver and vessel masks)
- **Preprocessing**: HU windowing, per-patch normalization, and 2.5D stacking
- **Split**: Typical split is 80% train, 10% val, 10% test (user configurable)

## Reproducibility

- All random seeds are set in NumPy and TensorFlow
- Training and validation splits are deterministic if seeds are fixed
- See `train.py` for experiment configuration and callbacks

## Usage

1. **Install dependencies:**

      ```bash
      pip install -r requirements.txt
      ```

2. **Prepare your data and update file paths in** `train.py`

3. **Train the model:**

      ```bash
      python train.py
      ```

4. **For deployment, see** `Dockerfile` **and** `api.py` **(FastAPI route for inference)**

## Project Structure

- `model.py`: Model architecture and loss functions
- `data.py`: Data generator and preprocessing
- `train.py`: Training loop and experiment entry point
- `requirements.txt`: Python dependencies
- `LICENSE`: MIT License
- `api.py`: FastAPI deployment example
- `Dockerfile`: Containerization for reproducible deployment

## Results

The model achieves state-of-the-art performance on the Decathlon hepatic vessel task, with strong vessel connectivity and minimal false positives. See logs and checkpoints for detailed metrics.

## Citation

If you use this code or ideas in your research, please cite:

```bibtex
@misc{toukal2026hepaticvessel,
   author = {Younes Toukal},
   title = {Context-Aware Hepatic Vessel Segmentation with 2.5D EfficientNet U-Net},
   year = {2026},
   url = {https://github.com/YounesToukal/context-aware-hepatic-vessel-segmentation}
}
```

## Contact

**Younes TOUKAL**  
younesstoukal2@gmail.com
