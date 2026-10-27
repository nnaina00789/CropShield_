# CropShield — Results Log

## Model 1: SimpleCNN (baseline, trained from scratch)
- Architecture: 4 conv blocks (Conv-BN-ReLU-MaxPool, 32→256 filters), global avg pool, dropout 0.3, linear (38 classes)
- Parameters: 399,142
- Training: 8 epochs, Adam (lr 0.001), batch size 32, mixed precision, RTX 4050 Laptop GPU
- Time: ~95 s per epoch (~13 min total)
- Best validation accuracy: 96.5% (epoch 7)
- Observation: validation accuracy stayed above training accuracy throughout (augmentation + dropout during training only); no sign of overfitting