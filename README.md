# Custom CNN's for Vegetable Classification
This project includes 3 Custom CNN's built over-time while overcoming the challenges and computer vision pipeline that classifies 15 types of vegetables in real-time.

This project shows Architecture Evolution of CNN's to achieve 99.67% test accuracy on Baseline CNN, 99.50% test accuracy on V2 CNN & 99.43% test accuracy on V3 CNN deployed via a custom OpenCV interface with temporal smoothing for highly stable live-webcam inference (only V2 & V3 CNN's).

**Dataset :** https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

## Model Architecture

**V1: Baseline Custom CNN :-**

The V1 model follows a hierarchical feature-extraction design, using the four Convolutional blocks followed by a globally pooled classification head.

* Input Layer: Expects standardized RGB images of shape (224, 224, 3).

* Preprocessing: Built-in Rescaling(1./255) layer to normalize pixel values from [0, 255] to [0, 1], ensuring stable gradient descent.

* Feature Extraction (4 Blocks):
  1. Block 1: Conv2D (32 filters, 3x3 kernel, ReLU) → BatchNormalization → MaxPooling2D (2x2). Extracts low-level features (edges, raw colors).
  2. Block 2: Conv2D (64 filters, 3x3 kernel, ReLU) → BatchNormalization → MaxPooling2D (2x2).
  3. Block 3: Conv2D (128 filters, 3x3 kernel, ReLU) → BatchNormalization → MaxPooling2D (2x2).
  4. Block 4: Conv2D (256 filters, 3x3 kernel, ReLU) → BatchNormalization → MaxPooling2D (2x2). Extracts high-level, complex semantic features.

* GlobalAveragePooling2D: Replaces standard Flatten layers to drastically reduce the parameter count and prevent spatial overfitting.

* Dense (256 units, ReLU): Interprets the pooled features.

* Dropout (0.5): Randomly disables 50% of neurons during training to enforce redundant learning paths.

* Dense (15 units, Softmax): Outputs a probability distribution across the 15 vegetable classes.

**V2: Robust Custom CNN (Data Augmentation Integration) :-**

The V2 model retains the exact Convolutional and Classification structure of V1 but includes a real-time, GPU-accelerated Real-World Sim at the very beginning of the pipeline.

* The Augmentation Block: Inserted directly after the Input layer.

* RandomFlip("horizontal")

* RandomRotation(0.15): Spins the image up to ~50 degrees to prevent spatial memorization.

* RandomZoom(0.2): Simulates varying distances from the webcam.

* RandomContrast(0.2) & RandomBrightness(0.2): Violently alters the lighting conditions, destroying specular highlights.

* Regularization Update: Dropout increased from 0.5 to 0.6 to further penalize network memorization in response to the artificially chaotic data.

**V3: HSV-Optimized Custom CNN :-**

The V3 model introduces a fundamental shift in how the network perceives pixel data by moving away from standard RGB space.

* Refined Augmentation: The destructive RandomContrast and RandomBrightness layers were removed to preserve the integrity of the color channels. Rotation and Zoom were retained.

* The Lambda Injection: After the Rescaling layer, a custom GPU-level mathematical operation was injected:

  layers.Lambda(lambda x: tf.image.rgb_to_hsv(x))

* Feature Extraction: The 4 Convolutional blocks remain identical, but they no longer process Red, Green, and Blue. They now process:
  1. Hue : The pure color identity (e.g., Red vs. Brown).
  2. Saturation : Color intensity.
  3. Value : Lighting and shadows.
 
## Training & Optimization Methodology

All the models shares the exact same underlying dataset (21,000 images across 15 vegetable classes), the training methodologies were sequentially changed to diagnose and cure specific inference failures (Domain Bias and Colorblindness).

In all the architectures the core hyperparameters and optimization loops were kept identical.
* Image Dimensions: Standardized to 224x224x3 (RGB) to match industry standard transfer-learning inputs.

* Loss Function: Categorical Crossentropy (optimized for multi-class probability distributions).

* Optimizer: Adam (Adaptive Moment Estimation) starting at a default learning rate of 0.001 for efficient gradient descent.

All models utilized a strict, dynamic training loop to prevent overfitting and ensure optimal convergence:

* EarlyStopping: Monitored val_loss with a patience of 10 epochs. If the validation loss stopped improving, training halted and the best weights were restored.

* ReduceLROnPlateau: Monitored val_loss with a patience of 3 epochs. If the model plateaued, the learning rate was slashed by a factor of 0.2 (down to a minimum of 1e-6) to allow the optimizer to "settle" into the global minimum.

* ModelCheckpoint: Strictly saved only the epoch with the highest val_accuracy.

**V1: Baseline Custom CNN :-**

Data : Images were fed into the network exactly as they appeared in the dataset, with only a standard 1./255 pixel normalization applied.

Training Dynamics: Because the data was static and clean, the model converged extremely rapidly. The training accuracy easily outpaced the validation accuracy, requiring heavy Dropout (0.5) to prevent catastrophic overfitting.

Flaw: When deployed to a live webcam, the model failed completely. Grad-CAM "X-Ray" analysis revealed the model was cheating: it wasn't looking at the vegetables; it was looking at the specular highlights (the shiny white glares caused by studio lighting). Because the webcam environment had different lighting, the model's domain bias caused it to collapse.

**V2: Robust Custom CNN :-**

Data : To cure the V1 lighting bias GPU-accelerated Data Augmentation block was injected. Every single epoch, the images were randomly:

Flipped horizontally.

Rotated up to ~50 degrees (0.15).

Zoomed in/out by 20% (0.2).

Shifted in Contrast and Brightness by 20% (0.2).

Training Dynamics (The "Validation Flip"): This methodology essentially placed "weights" on the neural network. Training accuracy plummeted initially because the model was constantly fighting extreme shadows and violent rotations. However, because Keras naturally disables augmentations during the validation step, the model exhibited a textbook Validation Flip—where val_accuracy (tested on clean data) consistently tracked higher than training accuracy.

Flaw: By aggressively penalizing the model for relying on lighting, we accidentally made it quasi-colorblind. During live inference, it confidently misclassified Tomatoes as Potatoes. Because a Tomato and a Potato share the exact same spherical geometry, a colorblind model cannot tell them apart.

**V3: HSV-Optimized Custom CNN :-**

Data : The destructive RandomContrast and RandomBrightness layers were removed to preserve the integrity of the color channels. Geometric augmentations (Flip, Rotate, Zoom) were retained to prevent spatial memorization.

Conversion (RGB to HSV): The fundamental methodology changed from altering the environment to altering the representation. A custom Lambda layer was injected to mathematically convert the normalized RGB pixels into HSV (Hue, Saturation, Value) space directly on the GPU during the forward pass.

Training Dynamics: By training on HSV, the Convolutional filters were forced to evaluate the "Hue" (pure color) entirely independently of the "Value" (lighting/shadows).

Flaw: When presented with a partially occluded Carrot (e.g., fingers covering the ends), the model guessed Potato. Because the Hue of a Carrot (Orange) is mathematically adjacent to a Potato (Light Brown), the model needed to look at microscopic skin textures to tell them apart. The lightweight Custom CNN simply lacked the parameter depth (brainpower) to learn micro-textures.

Result: The model converged faster than V2 (due to the removal of violent contrast shifts) and successfully decoupled object color from environmental lighting. It cured the V2 colorblindness, allowing a lightweight custom architecture to reliably differentiate overlapping geometric classes in live webcam conditions.

**EfficientNet: The Micro-Texture Solution :-**
I've compared V3 & EfficientNet side by side in the last of the jupyter notebook , it was to show how efficientnet can overcome the hard parameter ceiling of a from-scratch CNN and resolve the final Carrot/Potato micro-texture ambiguity because EfficientNet already possessed deeply learned mathematical filters for organic textures (wood, skin, leaves), it immediately recognized the microscopic difference between Carrot skin and Potato skin. In a live dual-inference showdown, EfficientNet maintained a flawless, 90%+ confidence lock on the Carrot despite heavy occlusion and dynamic rotation, officially closing the pipeline.

## Result & Evaluation

**V1: Baseline Custom CNN :-**

Final Test Accuracy: 99.67%

Final Test Loss: 0.0249

**V2: Robust Custom CNN :-**

Final Test Accuracy: 99.67%

Final Test Loss: 0.0249

**V3: HSV-Optimized Custom CNN :-**

Final Test Accuracy: 99.43%

Final Test Loss: 0.0223

**Note: All the webcam results are present in "Live Results" folder provided above**

