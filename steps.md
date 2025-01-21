Okay, let's combine the high-level and low-level implementation plans into a unified, comprehensive plan for the image inpainting project. This plan will interweave high-level objectives with specific implementation details, making it a practical guide for development.

**I. Project Setup and Environment**

*   **High-Level Goal:** Establish a reproducible development environment and project structure.
*   **Low-Level Details:**
    *   **Directory Structure:**
        *   `data/`: Raw dataset, preprocessed data, masks.
        *   `models/`: Model definitions (`generator.py`, `discriminator.py`, `blocks.py`).
        *   `utils/`: Data loading (`data_loader.py`), masking, augmentation.
        *   `training/`: Training scripts (`train.py`), loss functions (`loss_functions.py`).
        *   `evaluation/`: Metrics (`metrics.py`), evaluation script (`evaluate.py`).
        *   `streamlit_app/`: Streamlit application (`app.py`).
        *   `report/`: Report generation (`report_generator.py`), plots.
        *   `docker/`: Dockerfile, `docker-compose.yml`.
        *   `mlruns/`: MLflow output directory.
        *   `requirements.txt`: Project dependencies.
        *   `README.md`: Project overview and setup instructions.
    *   **Environment:**
        *   Create virtual environment using `conda` or `venv`.
        *   Install dependencies from `requirements.txt` (PyTorch, torchvision, NumPy, OpenCV, scikit-image, Streamlit, MLflow, tqdm, tensorboard, Pillow, scipy).
        *   Initialize a Git repository and add a `.gitignore` file.
        *  Set up Dockerfile and docker-compose for application deployment.

**II. Data Handling**

*   **High-Level Goal:** Load, preprocess, and augment the CelebA dataset with custom masks.
*   **Low-Level Details:**
    *   **`utils/data_loader.py`:**
        *   **Dataset Loading:** Use `torchvision.datasets.CelebA` to load images.
        *   **Mask Generation:**
            *   Implement functions to generate random rectangular, elliptical, and free-form masks using `cv2` (`cv2.rectangle`, `cv2.ellipse`, polygon drawing).
            *   Customize mask parameters (position, size, shape) via configurable settings.
            *   Create a binary mask array.
        *   **Data Augmentation:**
            *   Use `torchvision.transforms` for augmentations like `RandomHorizontalFlip`, `RandomRotation(degrees=10)`, `RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`.
            *   Apply augmentation to the images on-the-fly within the dataset class.
        *   **Data Loading Class (`torch.utils.data.Dataset`):**
            *   Implement `__len__` to return the dataset size.
            *   Implement `__getitem__`: Load image, generate mask, apply mask, apply transforms, convert to tensors, and return original and masked images.
        *   **Data Loaders (`torch.utils.data.DataLoader`):** Create train, validation, and test loaders with custom dataset class.
        *   Consider saving preprocessed data as TFRecords/HDF5 for faster loading during training.

**III. Model Architecture**

*   **High-Level Goal:** Design and implement the GAN architecture using PyTorch.
*   **Low-Level Details:**
    *   **`models/blocks.py`:**
        *  **ConvBlock**: Standard Convolutional block with Conv2d, BatchNorm2d, ReLU with stride and padding args
        *  **DeConvBlock:** Transposed Convolution block with ConvTranspose2d, BatchNorm2d, ReLU with stride and padding args.
        *  **AttentionBlock:** Implementation of Self-Attention or Contextual Attention module.
    *   **`models/generator.py`:**
        *   **U-Net:** Encoder-decoder architecture with skip connections.
            *   Encoder: Downsample using `ConvBlock` (conv layers, batch norm, activation).
            *   Bottleneck: Sequence of conv layers.
            *   Decoder: Upsample using `DeConvBlock` (transposed conv layers, batch norm, activation).
            *   Output Layer: `Conv2d` to map final output to the desired channels.
        *   **Attention Integration:** Add `AttentionBlock`s at appropriate layers.
    *   **`models/discriminator.py`:**
        *   **PatchGAN:** Series of convolutional layers (`ConvBlock`) with LeakyReLU activations to output a single value for each "patch."

**IV. Training Logic**

*   **High-Level Goal:** Implement the training loop and loss functions for GAN training using PyTorch.
*   **Low-Level Details:**
    *   **`training/loss_functions.py`:**
        *   **Adversarial Loss:** `torch.nn.BCELoss` (Binary Cross-Entropy) with label smoothing techniques.
        *   **Reconstruction Loss:** `torch.nn.L1Loss` or `torch.nn.MSELoss` (L1 or Mean Squared Error).
        *   **Contextual Loss:** Feature matching at certain discriminator levels or use of a pre-trained VGG to extract feature maps. Perceptual loss based on feature maps.
    *   **`training/train.py`:**
        *   **Optimizers:** Use `torch.optim.Adam` or `torch.optim.RMSprop` with different learning rates for generator and discriminator.
        *   **Training Loop:**
            1.  Iterate over training data loader.
            2.  Zero gradients of generator and discriminator.
            3.  Generate masked images using the generator.
            4.  **Discriminator Loss:** Compute loss using discriminator's predictions on real/fake (generated) images.
            5.  **Generator Loss:** Compute adversarial loss using discriminator's predictions, and reconstruction loss using L1/L2 or feature matching/perceptual loss.
            6.  Update generator and discriminator weights separately using backpropagation and optimizer steps.
            7.  Use MLflow, TensorBoard or Neptune/W&B for tracking training losses.
            8.  Evaluate the model on the validation set, tracking metrics.
            9. Save model checkpoints with best validation loss/metric.
        *   **Learning Rate Scheduler:** Use `torch.optim.lr_scheduler` (e.g., `StepLR`, `ReduceLROnPlateau`).
        *   **Hyperparameter Tuning:** Implement grid search or random search with early stopping.
        *   **Architecture Tuning:** Implement logic for easy model variations swapping, saving and loading weights.

**V. Evaluation**

*   **High-Level Goal:** Calculate metrics and visualize performance on a test set using evaluation scripts.
*   **Low-Level Details:**
    *   **`evaluation/metrics.py`:**
        *   **PSNR:** Use `skimage.metrics.peak_signal_noise_ratio`.
        *   **SSIM:** Use `skimage.metrics.structural_similarity`.
        *   **FID:** Compute FID using `pytorch_fid` library or implement with a pre-trained inception model (using scipy for statistics).
    *   **`evaluation/evaluate.py`:**
        *   Load trained model.
        *   Iterate through the test data loader.
        *   Calculate and store PSNR, SSIM, FID values.
        *   Visualize/save original images, masks, inpainted images for visual inspection.
        *   Generate loss and metrics plots.

**VI. Streamlit Application**

*   **High-Level Goal:** Develop an interactive user interface using Streamlit for real-time inpainting.
*   **Low-Level Details:**
    *   **`streamlit_app/app.py`:**
        *   **UI:**
            *   Use `st.file_uploader` to load images.
            *   Use `st.slider` or `st.text_input` to configure mask parameters.
            *   Use `st.image` to show original, masked, and inpainted images.
            *   Use `st.button` to start inpainting process.
        *   **Logic:**
            *   Load the trained model.
            *   Preprocess the uploaded image for the inpainting function.
            *   Generate mask using the configured parameters.
            *   Run the inpainting operation.
            *   Display the original/masked/inpainted images.

**VII. Reporting**

*   **High-Level Goal:** Generate a comprehensive report detailing the project, methods, and results.
*   **Low-Level Details:**
    *   **`report/report_generator.py`:**
        *   Use `matplotlib` for plot generation.
        *   Use `jinja2` for report templating.
        *   Include sections for dataset description, model architecture with diagrams, training details, loss, metrics, evaluation, comparison of models, hyperparameter selection, library lists, environment details, training and inference time, a bibliography, and a table of completed project items with points.
        *   Collect training metrics (loss curves, validation metrics), visualizations from evaluation, and model comparisons.
        *   Generate the report and include the link to the Git repository.

**VIII. Dockerization**

*  **High-Level Goal:** Containerize the application.
*  **Low-Level Details:**
    *  Create a `Dockerfile` with the project base operating system, install dependencies, and copy all the project code to the image.
    *  Create a `docker-compose.yml` file to configure the services that need to run in the application.

**Implementation Notes:**

*   **Modular Design:** Focus on modularity using functions and classes.
*   **Logging:** Use logging for tracking, debugging, and error handling.
*   **Configuration:** Allow parameters to be configurable (command-line arguments/config files).
*   **Reproducibility:** Implement the code with consistency in mind using Docker and version control.
*   **Error Handling:** Add exception handling and data validation.

This unified plan combines high-level goals with precise implementation steps, making it actionable and easier to follow. By using this plan as a guide, you can create a robust and effective image inpainting system.
