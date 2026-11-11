Homework: Medical Image Classification
Using Pretrained CNNs
Objective
In this assignment, students will learn how to apply transfer learning to vehicle image
classification using two pretrained convolutional neural networks: ResNet-50 and MobileNetV3.
Students will fine-tune only the final classification layer while freezing all other pretrained layers.
The training process must be monitored and visualized using TensorBoard.

Part 1. Download and Prepare the Vehicle Dataset
1. Download a vehicle image-classification dataset from Hugging Face.
Recommended dataset:
Brain Tumor MRI
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Or Hemg/Brain-Tumor-MRI-Dataset HuggingFace mirror — 155MB, 7,023 images, 4
proper classes (glioma, meningioma, notumor, pituitary).
https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset
2. Use the Hugging Face datasets library to load the images and corresponding class
labels.
3. Divide the original training data into training (80%) and validation (20%) subsets. Keep
the provided test set separate.
4. Preprocess the images:
○ Convert images to RGB.
○ Resize/crop images to 224 × 224 pixels.
○ Convert images into PyTorch tensors.
○ Normalize images using the pretrained models’ ImageNet normalization
parameters.
5. Create PyTorch DataLoader objects for training and validation.

Part 2. Download Pretrained CNN Models
Download the following two ImageNet-pretrained models from Hugging Face:
Model A: ResNet-50
● Hugging Face: https://huggingface.co/timm/resnet50.a1_in1k
Model B: MobileNetV3-Large
● Hugging Face: https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
Use the timm library to load both models with their pretrained weights.

Part 3. Transfer Learning and Fine-Tuning
For each pretrained model:
1. Load the pretrained network.
2. Freeze all pretrained layers so that their parameters are not updated during training.
3. Replace the original final classification layer with a new classification layer whose output
dimension equals the number of vehicle classes.
4. Train only the newly initialized classification layer.
5. Use:
○ Loss function: Cross-Entropy Loss
○ Optimizer: Adam or SGD
○ Batch size: 16 or 32
○ Number of epochs: At least 10
6. Evaluate performance on the validation dataset after every epoch.
Important: Do not fine-tune the complete network. Only the final classification layer may be
trained. Keep the frozen backbone’s BatchNorm running statistics unchanged.

Part 4. Training Visualization Using TensorBoard
Use PyTorch’s SummaryWriter to record the training process.
For both ResNet-50 and MobileNetV3, TensorBoard must display the following four metrics:
1. Training Loss: Average cross-entropy loss on the training dataset for each epoch.
2. Validation Loss: Average cross-entropy loss on the validation dataset for each epoch.
3. Training Accuracy: Percentage of correctly classified training images for each epoch.
4. Validation Accuracy: Percentage of correctly classified validation images for each
epoch.
The horizontal axis should represent the training epoch.
Students must provide TensorBoard screenshots showing:
● Training loss versus validation loss.
● Training accuracy versus validation accuracy.
● The corresponding learning curves for both ResNet-50 and MobileNetV3.
Use separate experiment runs so that the two models can be compared.

Part 5. Performance Comparison
Compare the performance of the two pretrained models using the same dataset partitions.
Report:
● Final training accuracy
● Best validation accuracy
● Corresponding validation loss
● Training time
● Number of trainable parameters
Briefly discuss which model performs better and whether there is evidence of overfitting.
Submission Requirements
Submit the following materials through Moodle:
1. Python source code (.py or .ipynb) for dataset preparation, model fine-tuning,
evaluation, and TensorBoard logging.
2. PDF report, including:
○ Dataset description and class labels
○ Model architectures and fine-tuning configuration
○ Training hyperparameters
○ TensorBoard screenshots of all four required metrics for both models
○ Performance comparison and brief discussion
3. Saved fine-tuned model weights for both ResNet-50 and MobileNetV3.
The source code must clearly demonstrate that only the final classification layer is trainable.



References
● Dataset: https://huggingface.co/datasets/DrBimmer/vehicle-classification
● ResNet-50: https://huggingface.co/timm/resnet50.a1_in1k
● MobileNetV3: https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
● Hugging Face timm documentation: https://huggingface.co/docs/timm/quickstart
● PyTorch TensorBoard tutorial:
https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html