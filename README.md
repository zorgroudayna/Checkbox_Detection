# Checkbox Detection
**Checkbox Detector Model using YOLOv8**

---

### Project Overview
This project focuses on detecting checkboxes in scanned document images using a YOLO-based object detection approach. It combines a cropped-based data augmentation strategy with a two-stage inference pipeline to achieve high accuracy on both cropped and full-page documents.

---

### About the Project
One of the main challenges in this project is the lack of publicly available datasets containing document images with accurately annotated checkboxes. Most open-source datasets either include isolated checkbox images or scanned documents without checkbox annotations, making direct training difficult.

To mitigate this, we collected more than 3,000 private document images and manually annotated them using the LabelImg software, and we downloaded approximately 2,500 additional annotated documents from Roboflow. However, this amount of data remains insufficient when dealing with very small objects such as checkboxes in full-page documents.  

Training YOLO directly on full images does not produce false detections, but it fails to detect all checkboxes, resulting in low recall. To address this limitation, we adopted a cropped-based approach as both a data augmentation and training strategy, where each document is divided into multiple horizontal crops. This significantly increases the dataset size to over 24,000 images while making checkboxes appear larger and easier for the model to learn.  

Nevertheless, this introduces a new challenge: the trained model performs well on cropped images but struggles when applied directly to full-page documents. To resolve this, we implemented a second-stage inference strategy in which full documents are dynamically cropped at inference time, detections are performed on each crop, and the results are merged back into the original document using an IoU-based merging method to eliminate duplicate detections.  

This final pipeline enables accurate checkbox detection on full documents while preserving the advantages of cropped-based training and remaining fully compatible with the YOLO architecture and annotation format.

---
![output_0_12_line7](https://github.com/user-attachments/assets/c5f58400-bb0a-49a3-bc68-eac9f0cbc1c8)
