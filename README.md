
This repository showcases MATLAB-based projects focusing on custom implementations of clustering and classification algorithms for various datasets. Each project combines rigorous mathematical methodologies with comprehensive MATLAB coding to address real-world classification challenges, covering techniques such as K-means, K-medoids, and clustering evaluations. **You can find the full description on the pdf file.**

## 📂 Project Overview

1. **Iris Dataset Clustering with K-means and K-medoids**
   - **Objective**: Classify the well-known Iris dataset into three species using custom implementations of K-means and K-medoids algorithms.
   - **Methodology**: The K-means and K-medoids algorithms were implemented from scratch. K-means minimizes within-cluster variance, while K-medoids (using the L1 distance) reduces sensitivity to outliers by choosing actual data points as medoids.
  
<p align="center">
    <img src="https://github.com/user-attachments/assets/92c03ec5-3783-4a03-8929-e90caa8ea12e" alt="Description of the image" width="400">
</p>

   - **Evaluation**: Results are evaluated using confusion matrices and misclassification counts across multiple runs, highlighting each algorithm’s stability and accuracy.
<p align="center">
    <img src="https://github.com/user-attachments/assets/65455a58-0704-4f98-a167-403c0449cbff" alt="Description of the image" width="400">
</p>

2. **Breast Cancer Biopsy Data Analysis with K-medoids**
   - **Objective**: Cluster biopsy data to distinguish between benign and malignant samples, focusing on maximizing accuracy and robustness.
   - **Methodology**: Missing data entries are handled, and K-medoids clustering is employed, using sensitivity and specificity metrics to assess the method’s effectiveness in identifying malignant cases.
<p align="center">
    <img src="https://github.com/user-attachments/assets/24dd9d62-d8ed-4f3a-b316-38a3c23cb167" alt="Description of the image" width="400">
</p>

   - **Evaluation**: Sensitivity and specificity scores reveal the clustering accuracy, making it a practical diagnostic tool. The algorithm is tested across multiple runs to ensure robust results.
3. **1984 Congressional Voting Records Analysis**
   - **Objective**: Investigate partisan voting behavior in the 1984 U.S. Congress by clustering representatives based on their voting patterns.
   - **Methodology**: A dissimilarity matrix was computed using a custom dissimilarity index for "yes"/"no" votes, while managing missing votes by assigning a neutral score. K-medoids clustering is then applied to group representatives by voting alignment.
   - **Evaluation**: Confusion matrices reveal a clear partisan split, showing the effectiveness of K-medoids in political data clustering. Additional analysis assesses the voting consistency within each cluster.
<p align="center">
    <img src="https://github.com/user-attachments/assets/8a1936dd-1f16-4742-8810-cb7675646db8" alt="Description of the image" width="400">
</p>

4. **Wine Classification Using Chemical Properties**
   - **Objective**: Classify Italian wines from three cultivars based on 13 chemical attributes, comparing the performance of K-means and K-medoids algorithms.
   - **Methodology**: Both K-means and K-medoids algorithms are applied to the wine dataset, with clusters evaluated for accuracy against the true wine cultivars.
<p align="center">
    <img src="https://github.com/user-attachments/assets/5c2d9c59-9257-4342-a9ab-ff073d79e8ac" alt="Description of the image" width="400">
</p>

   - **Evaluation**: Confusion matrices are generated to identify clustering accuracy. This project highlights the clustering challenges posed by similar chemical profiles across cultivars and compares the resilience of each algorithm.
<p align="center">
    <img src="https://github.com/user-attachments/assets/53e6d2af-ade0-4ee1-8d87-655d33fc5f8c" alt="Description of the image" width="400">
</p>

5. **Cardiac SPECT Data Clustering for Patient Classification**
   - **Objective**: Classify cardiac patients as normal or abnormal using binary data from SPECT images.
   - **Methodology**: Dissimilarity matrices are created using binary metrics, followed by K-medoids clustering to separate patients based on diagnostic markers.
   - **Evaluation**: Confusion matrices and a custom 2x2 classification matrix evaluate the clustering accuracy, reflecting how well attributes correspond to patient health status.
<p align="center">
    <img src="https://github.com/user-attachments/assets/56907847-3d76-4e83-873e-820c520012dc" alt="Description of the image" width="400">
</p>

## 🔧 Technology Stack
- **Language**: MATLAB
- **Techniques**: K-means clustering, K-medoids clustering, Confusion matrix analysis, Sensitivity and specificity scoring, Robustness testing
- **Datasets**: Iris, Biopsy, Congressional Voting, Wine, and Cardiac SPECT datasets

## 💡 Key Highlights
- **Custom Implementations**: All algorithms are coded from scratch to deepen understanding of clustering mechanics.
- **Robust Evaluations**: Each project includes evaluation metrics like confusion matrices and error analyses to measure clustering quality.
- **Data Insights**: These projects emphasize understanding algorithm strengths and limitations, particularly in handling outliers and noisy data.

## 📊 Visualizations
- Each project is accompanied by detailed plots and figures, including:
  - Cluster assignments with centroid/medoid visualizations
  - Confusion matrices for classification accuracy
  - Performance graphs illustrating algorithm convergence

## 🚀 Getting Started
To explore these projects:
1. Clone this repository.
2. Open MATLAB and navigate to the project folder.
3. Run each script to reproduce the results, visualizations, and metrics outlined above.

```bash
git clone https://github.com/yourusername/MATLAB-Clustering-Classification.git
cd MATLAB-Clustering-Classification
