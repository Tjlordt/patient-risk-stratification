Patient Risk Stratification Based on Healthcare Cost

Overview

This study develops an unsupervised machine learning approach to stratify patients into distinct risk categories based on healthcare cost, demographic characteristics, and lifestyle factors. The primary objective is to identify meaningful patient segments that reflect varying levels of healthcare utilisation and financial burden. Such stratification is essential for supporting targeted interventions, improving resource allocation, and enhancing population health management strategies.

Unlike supervised models that rely on predefined outcomes, this work adopts a data-driven clustering approach, allowing inherent structures within the dataset to define risk groupings. This mirrors real-world healthcare analytics practices, where risk segmentation is often derived from patterns in utilisation and cost data rather than explicit labels.



Methodology

Data Preparation

The dataset was first preprocessed to ensure suitability for clustering. Categorical variables such as sex, smoking status, and region were transformed into numerical representations using one-hot encoding. Continuous variables were standardised to ensure comparability across features, given that clustering algorithms are sensitive to scale.

Clustering Approach

KMeans clustering was employed to partition the patient population into homogeneous groups. To determine the optimal number of clusters, both the Elbow Method and Silhouette Score were applied. These techniques provided complementary insights into cluster compactness and separation, supporting the selection of three clusters as the most appropriate representation of the data structure.

Risk Stratification

Following clustering, each group was interpreted based on its average healthcare cost. Clusters were then mapped to Low, Medium, and High Risk categories, with higher average costs indicating greater risk. This approach aligns with established healthcare practices, where cost and utilisation serve as proxies for patient complexity and risk.

Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce the high-dimensional feature space into two principal components. This enabled visualisation of cluster separation and provided qualitative validation of the clustering results.



Results and Insights

The clustering analysis revealed distinct patient segments characterised by varying levels of healthcare cost and associated risk factors. High-risk patients were generally associated with significantly higher medical charges, increased BMI, and a higher likelihood of smoking. These findings are consistent with established evidence linking lifestyle factors and chronic conditions to increased healthcare utilisation.

Medium-risk groups displayed moderate cost patterns, suggesting transitional patient profiles that may benefit from early intervention. Low-risk groups were characterised by relatively low healthcare expenditure and fewer high-risk indicators.

The PCA visualisation demonstrated reasonable separation between clusters, supporting the validity of the segmentation.



Implications

The results highlight the value of unsupervised learning in identifying patient risk profiles without reliance on labelled outcomes. In practical settings, such stratification can inform:
	•	proactive care planning for high-risk patients
	•	targeted prevention strategies for medium-risk groups
	•	efficient allocation of healthcare resources
	•	cost management initiatives within insurance and healthcare systems



Limitations

While the methodology reflects real-world analytical approaches, the dataset used is relatively simplified and lacks detailed clinical variables such as diagnoses, lab results, and longitudinal patient history. As such, the stratification is primarily driven by cost and demographic features, which may not fully capture clinical risk.

Future work could incorporate richer healthcare datasets to improve the depth and clinical relevance of the segmentation.



Conclusion

This project demonstrates how unsupervised machine learning can be applied to derive meaningful patient risk segments from healthcare cost data. By allowing patterns within the data to define groupings, the approach provides a flexible and scalable framework for population health analysis. Despite data limitations, the findings offer practical insights into patient risk distribution and highlight the potential of clustering techniques in healthcare decision-making.
