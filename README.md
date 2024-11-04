# MLforHealth-CXR_Fairness-Replication

This is a project draft to replicate research on CXR Fairness.

Based on the research paper, several claims and results are presented. We will empirically test these results and validate the claims.

Introduction
This research paper aims to highlight discrepancies in machine learning model diagnoses among different protected cohorts. These cohorts are classified by age, sex, insurance, and other factors. Despite their similarities in our sociopolitical context, these protected groups are often underrepresented in training datasets due to several factors.

This paper examines a chest X-ray classifier and conducts various tests to validate several claims made by the authors.

We will also evaluate nine debiasing methods and benchmark them to identify the best approach that improves the worst-case outcomes without negatively impacting other cohorts.

Scope of reproducibility

Here, some results of this research paper can be thoroughly validated. Benchmarking and selecting the best-performing debiasing method, particularly for protected groups in the prediction process, can be properly replicated.

One area that may lack thoroughness is in testing the accuracy of the CheXpert labeling, particularly for the "No Finding" label. The paper highlights this by referencing other studies that have pointed out the inaccuracies of the rule-based labeling used in CheXpert.

The paper uses its own expert's labeling as the gold standard to demonstrate discrepancies in accuracy. Without access to a similar expert, we will rely on the expert-labeled data provided by the paper, which may be subject to bias.
