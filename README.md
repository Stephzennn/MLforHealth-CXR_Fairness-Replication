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

Methodology

In the paper, first there is a benchmarking of three baseline methods

Empirical Risk Minimization (ERM): This method minimizes the overall population risk without considering the composition of different groups. The goal is to reduce the average error across the entire population.

Balanced ERM: This technique addresses the issue of imbalanced group sizes by upsampling the minority groups. While most debiasing methods negatively impact larger sampled cohorts by reducing their performance, this method increases the sampling of the protected group to achieve parity. By doing so, it aims to minimize the risk in a population where all groups are equally represented. This helps ensure that minority groups are not overshadowed by majority groups in the training process.

Stratified ERM: Instead of training a single model for the entire population, this approach involves learning a separate model for each protected group. Intuitively, this method aims to develop a specific model for each group, which are then combined with other groups' models to ensure consistent performance across all groups.

Adversarial (Wadsworth et al., 2018): This method uses an adversarial approach to enforce fairness. An adversary model is trained to detect and penalize any bias in the predictions. By making it difficult for the adversary to find bias, the main model is encouraged to produce fairer predictions, ensuring that S(predictions) are independent of G(protected group), given Y(true label).

MMDMatch (Pfohl et al., 2021a): This method penalizes the Maximum Mean Discrepancy (MMD) distance, which measures the difference between two distributions. It aims to minimize the distance between the distribution of predictions conditioned on both the true label and the protected group, P(S∣Y=y,G=g), and the distribution of predictions conditioned only on the true label, P(S∣Y=y). This reduces biases related to the protected group.

MeanMatch (Pfohl et al., 2021a): Similar to MMDMatch, but it focuses on aligning the means of the distributions. It penalizes the difference in means between P(S∣Y=y,G=g) and P(S∣Y=y). The goal is to ensure that the average predictions are fair across different groups.

FairALM (Lokhande et al., 2020): This method uses an augmented Lagrangian method to enforce fairness constraints. It incorporates fairness constraints into the optimization problem and uses the augmented Lagrangian technique to solve it. This ensures that the model's predictions adhere to specified fairness criteria while minimizing the overall risk.

GroupDRO (Sagawa et al., 2020): This method exponentially increases the weights of groups with the worst loss after each minibatch. The idea is to prioritize the groups that are performing the worst, ensuring that their performance improves over time.

ARL (Lahoti et al., 2020): ARL stands for Adversarial Reweighted Learning. It's a group-unaware method, meaning it doesn't explicitly consider group membership. Instead, it weights each sample using an adversary that tries to maximize the weighted loss. This technique aims to make the model focus on harder-to-predict samples, indirectly improving fairness.

JTT (Liu et al., 2021): Just Train Twice (JTT) is another group-unaware method. It trains an additional classifier that increases the weights of samples misclassified by the initial ERM (Empirical Risk Minimization) classifier. This helps to correct errors and improve the overall model performance, especially for minority or difficult cases.
