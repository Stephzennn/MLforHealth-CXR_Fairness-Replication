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

\begin{itemize}
\item \textbf{Empirical Risk Minimization (ERM)}: This method minimizes the overall population risk without considering the composition of different groups. The goal is to reduce the average error across the entire population.

    \item \textbf{Balanced ERM}: This technique addresses the issue of imbalanced group sizes by upsampling the minority groups. By doing so, it aims to minimize the risk in a population where all groups are equally represented. This helps ensure that minority groups are not overshadowed by majority groups in the training process.

    \item \textbf{Stratified ERM}: Instead of training a single model for the entire population, this approach involves learning a separate model for each protected group. This allows for more tailored and potentially fairer predictions, as each group's unique characteristics are taken into account.

\end{itemize}
