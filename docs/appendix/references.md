# 參考資料與授權

本頁彙整各章實際引用的來源，同一來源在多章出現時合併為一筆，「章節」欄列出用到的章。授權欄記的是撰寫時實際查到的內容；標「未驗證」者代表沒有逐頁核對，重新散布前請自行查證。

!!! note "引用原則"
    - 教材、課程與書籍多數**只連結、未改作**；文字為 CC BY-NC-ND 者（如 VanderPlas）更不得改作。
    - 論文只引用其結論或摘要層級的事實，並在正文標明年份與範圍；請以原文為準。
    - 圖表與互動 demo 皆為本站自產，沒有使用第三方圖片。

## 一、教材與課程

| 標題 | 作者／機構 | 年份 | 授權 | 章節 |
|---|---|---|---|---|
| [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)（含 [Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data)、[Neural networks](https://developers.google.com/machine-learning/crash-course/neural-networks)、[ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)、[Imbalanced datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)、[Intro to Large Language Models](https://developers.google.com/machine-learning/crash-course/llm)） | Google | 持續更新 | 內容 CC BY 4.0、程式 Apache 2.0（依研究筆記；未逐頁確認）；只連結、只轉述概念 | 01、03、04、09、11、15 |
| [ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners)（含 [2-Regression](https://github.com/microsoft/ML-For-Beginners/tree/main/2-Regression)、[5-Clustering](https://github.com/microsoft/ML-For-Beginners/tree/main/5-Clustering)） | Microsoft | 持續更新 | MIT；教法參考、未複製內容 | 01、04、08 |
| [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)（第 05.05 Naive Bayes、05.06 Linear Regression、05.07 SVM、05.08 Random Forests、05.09 PCA、05.11 k-Means 各節） | Jake VanderPlas | 2016 | 文字 CC BY-NC-ND、程式碼 MIT；**只連結、未改作** | 02、04、05、06、07、08、10 |
| [Dive into Deep Learning](https://d2l.ai/)（[Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)、[Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)、[Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/index.html)、[Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)） | Zhang A, Lipton ZC, Li M, Smola AJ | 持續更新 | 文字 CC BY-SA 4.0、程式碼 modified MIT；教學順序參考、未複製內文；第 14 章 RNN 為延伸閱讀 | 09、12、14、15 |
| [李宏毅 Machine Learning 2021 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) | 李宏毅（國立臺灣大學） | 2021 | 未驗證；只連結 | 09、12、15 |
| [StatQuest: Naive Bayes, Clearly Explained](https://www.youtube.com/results?search_query=statquest+naive+bayes+clearly+explained)（YouTube 搜尋連結，未確認單一影片網址） | Josh Starmer | — | 版權屬作者，未查證；只連結 | 05 |
| [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | NumPy 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) | pandas 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) | pandas 社群 | 持續更新 | BSD-3-Clause（未逐頁核對）；第 2 章陷阱 1 的官方依據 | 02 |
| [Matplotlib Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) | Matplotlib 社群 | 持續更新 | PSF-based license（未逐頁核對） | 02 |
| [SciPy statistics tutorial](https://docs.scipy.org/doc/scipy/tutorial/stats.html) | SciPy 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [Keras 3 文件](https://keras.io/getting_started/) | Keras 團隊 | 持續更新 | Apache-2.0 | 09、12、13、14、15 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar | 2018 | 部落格 repo MIT，只連結 | 15 |

### scikit-learn 官方文件

授權皆為 BSD-3-Clause。

| 頁面 | 章節 |
|---|---|
| [An introduction to machine learning](https://scikit-learn.org/stable/tutorial/basic/tutorial.html) | 01 |
| [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)（資料洩漏示範構想，程式自行重寫） | 03、11 |
| [Column Transformer with Mixed Types](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) | 03 |
| [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) | 04 |
| [1.8 release notes](https://scikit-learn.org/stable/whats_new/v1.8.html)（`LogisticRegression` 的 `penalty` 棄用，第 4 章陷阱 3） | 04 |
| [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) | 05 |
| [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | 05、11 |
| [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) | 06 |
| [RBF SVM parameters](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) | 06 |
| [1.9 What's New](https://scikit-learn.org/stable/whats_new/v1.9.html)（`SVC(probability=True)` 棄用，第 6 章陷阱 2） | 06 |
| [Decision Trees](https://scikit-learn.org/stable/modules/tree.html) | 07 |
| [Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest) | 07 |
| [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) | 07 |
| [Clustering: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means) | 08 |
| [`MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) | 09 |
| [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) | 10 |
| [Decomposing signals in components (PCA)](https://scikit-learn.org/stable/modules/decomposition.html) | 10 |
| [Algorithm cheat-sheet](https://scikit-learn.org/stable/machine_learning_map.html)（流程圖結構參考，本站自行重畫） | 11 |
| [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)、[Grid search](https://scikit-learn.org/stable/modules/grid_search.html) | 11 |

## 二、論文與書籍

### 機器學習在醫學的應用與方法學

| 標題 | 作者 | 年份 | 出處／識別碼 | 授權 | 章節 |
|---|---|---|---|---|---|
| Machine Learning in Medicine | Deo RC | 2015 | Circulation 132(20):1920-30；[PMID 26572668](https://pubmed.ncbi.nlm.nih.gov/26572668/)、PMC5831252 | 期刊版權；只引用、不改作 | 01 |
| Machine Learning in Medicine | Rajkomar A, Dean J, Kohane I | 2019 | N Engl J Med 380(14):1347-58；[PMID 30943338](https://pubmed.ncbi.nlm.nih.gov/30943338/) | 期刊版權；只連結 | 01 |
| How to Read Articles That Use Machine Learning: Users' Guides to the Medical Literature | Liu Y, Chen PC, Krause J, Peng L | 2019 | JAMA 322(18):1806-16；[DOI 10.1001/jama.2019.16489](https://doi.org/10.1001/jama.2019.16489)；[PMID 31714992](https://pubmed.ncbi.nlm.nih.gov/31714992/) | 期刊版權；只引用（讀摘要） | 01、11 |
| TRIPOD+AI statement | Collins GS, et al. | 2024 | BMJ 385:e078378；[DOI 10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)；[PMID 38626948](https://pubmed.ncbi.nlm.nih.gov/38626948/)（PMC11019967） | 期刊版權；只引用 | 11 |
| A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models | Christodoulou E, et al. | 2019 | J Clin Epidemiol 110:12-22；[DOI 10.1016/j.jclinepi.2019.02.004](https://doi.org/10.1016/j.jclinepi.2019.02.004)；[PMID 30763612](https://pubmed.ncbi.nlm.nih.gov/30763612/) | 期刊版權；只引用（數字取自 PubMed 摘要） | 11 |
| Why do tree-based models still outperform deep learning on typical tabular data? | Grinsztajn L, Oyallon E, Varoquaux G | 2022 | NeurIPS Datasets and Benchmarks；[arXiv 2207.08815](https://arxiv.org/abs/2207.08815) | arXiv 論文；只引用結論 | 09 |
| Bias in random forest variable importance measures: illustrations, sources and a solution | Strobl C, Boulesteix AL, Zeileis A, Hothorn T | 2007 | BMC Bioinformatics 8:25；[DOI 10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25)；[PMID 17254353](https://pubmed.ncbi.nlm.nih.gov/17254353/) | 開放取用（CC BY）；只引用 | 07 |
| Random Forests | Breiman L | 2001 | Machine Learning 45:5–32；[DOI 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | 期刊版權；只連結（書目憑既有知識，未線上查證） | 07 |
| Variable selection – A review and recommendations for the practicing statistician | Heinze G, Wallisch C, Dunkler D | 2018 | Biom J 60(3):431–449；[DOI 10.1002/bimj.201700067](https://doi.org/10.1002/bimj.201700067)；[PMID 29292533](https://pubmed.ncbi.nlm.nih.gov/29292533/)（PMC5969114） | 期刊文章（PMC 開放取用）；只引用觀點 | 10 |
| Regression Modeling Strategies, 2nd ed. | Harrell FE Jr. | 2015 | Springer；[DOI 10.1007/978-3-319-19425-7](https://doi.org/10.1007/978-3-319-19425-7) | 商業書籍；只引用書名與一般觀點 | 10 |
| Novel subgroups of adult-onset diabetes and their association with outcomes: a data-driven cluster analysis of six variables | Ahlqvist E, et al. | 2018 | Lancet Diabetes Endocrinol 6:361–369；[PMID 29503172](https://pubmed.ncbi.nlm.nih.gov/29503172/) | 期刊版權；只引用事實（依 PubMed 摘要） | 08 |
| Phenomapping for novel classification of heart failure with preserved ejection fraction | Shah SJ, et al. | 2015 | Circulation 131:269–279；[PMID 25398313](https://pubmed.ncbi.nlm.nih.gov/25398313/) | 期刊版權；只引用事實（依 PubMed 摘要） | 08 |
| Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study | Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK | 2018 | PLoS Med 15(11):e1002683；[DOI 10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683)；[PMID 30399157](https://pubmed.ncbi.nlm.nih.gov/30399157/)（PMC6219764） | CC BY 4.0（PLoS 開放取用；未逐頁核對）；數字取自 PubMed 摘要 | 12、13 |
| Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists | Rajpurkar P, Irvin J, Ball RL, et al. | 2018 | PLoS Med 15(11):e1002686；[DOI 10.1371/journal.pmed.1002686](https://doi.org/10.1371/journal.pmed.1002686)；[PMID 30457988](https://pubmed.ncbi.nlm.nih.gov/30457988/)（PMC6245676） | CC BY 4.0（PLoS 開放取用；未逐頁核對）；數字取自 PubMed 摘要 | 12 |
| Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | Selvaraju RR, Cogswell M, Das A, et al. | 2020 | Int J Comput Vis 128:336–359；[DOI 10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7)（arXiv:1610.02391） | 期刊版權；只引用方法 | 13 |
| Sanity Checks for Saliency Maps | Adebayo J, Gilmer J, Muelly M, et al. | 2018 | NeurIPS 2018；[arXiv:1810.03292](https://arxiv.org/abs/1810.03292) | arXiv；只引用結論 | 13 |
| The false hope of current approaches to explainable artificial intelligence in health care | Ghassemi M, Oakden-Rayner L, Beam AL | 2021 | Lancet Digit Health 3(11):e745；[DOI 10.1016/S2589-7500(21)00208-9](https://doi.org/10.1016/S2589-7500(21)00208-9)；[PMID 34711379](https://pubmed.ncbi.nlm.nih.gov/34711379/) | 期刊版權；只引用論點 | 13 |
| Benchmarking saliency methods for chest X-ray interpretation | Saporta A, et al. | 2022 | Nat Mach Intell 4:867；[DOI 10.1038/s42256-022-00536-x](https://doi.org/10.1038/s42256-022-00536-x) | 期刊版權；只引用結論 | 13 |
| Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition | Winkler JK, Fink C, Toberer F, et al. | 2019 | JAMA Dermatol 155(10):1135；[DOI 10.1001/jamadermatol.2019.1735](https://doi.org/10.1001/jamadermatol.2019.1735)；[PMID 31411641](https://pubmed.ncbi.nlm.nih.gov/31411641/) | 期刊版權；只引用結論 | 13 |
| Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs | Gulshan V, Peng L, Coram M, et al. | 2016 | JAMA 316(22):2402；[DOI 10.1001/jama.2016.17216](https://doi.org/10.1001/jama.2016.17216)；[PMID 27898976](https://pubmed.ncbi.nlm.nih.gov/27898976/) | 期刊版權；數字取自摘要 | 13 |
| A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics for the Detection of Diabetic Retinopathy | Beede E, et al. | 2020 | CHI 2020；[DOI 10.1145/3313831.3376718](https://doi.org/10.1145/3313831.3376718) | ACM；只引用結論 | 13 |
| Real-time diabetic retinopathy screening by deep learning in a multisite national screening programme: a prospective interventional cohort study | Ruamviboonsuk P, Tiwari R, Sayres R, et al. | 2022 | Lancet Digit Health 4(4):e235–e244；[DOI 10.1016/S2589-7500(22)00017-6](https://doi.org/10.1016/S2589-7500(22)00017-6)；[PMID 35272972](https://pubmed.ncbi.nlm.nih.gov/35272972/) | 期刊版權；數字取自 PubMed 摘要 | 13 |
| Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices | Abràmoff MD, Lavin PT, Birch M, et al. | 2018 | NPJ Digit Med 1:39；[DOI 10.1038/s41746-018-0040-6](https://doi.org/10.1038/s41746-018-0040-6)；[PMID 31304320](https://pubmed.ncbi.nlm.nih.gov/31304320/) | CC BY 4.0（npj；未逐頁核對） | 13 |
| MobileNetV2: Inverted Residuals and Linear Bottlenecks | Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC | 2018 | CVPR 2018；[arXiv:1801.04381](https://arxiv.org/abs/1801.04381) | arXiv；只引用架構 | 13 |
| Adam: A Method for Stochastic Optimization | Kingma DP, Ba J | 2015 | ICLR 2015；[arXiv:1412.6980](https://arxiv.org/abs/1412.6980) | arXiv；只引用公式 | 13 |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | Ioffe S, Szegedy C | 2015 | ICML 2015；[arXiv:1502.03167](https://arxiv.org/abs/1502.03167) | arXiv；只引用概念 | 13 |
| Attention Is All You Need | Vaswani A, et al. | 2017 | NeurIPS 2017；[arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | arXiv 論文；只引用概念 | 15 |
| Large language models encode clinical knowledge | Singhal K, et al. | 2023 | Nature 620:172–180；[DOI 10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2)；[PMID 37438534](https://pubmed.ncbi.nlm.nih.gov/37438534/) | 期刊版權；只引用結論 | 15 |
| Performance of ChatGPT on USMLE | Kung TH, et al. | 2023 | PLOS Digit Health 2:e0000198；[DOI 10.1371/journal.pdig.0000198](https://doi.org/10.1371/journal.pdig.0000198)；[PMID 36812645](https://pubmed.ncbi.nlm.nih.gov/36812645/) | CC BY 4.0（未逐頁確認） | 15 |
| Leakage and the reproducibility crisis in machine-learning-based science | Kapoor S, Narayanan A | 2023 | Patterns 4(9):100804；[DOI 10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)；[PMID 37720327](https://pubmed.ncbi.nlm.nih.gov/37720327/)（PMC10499856） | 開放取用（未逐頁核對授權細節）；數字取自摘要，洩漏類型取自全文 taxonomy 小節 | 14 |
| Automatic classification of heartbeats using ECG morphology and heartbeat interval features | de Chazal P, O'Dwyer M, Reilly RB | 2004 | IEEE Trans Biomed Eng 51(7):1196–1206；[DOI 10.1109/TBME.2004.827359](https://doi.org/10.1109/TBME.2004.827359)；[PMID 15248536](https://pubmed.ncbi.nlm.nih.gov/15248536/) | 期刊版權；數字取自 PubMed 摘要 | 14 |

### 資料集的原始文獻

| 標題 | 作者 | 年份 | 出處／識別碼 | 授權 | 章節 |
|---|---|---|---|---|---|
| Nuclear feature extraction for breast tumor diagnosis（WDBC） | Street WN, Wolberg WH, Mangasarian OL | 1993 | SPIE 1905；UCI DOI 10.24432/C5DW2B | 期刊／會議版權；只引用 | 01、04、05、06、09、10、11 |
| Survival analysis of heart failure patients: a case study | Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA | 2017 | PLoS One 12(7):e0181001；[DOI 10.1371/journal.pone.0181001](https://doi.org/10.1371/journal.pone.0181001) | CC BY 4.0（PLoS One 預設授權；未逐頁核對） | 02、08 |
| Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone | Chicco D, Jurman G | 2020 | BMC Med Inform Decis Mak 20:16；[DOI 10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5)；PMID 32013925 | CC BY 4.0（BMC 開放取用；未逐頁核對） | 02、07、08 |
| International application of a new probability algorithm for the diagnosis of coronary artery disease | Detrano R, et al. | 1989 | Am J Cardiol 64(5):304–310；[DOI 10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)；[PMID 2756873](https://pubmed.ncbi.nlm.nih.gov/2756873/) | 期刊版權；只引用 | 07、10 |
| Least Angle Regression（sklearn diabetes 資料集出處） | Efron B, Hastie T, Johnstone I, Tibshirani R | 2004 | Ann Stat 32(2):407–499 | 期刊版權；只引用 | 04、10 |
| MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification | Yang J, Shi R, Wei D, et al. | 2023 | Sci Data 10:41；[DOI 10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)；PMID 36658144。另須引用 Yang J, Shi R, Ni B. MedMNIST Classification Decathlon, IEEE ISBI 2021 | 論文開放取用；PneumoniaMNIST、BloodMNIST 為 CC BY 4.0 | 09、12、13 |
| Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning（PneumoniaMNIST 的原始影像來源） | Kermany DS, et al. | 2018 | Cell 172(5):1122–1131.e9；[DOI 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)；PMID 29474911 | 論文依期刊版權；原始影像資料集 CC BY 4.0（Mendeley Data） | 09、12 |
| A dataset of microscopic peripheral blood cell images for development of automatic recognition systems（BloodMNIST 的原始影像來源） | Acevedo A, Merino A, Alférez S, et al. | 2020 | Data Brief 30:105474；[DOI 10.1016/j.dib.2020.105474](https://doi.org/10.1016/j.dib.2020.105474)；PMID 32346559（PMC7182702） | 開放取用（未逐頁核對授權細節）；BloodMNIST 子集為 CC BY 4.0 | 12 |
| Dataset of breast ultrasound images | Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A | 2020 | Data Brief 28:104863；[DOI 10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)；[PMID 31867417](https://pubmed.ncbi.nlm.nih.gov/31867417/) | open access；資料集授權以原頁為準 | 13 |
| Letter to the Editor. Re: "Dataset of breast ultrasound images" | Pawłowska A, Karwat P, Żołek N | 2023 | Data Brief 48:109247；[DOI 10.1016/j.dib.2023.109247](https://doi.org/10.1016/j.dib.2023.109247)；[PMID 37383756](https://pubmed.ncbi.nlm.nih.gov/37383756/) | open access；只引用結論 | 13 |
| Gretel symptom_to_diagnosis（Hugging Face 資料集卡片；上游為 Kaggle Symptom2Disease） | Gretel.ai；上游 Barman NR | — | [HF dataset card](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) | Apache-2.0（HF 實查）；上游 Kaggle 為 CC0 | 15 |
| The impact of the MIT-BIH Arrhythmia Database | Moody GB, Mark RG | 2001 | IEEE Eng Med Biol Mag 20(3):45–50；[DOI 10.1109/51.932724](https://doi.org/10.1109/51.932724)；[PMID 11446209](https://pubmed.ncbi.nlm.nih.gov/11446209/) | 期刊版權；資料庫本身 ODC-By v1.0 | 14 |
| PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals | Goldberger AL, Amaral LAN, Glass L, et al. | 2000 | Circulation 101(23):e215–e220（PhysioNet 要求的標準引用） | 期刊版權 | 14 |
| PTB-XL, a large publicly available electrocardiography dataset | Wagner P, Strodthoff N, Bousseljot RD, et al. | 2020 | Sci Data 7:154；[DOI 10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)；[PMID 32451379](https://pubmed.ncbi.nlm.nih.gov/32451379/)（PMC7248071） | CC BY 4.0 | 14（只在文字介紹 `strat_fold`，未下載） |

## 三、資料集來源

各資料集的載入方式、樣本數、授權與已知問題，請見 **[資料集一覽](datasets.md)**，此處不重複。本站各章使用的資料集有：Breast Cancer Wisconsin（第 1、4、5、6、9、10、11 章）、Heart Failure Clinical Records（第 2、7、8 章）、Chronic Kidney Disease、Pima Indians Diabetes 與 WHO GHO API（第 3 章）、sklearn Diabetes（第 4、10 章）、Medical Abstracts TC Corpus（第 5 章）、Heart Disease Cleveland（第 7、10 章）、PneumoniaMNIST（第 9、12 章）、BloodMNIST（第 12 章）、BreastMNIST（第 13 章）、Gretel symptom_to_diagnosis（第 15 章）、CDC Diabetes Health Indicators（第 11 章）、MIT-BIH Arrhythmia Database（第 14 章，PhysioNet，ODC-By v1.0）。

## 四、互動元件、工具與其他

### 互動元件

| 標題 | 作者／機構 | 授權 | 用法 | 章節 |
|---|---|---|---|---|
| [TensorFlow Playground](https://playground.tensorflow.org)（[原始碼](https://github.com/tensorflow/playground)） | Smilkov D, Carter S（Google） | Apache-2.0 | 第 9 章 iframe 嵌入（附出處與開新分頁連結）；第 1 章只連結 | 01、09 |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/)（[原始碼](https://github.com/poloclub/cnn-explainer)） | Wang ZJ, Turko R, Shaikh O, et al.（Georgia Tech Polo Club of Data Science） | MIT（GitHub API，2026-10-06） | 第 12 章 iframe 嵌入（附出處與開新分頁連結） | 12 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Georgia Tech Polo Club | MIT | 第 15 章連結＋折疊 iframe | 15 |
| [MLU-Explain](https://mlu-explain.github.io/)：[Logistic Regression](https://mlu-explain.github.io/logistic-regression/)、[Decision Trees](https://mlu-explain.github.io/decision-tree/)、[Random Forest](https://mlu-explain.github.io/random-forest/) | Amazon MLU | 文章 CC BY-SA 4.0（aws-samples/aws-mlu-explain）；Logistic Regression 頁授權未逐頁驗證 | **只連結、未改作** | 04、07 |
| [setosa.io：Ordinary Least Squares Regression](https://setosa.io/ev/ordinary-least-squares-regression/)、[Principal Component Analysis Explained Visually](https://setosa.io/ev/principal-component-analysis/) | Victor Powell、Lewis Lehe | repo 為 MIT；頁面文字授權未聲明 | 只連結、未 iframe | 04、10 |
| [ConvNetJS 2D classification demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) | Andrej Karpathy | MIT（github.com/karpathy/convnetjs） | 只連結 | 06 |
| [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/) | Naftali Harris | 未聲明授權 | 只連結、未 iframe | 08 |

### 建置與前端工具

| 標題 | 作者／機構 | 授權 | 用途 | 章節 |
|---|---|---|---|---|
| [Plotly.js](https://github.com/plotly/plotly.js) 2.35.2 | Plotly | MIT | 自寫互動 demo 的繪圖（全站載入） | 04、05、06、10 |
| [Cubic 11 俐方體 11 號](https://github.com/ACh-K/Cubic-11) v1.500 | ACh-K（衍生自 JF Dot M+H 12 / M+ BITMAP FONTS） | SIL OFL-1.1（授權檔隨站附於 `assets/fonts/Cubic_11_OFL.txt`） | 站名、導覽列、標題、按鈕的像素字型（自架 woff2） | 全站 |
| [MkDocs](https://www.mkdocs.org/) 1.6.1 | MkDocs 社群 | BSD-2-Clause（pip 套件 metadata） | 站台建置 | 全站 |
| [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 9.7.7 | Martin Donath | MIT（pip 套件 metadata） | 站台主題 | 全站 |
| [KaTeX](https://katex.org/) 0.16.11 | KaTeX 社群 | MIT（未逐一驗證） | 數學式渲染（CDN） | 全站 |
| [Anaconda Distribution](https://www.anaconda.com/download) | Anaconda, Inc. | 商業網站；只連結 | 第 1 章環境安裝（選配） | 01 |
| [Keras Applications：MobileNetV2 ImageNet 預訓練權重](https://keras.io/api/applications/mobilenet/) | Keras team（權重轉自 tensorflow/models 的 checkpoint） | 程式碼 Apache-2.0；**權重授權官方未明列，以官方為準**（查證紀錄見 `research/refs/ch13.md`）；ImageNet 使用條款限非商業研究／教育；本站不重新散布權重 | notebook 執行時下載的預訓練骨幹 | 13 |

### 術語、法規與標準

| 標題 | 機構 | 授權 | 用途 | 章節 |
|---|---|---|---|---|
| [國家教育研究院樂詞網](https://terms.naer.edu.tw/) | 國家教育研究院 | 政府網站；只引用詞條 | 術語官方譯名（如「過度配適」） | 01 |
| [政府資料開放授權條款－第 1 版](https://data.gov.tw/license) | 數位發展部 | 條款本身；允許任何目的利用，需顯名聲明 | data.gov.tw 授權說明 | 03 |
| [個人資料保護法](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=I0050021)（第 6 條特種個資） | 全國法規資料庫 | 法規（公共領域） | 爬蟲倫理段落，僅概述 | 03 |
| [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309) | IETF | IETF Trust；只引用 | robots.txt 標準 | 03 |
| [WHO robots.txt](https://www.who.int/robots.txt) | WHO | 網站檔案；僅讀取解析、不轉載 | `robotparser` 示範 | 03 |
| [Artificial Intelligence-Enabled Medical Devices](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices) | U.S. FDA | 美國政府網站 | AI 醫材清單筆數（1,614 筆，2026-10-06 擷取） | 13 |
| [人工智慧/機器學習技術之醫療器材軟體查驗登記技術指引](https://www.fda.gov.tw/tc/siteListContent.aspx?id=34961)（2020-09-11 公告） | 衛生福利部食品藥物管理署 | 政府網站；只引用名稱與日期 | 台灣 AI 醫材法規 | 13 |

## 本站授權

- **文字內容**：以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hant) 授權，轉載或改作時請標註出處。
- **程式碼**（含 notebook 與自寫互動 demo）：以 [MIT 授權](https://opensource.org/license/mit)釋出。
- **第三方內容**（資料集、論文、教材、互動元件）依各自授權，本站未取得其再授權；資料集皆於執行時由原站下載，不隨本站散布。
- **MLU-Explain** 為 CC BY-SA 4.0，本站**僅連結、未改作**，因此沒有把 CC BY-SA 的相同方式分享條款帶入本站內容。
- 本站圖表皆為自產示意圖或由公開資料計算，不含第三方圖片。
