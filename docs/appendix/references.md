# 參考資料與授權

本頁彙整各章實際引用的來源，同一來源在多章出現時合併為一筆，「章節」欄列出用到的章。授權欄記的是撰寫時實際查到的內容；標「未驗證」者代表沒有逐頁核對，重新散布前請自行查證。

!!! note "引用原則"
    - 教材、課程與書籍多數**只連結、未改作**；文字為 CC BY-NC-ND 者（如 VanderPlas）更不得改作。
    - 論文只引用其結論或摘要層級的事實，並在正文標明年份與範圍；請以原文為準。
    - 圖表與互動 demo 皆為本站自產，沒有使用第三方圖片。

## 一、教材與課程

| 標題 | 作者／機構 | 年份 | 授權 | 章節 |
|---|---|---|---|---|
| [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)（含 [Numerical data](https://developers.google.com/machine-learning/crash-course/numerical-data)、[Neural networks](https://developers.google.com/machine-learning/crash-course/neural-networks)、[ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)、[Imbalanced datasets](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)） | Google | 持續更新 | 內容 CC BY 4.0、程式 Apache 2.0（依研究筆記；未逐頁確認）；只連結、只轉述概念 | 01、03、04、09、11 |
| [ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners)（含 [2-Regression](https://github.com/microsoft/ML-For-Beginners/tree/main/2-Regression)、[5-Clustering](https://github.com/microsoft/ML-For-Beginners/tree/main/5-Clustering)） | Microsoft | 持續更新 | MIT；教法參考、未複製內容 | 01、04、08 |
| [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)（第 05.05 Naive Bayes、05.06 Linear Regression、05.07 SVM、05.08 Random Forests、05.09 PCA、05.11 k-Means 各節） | Jake VanderPlas | 2016 | 文字 CC BY-NC-ND、程式碼 MIT；**只連結、未改作** | 02、04、05、06、07、08、10 |
| [Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html) | Zhang A, Lipton ZC, Li M, Smola AJ | 持續更新 | 文字 CC BY-SA 4.0、程式碼 modified MIT；教學順序參考、未複製內文 | 09 |
| [李宏毅 Machine Learning 2021 Spring](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) | 李宏毅（國立臺灣大學） | 2021 | 未驗證；只連結 | 09 |
| [StatQuest: Naive Bayes, Clearly Explained](https://www.youtube.com/results?search_query=statquest+naive+bayes+clearly+explained)（YouTube 搜尋連結，未確認單一影片網址） | Josh Starmer | — | 版權屬作者，未查證；只連結 | 05 |
| [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | NumPy 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) | pandas 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) | pandas 社群 | 持續更新 | BSD-3-Clause（未逐頁核對）；第 2 章陷阱 1 的官方依據 | 02 |
| [Matplotlib Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html) | Matplotlib 社群 | 持續更新 | PSF-based license（未逐頁核對） | 02 |
| [SciPy statistics tutorial](https://docs.scipy.org/doc/scipy/tutorial/stats.html) | SciPy 社群 | 持續更新 | BSD-3-Clause（未逐頁核對） | 02 |
| [Keras 3 文件](https://keras.io/getting_started/) | Keras 團隊 | 持續更新 | Apache-2.0 | 09 |

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

### 資料集的原始文獻

| 標題 | 作者 | 年份 | 出處／識別碼 | 授權 | 章節 |
|---|---|---|---|---|---|
| Nuclear feature extraction for breast tumor diagnosis（WDBC） | Street WN, Wolberg WH, Mangasarian OL | 1993 | SPIE 1905；UCI DOI 10.24432/C5DW2B | 期刊／會議版權；只引用 | 01、04、05、06、09、10、11 |
| Survival analysis of heart failure patients: a case study | Ahmad T, Munir A, Bhatti SH, Aftab M, Raza MA | 2017 | PLoS One 12(7):e0181001；[DOI 10.1371/journal.pone.0181001](https://doi.org/10.1371/journal.pone.0181001) | CC BY 4.0（PLoS One 預設授權；未逐頁核對） | 02、08 |
| Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone | Chicco D, Jurman G | 2020 | BMC Med Inform Decis Mak 20:16；[DOI 10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5)；PMID 32013925 | CC BY 4.0（BMC 開放取用；未逐頁核對） | 02、07、08 |
| International application of a new probability algorithm for the diagnosis of coronary artery disease | Detrano R, et al. | 1989 | Am J Cardiol 64(5):304–310；[DOI 10.1016/0002-9149(89)90524-9](https://doi.org/10.1016/0002-9149(89)90524-9)；[PMID 2756873](https://pubmed.ncbi.nlm.nih.gov/2756873/) | 期刊版權；只引用 | 07、10 |
| Least Angle Regression（sklearn diabetes 資料集出處） | Efron B, Hastie T, Johnstone I, Tibshirani R | 2004 | Ann Stat 32(2):407–499 | 期刊版權；只引用 | 04、10 |
| MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification | Yang J, Shi R, Wei D, et al. | 2023 | Sci Data 10:41；[DOI 10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)；PMID 36658144。另須引用 Yang J, Shi R, Ni B. MedMNIST Classification Decathlon, IEEE ISBI 2021 | 論文開放取用；PneumoniaMNIST 為 CC BY 4.0 | 09 |
| Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning（PneumoniaMNIST 的原始影像來源） | Kermany DS, et al. | 2018 | Cell 172(5):1122–1131.e9；[DOI 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)；PMID 29474911 | 論文依期刊版權；原始影像資料集 CC BY 4.0（Mendeley Data） | 09 |

## 三、資料集來源

各資料集的載入方式、樣本數、授權與已知問題，請見 **[資料集一覽](datasets.md)**，此處不重複。本站各章使用的資料集有：Breast Cancer Wisconsin（第 1、4、5、6、9、10、11 章）、Heart Failure Clinical Records（第 2、7、8 章）、Chronic Kidney Disease、Pima Indians Diabetes 與 WHO GHO API（第 3 章）、sklearn Diabetes（第 4、10 章）、Medical Abstracts TC Corpus（第 5 章）、Heart Disease Cleveland（第 7、10 章）、PneumoniaMNIST（第 9 章）、CDC Diabetes Health Indicators（第 11 章）。

## 四、互動元件、工具與其他

### 互動元件

| 標題 | 作者／機構 | 授權 | 用法 | 章節 |
|---|---|---|---|---|
| [TensorFlow Playground](https://playground.tensorflow.org)（[原始碼](https://github.com/tensorflow/playground)） | Smilkov D, Carter S（Google） | Apache-2.0 | 第 9 章 iframe 嵌入（附出處與開新分頁連結）；第 1 章只連結 | 01、09 |
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

### 術語、法規與標準

| 標題 | 機構 | 授權 | 用途 | 章節 |
|---|---|---|---|---|
| [國家教育研究院樂詞網](https://terms.naer.edu.tw/) | 國家教育研究院 | 政府網站；只引用詞條 | 術語官方譯名（如「過度配適」） | 01 |
| [政府資料開放授權條款－第 1 版](https://data.gov.tw/license) | 數位發展部 | 條款本身；允許任何目的利用，需顯名聲明 | data.gov.tw 授權說明 | 03 |
| [個人資料保護法](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=I0050021)（第 6 條特種個資） | 全國法規資料庫 | 法規（公共領域） | 爬蟲倫理段落，僅概述 | 03 |
| [RFC 9309: Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309) | IETF | IETF Trust；只引用 | robots.txt 標準 | 03 |
| [WHO robots.txt](https://www.who.int/robots.txt) | WHO | 網站檔案；僅讀取解析、不轉載 | `robotparser` 示範 | 03 |

## 本站授權

- **文字內容**：以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.zh-hant) 授權，轉載或改作時請標註出處。
- **程式碼**（含 notebook 與自寫互動 demo）：以 [MIT 授權](https://opensource.org/license/mit)釋出。
- **第三方內容**（資料集、論文、教材、互動元件）依各自授權，本站未取得其再授權；資料集皆於執行時由原站下載，不隨本站散布。
- **MLU-Explain** 為 CC BY-SA 4.0，本站**僅連結、未改作**，因此沒有把 CC BY-SA 的相同方式分享條款帶入本站內容。
- 本站圖表皆為自產示意圖或由公開資料計算，不含第三方圖片。
