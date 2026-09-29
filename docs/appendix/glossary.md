# 中英術語對照

本表以台灣慣用詞為準，並在教材內統一使用。少數詞的官方譯名（國家教育研究院樂詞網）與實務用法不同，會在白話解釋裡註明，方便你搜尋。

## 基礎概念

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 人工智慧 | artificial intelligence（AI） | 讓機器表現出類似人類智慧行為的技術總稱，機器學習是其中一支。 |
| 機器學習 | machine learning（ML） | 讓電腦從資料中找出規律，而不是由人逐條寫規則。 |
| 深度學習 | deep learning | 使用很多層神經網路的機器學習，擅長影像、語音與文字。 |
| 監督式學習 | supervised learning | 資料附有標準答案，模型學著預測答案。 |
| 非監督式學習 | unsupervised learning | 資料沒有答案，讓模型自己找出結構，例如分群。 |
| 強化學習 | reinforcement learning | 模型透過試誤與獎勵，學會一連串決策。 |
| 分類 | classification | 預測類別，例如良性或惡性。 |
| 迴歸 | regression | 預測連續數值，例如血壓或住院費用。 |
| 分群 | clustering | 把相似的樣本歸成一堆；樂詞網作「聚類」。 |
| 特徵 | feature | 輸入模型的變數，等同統計裡的自變數。 |
| 標籤 | label | 資料的標準答案，例如「惡性」。 |
| 目標變數 | target / response | 想預測的變數，醫學統計常稱依變數或結果變數。 |
| 樣本 | sample | 資料中的一筆紀錄，例如一位病人。 |
| 資料集 | dataset | 一組用來訓練或評估模型的資料。 |
| 演算法 | algorithm | 解決問題的一套明確步驟。 |
| 模型 | model | 從資料學得、可拿來做預測的數學函式。 |
| 參數 | parameter | 模型在訓練中學到的數字，例如迴歸係數。 |
| 超參數 | hyperparameter | 訓練前由人設定的旋鈕，例如樹的深度。 |
| 預測 | prediction | 模型對新樣本給出的輸出。 |
| 推論 | inference | 在統計是「解釋與推估母體」，在深度學習常指「用訓練好的模型做預測」，注意語境。 |
| 一般化能力 | generalization | 模型面對沒見過的新資料仍能表現良好的能力，也常說泛化。 |
| 過擬合 | overfitting | 模型把訓練資料的雜訊也背起來，遇到新資料表現變差；又稱過度配適（樂詞網官方譯名）。 |
| 配適不足 | underfitting | 模型太簡單，連訓練資料的規律都沒抓到。 |
| 偏差 | bias | 模型太簡單而造成的系統性誤差；流行病學的研究偏誤是另一回事。 |
| 變異數 | variance | 模型對訓練資料的微小變動過度敏感的程度。 |
| 偏差—變異數取捨 | bias-variance trade-off | 模型越複雜偏差越小、變異數越大，需要找平衡點。 |
| 雜訊 | noise | 資料中與規律無關的隨機波動。 |

## 資料處理

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 資料前處理 | data preprocessing | 建模之前的清理、轉換與整理工作。 |
| 遺漏值 | missing value | 沒有被記錄到的欄位值，又稱缺值、缺失值。 |
| 插補 | imputation | 用合理的值（如平均數）填補遺漏值。 |
| 離群值 | outlier | 明顯偏離大多數資料的極端值。 |
| 標準化 | standardization | 把變數轉成平均 0、標準差 1（z 分數轉換）。 |
| 正規化 | normalization | 把變數縮放到固定範圍，常見是 0 到 1（Min-Max 縮放）。 |
| z 分數 | z-score | 一個值距離平均幾個標準差。 |
| 獨熱編碼 | one-hot encoding | 把類別變數拆成多個 0/1 欄位，每類一欄。 |
| 類別編碼 | categorical encoding | 把文字類別轉成模型能讀的數字。 |
| 特徵工程 | feature engineering | 設計或轉換變數，讓模型更容易學到規律。 |
| 特徵縮放 | feature scaling | 讓各特徵處於相近尺度的處理，標準化與正規化都屬於此類。 |
| 訓練集 | training set | 用來讓模型學習的資料。 |
| 驗證集 | validation set | 用來調整超參數、比較模型的資料；樂詞網作「確認資料集」。 |
| 測試集 | test set | 最後一次檢驗模型表現、之前完全不碰的資料。 |
| 資料洩漏 | data leakage | 測試集或結果的資訊不當進入訓練，使成績虛高。 |
| 抽樣 | sampling | 從資料中選出一部分樣本。 |
| 分層抽樣 | stratified sampling | 抽樣時讓各類別比例與原資料一致。 |
| 重抽樣 | resampling | 對資料重複抽樣，例如自助法或過採樣、欠採樣。 |
| 類別不平衡 | class imbalance | 某一類樣本遠多於另一類，例如罕見疾病。 |
| 管線 | pipeline | 把前處理與模型串成一條處理流程，避免資料洩漏。 |
| 資料框 | DataFrame | pandas 中像試算表的二維資料表格。 |
| 陣列 | array | NumPy 中存放同型別數字的多維容器。 |
| 張量 | tensor | 多維陣列，神經網路框架的基本資料單位。 |
| 矩陣 | matrix | 二維的數字表格。 |
| 向量 | vector | 一維的一串數字，例如一位病人的所有特徵。 |
| 降維 | dimensionality reduction | 用較少的變數表達原本很多變數的資訊。 |
| 維度詛咒 | curse of dimensionality | 特徵越多，資料在空間中越稀疏，模型越難學。 |
| 共線性 | collinearity | 多個特徵之間高度相關，使迴歸係數不穩定。 |
| 特徵選取 | feature selection | 從既有特徵中挑出有用的子集。 |
| 特徵萃取 | feature extraction | 由原有特徵組合出新特徵，例如主成分分析。 |

## 模型

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 線性迴歸 | linear regression | 用直線（或超平面）描述特徵與連續結果的關係。 |
| 邏輯迴歸 | logistic regression | 用 sigmoid 把線性組合轉成機率的分類模型，又稱羅吉斯迴歸。 |
| 多項式迴歸 | polynomial regression | 加入特徵的平方、立方項，讓迴歸線可以彎曲。 |
| 逐步迴歸 | stepwise regression | 依統計準則逐步加入或移除變數的方法。 |
| 係數 | coefficient | 迴歸式中每個特徵前面的權重。 |
| 殘差 | residual | 觀測值減去預測值。 |
| 貝氏定理 | Bayes' theorem | 由檢驗前機率與檢驗特性推算檢驗後機率。 |
| 簡單貝氏分類器 | naive Bayes classifier | 假設特徵彼此條件獨立的貝氏分類器，也譯單純貝氏。 |
| 先驗機率 | prior probability | 看到資料前對某類別的機率估計，如盛行率；又稱事前機率。 |
| 概似 | likelihood | 在某類別下觀察到目前資料的可能性。 |
| 後驗機率 | posterior probability | 看到資料後更新的類別機率；又稱事後機率。 |
| 支援向量機 | support vector machine（SVM） | 找出間隔最大的分界面的分類器。 |
| 支援向量 | support vector | 最靠近分界面、決定邊界位置的那些樣本。 |
| 間隔 | margin | 分界面到最近樣本的距離。 |
| 核函數 | kernel function | 不用真的升到高維，就能算出高維空間相似度的函式。 |
| 決策樹 | decision tree | 用一連串是非題把樣本分到不同葉節點的模型。 |
| 節點 | node | 決策樹中的一個判斷點。 |
| 剪枝 | pruning | 移除決策樹中不必要的分支，以減少過擬合。 |
| 吉尼不純度 | Gini impurity | 衡量一個節點裡類別混雜程度的指標。 |
| 熵 | entropy | 衡量不確定性或混亂程度的指標。 |
| 隨機森林 | random forest | 由許多棵隨機化決策樹投票組成的模型。 |
| 集成學習 | ensemble learning | 結合多個模型的預測以提升表現。 |
| 自助聚合法 | bagging | 對資料重複抽樣訓練多個模型再平均，是隨機森林的基礎。 |
| 提升法 | boosting | 依序訓練模型，讓後面的模型專攻前面做錯的樣本。 |
| K-平均分群 | K-means clustering | 反覆指定群中心、分配樣本的分群方法；又稱 K 平均數分群，本站多直接寫 K-Means。 |
| 群中心 | centroid | 一群樣本的平均位置。 |
| 主成分分析 | principal component analysis（PCA） | 找出變異最大的方向，用較少的軸表達資料。 |
| 主成分 | principal component | PCA 找到的新座標軸，是原特徵的線性組合。 |
| 特徵重要性 | feature importance | 衡量各特徵對預測貢獻的分數。 |
| 基準模型 | baseline model | 拿來當比較底線的簡單模型。 |

## 評估指標

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 混淆矩陣 | confusion matrix | 把預測與實際結果交叉成 2×2 表，等同診斷檢驗的四格表。 |
| 真陽性 | true positive（TP） | 實際有病、預測有病。 |
| 偽陽性 | false positive（FP） | 實際沒病、預測有病。 |
| 真陰性 | true negative（TN） | 實際沒病、預測沒病。 |
| 偽陰性 | false negative（FN） | 實際有病、預測沒病，臨床上通常代價最高。 |
| 準確率 | accuracy | 預測正確的比例，類別不平衡時容易誤導。 |
| 敏感度 | sensitivity | 有病的人中被抓出來的比例，等於召回率。 |
| 特異度 | specificity | 沒病的人中被正確排除的比例。 |
| 召回率 | recall | 實際為陽性者中被找出的比例，等於敏感度。 |
| 精確率 | precision | 預測為陽性者中真的陽性的比例，等於陽性預測值。 |
| 陽性預測值 | positive predictive value（PPV） | 檢驗陽性時真的有病的機率，受盛行率影響。 |
| 陰性預測值 | negative predictive value（NPV） | 檢驗陰性時真的沒病的機率。 |
| F1 分數 | F1 score | 精確率與召回率的調和平均。 |
| ROC 曲線 | ROC curve | 畫出各分類閾值下敏感度對（1 − 特異度）的曲線。 |
| 曲線下面積 | area under the curve（AUC） | ROC 曲線下的面積，醫學文獻常稱 C 統計量。 |
| 精確率—召回率曲線 | precision-recall curve | 各閾值下精確率與召回率的關係，適合不平衡資料。 |
| 分類閾值 | classification threshold | 機率超過多少就判為陽性的切點。 |
| 均方誤差 | mean squared error（MSE） | 預測誤差平方後的平均。 |
| 決定係數 | coefficient of determination（R²） | 模型解釋了多少比例的結果變異。 |
| 校準 | calibration | 模型給的機率是否與實際發生比例一致。 |
| 交叉驗證 | cross-validation | 把資料輪流切成訓練與驗證，取平均表現；樂詞網作「交叉確認」。 |
| 網格搜尋 | grid search | 把超參數的候選組合全部試一遍。 |
| 外部驗證 | external validation | 用其他機構或時期的資料檢驗模型。 |
| 可解釋性 | interpretability | 人能否理解模型為何做出這個預測。 |

## 神經網路

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 類神經網路 | artificial neural network（ANN） | 由許多簡單運算單元連成的模型，簡稱神經網路。 |
| 神經元 | neuron | 對輸入做加權總和再套用活化函數的運算單元。 |
| 感知器 | perceptron | 最簡單的單層神經網路。 |
| 權重 | weight | 每條連線的重要性係數。 |
| 偏置 | bias term | 加在加權總和上的常數項，不要與統計偏誤混淆。 |
| 層 | layer | 一組同時運算的神經元。 |
| 隱藏層 | hidden layer | 位於輸入與輸出之間的層。 |
| 活化函數 | activation function | 決定神經元輸出的非線性函式；又稱激活函數、啟動函數，樂詞網作「啟動功能」。 |
| 損失函數 | loss function | 衡量預測與答案差多少的函數。 |
| 梯度下降 | gradient descent | 沿著讓損失下降最快的方向逐步調整參數。 |
| 學習率 | learning rate | 每一步調整參數的幅度，樂詞網作「學習速度」。 |
| 反向傳播 | backpropagation | 由輸出往回計算各權重梯度的方法。 |
| 前向傳播 | forward propagation | 輸入依序經過各層算出預測的過程。 |
| 訓練週期 | epoch | 完整看過一次全部訓練資料。 |
| 批次 | batch | 每次更新參數所使用的一小批樣本。 |
| 迭代 | iteration | 一次參數更新。 |
| 正則化 | regularization | 限制模型複雜度以降低過擬合的技巧。 |
| Dropout | dropout | 訓練時隨機關閉部分神經元以降低過擬合；又稱丟棄法，本站直接用英文 Dropout。 |
| 早停 | early stopping | 驗證表現不再進步就停止訓練；又稱提早停止。 |
| 卷積神經網路 | convolutional neural network（CNN） | 擅長處理影像、以卷積層擷取局部特徵的網路。 |

## 程式與工具

| 中文 | English | 一句話白話解釋 |
|---|---|---|
| 套件 | package / library | 別人寫好、可直接引用的一組程式功能。 |
| 模組 | module | 套件中的一個程式檔或功能單元。 |
| 函式 | function | 把一段程式包起來、可重複呼叫的單位。 |
| 變數 | variable | 程式中用來存放值的名稱。 |
| 預設值 | default | 沒有另外指定時程式採用的值。 |
| 最佳化 | optimization | 找出讓目標函數最小或最大的參數。 |
| 目標函數 | objective function | 最佳化要最小化或最大化的函數。 |
| 迴圈 | loop | 重複執行一段程式。 |
| 檔案 | file | 儲存在電腦裡的資料單位。 |
| 資料庫 | database | 有結構地儲存大量資料的系統。 |
| 記憶體 | memory | 電腦暫時存放運算中資料的空間。 |
| 亂數種子 | random seed | 固定隨機過程起點，讓結果可重現。 |
| 可重現性 | reproducibility | 別人用相同資料與程式能得到相同結果。 |
| 筆記本 | notebook | 把程式、輸出與說明放在一起的互動式文件（Jupyter Notebook）。 |
| 執行環境 | runtime / environment | 程式執行時所依賴的軟體與套件組合。 |
| 應用程式介面 | application programming interface（API） | 讓程式向另一個服務要資料或功能的約定。 |
| 網頁爬蟲 | web scraping | 用程式自動抓取網頁內容。 |
| 巨量資料 | big data | 資料量、速度或種類大到傳統工具難以處理的資料，用詞需審慎。 |
