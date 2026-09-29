# 醫學生的機器學習入門

給醫學與生醫相關科系學生的 Machine Learning 入門教材：直覺先行、圖解為主，每章附醫學公開資料集與可在 Google Colab 執行的 notebook。

網站：<https://med-study-rpg.com/ml/>

## 章節

1. 機器學習簡介與環境安裝
2. Python 四大套件實務：NumPy、Pandas、Matplotlib、SciPy
3. 大數據收集與資料前處理實戰
4. 迴歸分析：線性、多項式與邏輯迴歸
5. 簡單貝氏分類
6. 支援向量機（SVM）
7. 決策樹與隨機森林
8. K-平均分群
9. 人工神經網路基礎
10. 資料降維：反向淘汰法、卡方檢定法、PCA
11. 各種模型使用時機比較與效能提升策略

Notebook 位於 [`docs/notebooks/`](docs/notebooks/)，以 Colab 預裝版本（scikit-learn 1.6 / pandas 2.2）為相容底線，並在 scikit-learn 1.9 / pandas 3.0 測試過。

## 本機建置

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/mkdocs serve
```

部署步驟見 [`deploy/DEPLOY.md`](deploy/DEPLOY.md)。

## 授權

- 文字內容：[CC BY 4.0](LICENSE-CONTENT)
- 程式碼（notebooks、scripts、demo JS）：[MIT](LICENSE)
- 第三方資料集、字型與元件依各自授權，詳見網站「附錄：參考資源與授權」。

本站內容僅供教學，不構成醫療建議。
