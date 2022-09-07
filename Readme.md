# 自動最佳化 Kmeans  (Self‑optimizing Kmeans)

## 介紹 Introdution

K-means 利用距離公式（本篇採用歐式距離）將 m 維 的 n個 資料點分群。

這裡提供兩種最佳化分群 K 值的方法（GAP Statistic, Silhouette Coefficient），

自動將最佳 K 值帶入程式，提供最佳分群選擇。

K-means use distantce formula convert m dim, n data point clustered.

We offer two optimal method to search K (cluster value),

auto input Best-K to Kmeans algorithm, and show best cluster result.


## 用法 Usage

    python Optimal_Kmeans.py <optional> -m (gap or sil)

### 輸出入範例 input and output example

### 輸入 input

一個 N*M維矩陣的 CSV 檔案。

A N*M dim martrix in CSV file.

### 輸出 output

1.cluster each dim mean

2.cluster each dim median

3.cluster each dim standard deviation

4.point of this cluster numbers

5.point of this cluster ratio


## 最佳 K 值選擇 Best-K chosen

[Image]

[Image]

