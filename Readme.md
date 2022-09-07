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

![Image](https://github.com/wuyiulin/OptimalKmeans/blob/main/img/GSS.jpeg)

![Image](https://github.com/wuyiulin/OptimalKmeans/blob/main/img/sil.jpeg)



## Acknowledgement

Gap statistic source code from:

https://medium.com/@pahome.chen/clustering%E6%B1%BA%E5%AE%9A%E5%88%86%E7%BE%A4%E6%95%B8%E7%9A%84%E6%96%B9%E6%B3%95-abedc1d81ccb


## Contact

Further information please contact me.

wuyiulin@gmail.com