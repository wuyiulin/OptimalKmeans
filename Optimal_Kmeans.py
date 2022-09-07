from ast import parse
from unicodedata import name
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from method import gap,sil
import argparse
import pdb


def Statistics(processed_data):
    count = 1
    for D in processed_data:
        print("Data_%d mean: "%count , np.mean(D))
        print("Data_%d median: "%count,  np.median(D))
        print("Data_%d standard deviation: "%count,  np.std(D))

        count+=1


def parse_args():
    parser = argparse.ArgumentParser(prog='gKmeans.py', description='Ben100db') 
    parser.add_argument('--method', '-m', default='gap', type=str, required=False,  help='chose method for best K')

    return parser.parse_args()

def chooseK (data, method, initC=10, endC=40):
    
    ks = range(initC,endC+1)
    try:
        if (method == 'gap'):
            __, __, GGS = gap(data, None, 20, ks)
            Best_K = np.argmax(GGS) + initC + 1
            return Best_K
        elif (method== 'sil'):
            sil_score = sil(data, ks)
            Best_K = np.argmax(sil_score) + initC + 1
            return Best_K
        else:
            raise
    except Exception:
        print('GGE19')


def main():

    data=np.loadtxt("your_data.csv",dtype=np.int32,delimiter=',')
    names = globals()
    Best_K = chooseK(data, args.method)
    print("i found BestK: ", Best_K)
    pca = PCA(n_components=2)
    #data = pca.fit_transform(data)
    cols = len(data[0])

    cluster_range = range(Best_K,Best_K+1) # 如果想要手動調整分群數，可以從這裡改變。
                                           # if you wnat to change cluster range by yourself, check here. 

    for n_clusters in cluster_range:
        start_time = time.time()

        print("Now start cluster-%s" %n_clusters)
        kmeans = KMeans(n_clusters)
        kmeans.fit(data)

        print('Kmeans Done!')
        globals()['cluster_label%s' %n_clusters] = kmeans.predict(data)+1

        for n in range(1,n_clusters+1):
            globals()['nd%s' %n_clusters+'_'+'%s' %n] = np.empty(cols)
            
        for k in cluster_range:
            cluster_label = names.get('cluster_label'+str(k))
            processing_data = names.get('nd'+str(n_clusters)+'_'+str(k))
            for i in cluster_range:
                for j in range(1,n_clusters+1):
                    if (cluster_label[i] == j):
                        processing_data =np.vstack([processing_data, data[i]])
                        globals()['nd%s' %n_clusters+'_'+'%s' %j] = processing_data
            __, counts = np.unique(names.get('cluster_label'+str(k)), return_counts=True)
            counts_sum = sum(counts)


        for i in range(1,n_clusters+1):
            print("\nStart print nd"+str(n_clusters)+"_"+str(i))
            processed_data = names.get('nd'+str(n_clusters)+'_'+str(i))
            processed_data = np.asarray(processed_data, dtype = int)
            processed_data = np.transpose(processed_data)
            Statistics(processed_data)
            print("local_cls_nums : " + str(counts[i-1]))
            print("local_cls_ratio : " + str(counts[i-1]/counts_sum))


        end_time = time.time()
        print("使用了 "+str((end_time-start_time)/60)+" 分鐘")

        # 繪圖區
        # graph area
        colors = names.get('cluster_label'+str(n_clusters))
        data = pca.fit_transform(data)
        plt.scatter(data[:,0], data[:,1], c=colors)
        plt.show()
        plt.pause(0)
        if (plt.waitforbuttonpress=='q'):
            continue
    
    
        print('Done')

if __name__ == '__main__':
    args = parse_args()
    main()