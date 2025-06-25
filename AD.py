import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
#from TSB_UAD.utils.visualisation import plotFig
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve  # per best_threshold
from sklearn.neighbors import NearestNeighbors    # per curve k dist
from pyod.models.iforest import IForest # tsb uad ha solo pochi modelli
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
import seaborn as sns
from pathlib import Path

import os
#os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import global_var


##################
# PRE-PROCESSING #
##################


def read_dataset(filepath, full_length):
    if "Turbine" in filepath:
        df = pd.read_parquet(filepath).dropna().to_numpy()

        if full_length or len(df) < global_var.limit_length:
            max_length = len(df)
        else:
            max_length = global_var.limit_length

        if global_var.single_column:
            data = df[:max_length, global_var.idx_column].astype(float)
        else:
            data = df[:max_length, 0:global_var.idx_column+1].astype(float)

        return data, None, max_length
    else:
        df = pd.read_csv(filepath, header=None).dropna().to_numpy()

        if full_length or len(df) < global_var.limit_length:
            max_length = len(df)
        else:
            max_length = global_var.limit_length

        data = df[:max_length,0].astype(float)
        if df.ndim > 1:
            label = df[:max_length,1].astype(int)
        else:
            label = None

        if np.sum(label) == 0:
            print("⚠️ Nessuna anomalia presente nei dati etichettati (label).")
            print("Le metriche non possono essere calcolate in modo significativo.")
            label = None

        return data, label, max_length


def create_windows(data, label, slide):

    if data.ndim > 1:
        len_Win = global_var.len_Win_multi
        data = StandardScaler().fit_transform(data)
        step = data.shape[1] if slide else data.shape[1]*len_Win
        data = data.flatten()
    else:
        len_Win = find_length(data)
        data = StandardScaler().fit_transform(data.reshape(-1, 1)).flatten()
        step = 1 if slide else len_Win

    W_data = Window(window=len_Win).convert(data)[::step].to_numpy()

    if not slide and data.ndim==1 and label is not None:
        W_label = Window(window=len_Win).convert(label)[::step].to_numpy()
        label = W_label.any(axis=1).astype(int)

    return W_data, label, len_Win


################
# CREA MODELLO #
################


def create_model(W_data, modelName):
    if modelName == 'IForest':
        clf = IForest(random_state=42, n_jobs=1)
    elif modelName.startswith("KNN"):
        clf = KNN(n_neighbors=int(modelName.split("_")[-1]), metric=global_var.metric_kNN, n_jobs=1)
    elif modelName == 'AutoEncoder':
        clf = AutoEncoder()
    
    elif modelName == "Best_kNN":
        k = best_k(W_data, metric=global_var.metric_kNN)
        print(k)
        clf = KNN(n_neighbors=k, metric=global_var.metric_kNN)   # options: 'euclidean', 'l1', 'l2', 'manhattan', 'mahalanobis', 'minkowski'


    start_time = time.time()
    clf.fit(W_data)
    score = clf.decision_scores_
    execution_time = time.time() - start_time
    
    return score, execution_time


def best_k(X, metric='euclidean'):
    # tecnica della Curve k-dist
    k_scores = {}
    
    for k in range(1, 20):
        vicini = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        distanze, _ = vicini.kneighbors(X)
        k_distanze = np.sort(distanze[:, k])
        
        d2 = np.abs(np.diff(k_distanze, n=2))
        k_scores[k] = d2.max() if len(d2) > 0 else 0.0
    
    best_k = max(k_scores, key=k_scores.get)
    return best_k


###################
# POST-PROCESSING #
###################


def post_processing_score(score, len_Win, slide):
    score = MinMaxScaler((0,1)).fit_transform(score.reshape(-1,1)).ravel()
    if slide == True:
        score = np.array([score[0]]*math.ceil((len_Win-1)/2) + list(score) + [score[-1]]*((len_Win-1)//2))
    
    return score


###########
# METRICS #
###########


def iqr_threshold(anomaly_scores, coef=1.5):
    q1 = np.percentile(anomaly_scores, 25)
    q3 = np.percentile(anomaly_scores, 75)
    iqr = q3 - q1
    iqr_threshold = q3 + coef * iqr

    return iqr_threshold


def best_thresh(label, scores):
    precision, recall, thresholds = precision_recall_curve(label, scores)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_thresh = thresholds[best_idx]

    return best_thresh


def calculate_metrics(labels, score):
    thresholdlist = threshold_func(score)
    auc = roc_auc_score(labels, score)

    thresholds_to_use = thresholdlist if global_var.allthresholds else [thresholdlist[global_var.idx_threshold]]
    metrics = []

    for threshold in thresholds_to_use:
        predictions = (score >= threshold).astype(int)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        metrics.append({
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'threshold': threshold
        })

    return metrics



###########
# GRAFICO #
###########

def ensure_save_path(filepath, slide, name):
    folder = f"risultati/plotFig/{filepath}/{'Slide' if slide else 'NoSlide'}"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{name}.png")


def create_plot(W_data, data, score, label, slide, name, modelName, filepath):
    if label is not None:
        all_metrics = calculate_metrics(label, score)
        selected = all_metrics[global_var.idx_threshold]
        auc = selected['auc']
        prec = selected['precision']
        rec = selected['recall']
        f1 = selected['f1']
        threshold = selected['threshold']
    else:
        threshold = threshold_func(score)[global_var.idx_threshold]

    plot_data = W_data.mean(axis=1) if not slide else data[:,0] if data.ndim > 1 else data
    plotFig_new(plot_data, label, score, fileName=name, threshold=threshold)

    fig = plt.gcf()

    if label is not None:
        metric_text = f"AUC: {auc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\nThreshold: {threshold:.4f}"
        #fig.suptitle(f"{metric_text}", fontsize=14)
    else:
        metric_text = f"Threshold: {threshold:.4f}"

    if global_var.show_metrics:
        fig.text(
            0.99, 0.94,
            metric_text,
            fontsize=12,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5),
            transform=fig.transFigure
        )
        
    if global_var.save_plot:
        save_path = ensure_save_path(filepath, slide, name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    if global_var.show_plot:   
        plt.show()
    plt.close(fig)


def plotFig_new(data, label, score, fileName, threshold=None, plotRange=None):
    max_length = len(data)
    if global_var.full_plot:
        plotRange = [0, max_length]
    else:
        plotRange = global_var.plotRange

    has_labels = label is not None

    if has_labels:
        max_length = min(max_length, len(label))
        if score is not None:
            max_length = min(max_length, len(score))
        if threshold is None and score is not None:
            threshold = np.mean(score) + 3 * np.std(score)
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)


    # plot 1: dati
    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.plot(data[:max_length], 'k')

    if has_labels:
        in_anomaly = False
        for i in range(max_length):
            if label[i] == 1:
                if not in_anomaly:
                    start = i
                    in_anomaly = True
            elif in_anomaly:
                if i - start == 1:
                    ax1.plot(start, data[start], 'ro')  # punto singolo
                else:
                    ax1.plot(range(start, i), data[start:i], 'r', linewidth=2)
                in_anomaly = False
        if in_anomaly:
            if max_length - start == 1:
                ax1.plot(start, data[start], 'ro')  # ultimo punto singolo
            else:
                ax1.plot(range(start, max_length), data[start:max_length], 'r', linewidth=2)

    ax1.set_xlim(plotRange)
    if (global_var.Turbine or global_var.TurbAgg) and global_var.single_column:
        col_name = col_names_Turbine()
        ax1.set_title(col_name[global_var.idx_column])
    else:
        ax1.set_title("Serie originale" if global_var.full_plot else f"Serie originale da {global_var.plotRange[0]} a {global_var.plotRange[1]}")

    # plot 2: score
    ax2 = fig.add_subplot(gs[1, :-1])
    ax2.plot(score[:max_length], label="Score")
    ax2.hlines(threshold, 0, max_length, linestyles='--', color='red', label="Threshold")
    ax2.set_ylabel('Score')
    ax2.set_xlim(plotRange)
    ax2.legend()
    ax2.set_title("Anomaly Scores")

    
    # plot 3: TP, FP, FN, TN
    ax3 = fig.add_subplot(gs[2, :-1])
    predictions = (score[:max_length] >= threshold).astype(int)
    if has_labels:
        index = label[:max_length] + 2 * predictions
        TP_idx = np.where(index == 3)[0]
        FP_idx = np.where(index == 2)[0]
        FN_idx = np.where(index == 1)[0]
        TN_idx = np.where(index == 0)[0]

        ax3.scatter(TN_idx, data[TN_idx], c='black', label='TN', marker='.', alpha=0.8)
        ax3.scatter(FP_idx, data[FP_idx], c='green', label='FP', marker='.')
        ax3.scatter(FN_idx, data[FN_idx], c='red', label='FN', marker='.')
        ax3.scatter(TP_idx, data[TP_idx], c='blue', label='TP', marker='.')


    else:
        anomalies = np.where(predictions == 1)[0]
        normal = np.where(predictions == 0)[0]

        ax3.scatter(anomalies, data[anomalies], c='red', label='Anomalie rilevate', marker='o')
        ax3.scatter(normal, data[normal], c='black', label='Valori normali', marker='.')
 
    ax3.set_xlim(plotRange)
    ax3.set_title("Predizioni")
    ax3.legend(loc='upper left')

    #fig.suptitle(f"{fileName}", fontsize=14)


def plot_semplice(dati):
    plt.figure(figsize=(12, 4))
    plt.plot(dati, color='blue')
    plt.title(f"Segnale dalla prima colonna (fino a {len(dati)} valori)")
    plt.xlabel("Indice temporale")
    plt.ylabel("Valore del sensore")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



#########
# ALTRO #
#########


def col_names_Turbine():
    return [
        "tag20SV_401/2 : Velocità turbina",
        "tagIM9 : Potenza erogata",
        "tag20FT_403 : Portata vapore ammissione",
        "tag20PT_411 : Pressione vapore ammissione",
        "tag20TT_411 : Temperatura vapore ammissione",
        "tag20FT_405 : Portata vapore estrazione regolata",
        "tagWA1_EXTRACTION : Pressione estrazione (set up)",
        "tag20PT_414 : Pressione vapore prelievo",
        "tag20TT_413 : Temperatura vapore prelievo",
        "tagZT_4-235C : Feedback apertura valvola woodward",
        "tagZT_4-235D : Feedback apertura valvola woodward",
        "tagZT_4-135C(Woodward) : Apertura valvola prelievo regolato",
        "tagZT_4-135D(Woodward) : Apertura valvola prelievo regolato",
        "tag20PT_416 : Pressione vapore scarico",
        "tag20TT_414 : Temperatura vapore scarico",
        "tag20TT_416 : Temperatura cassa",
        "tag20TT_425 : Temperatura cuscinetto anteriore",
        "tag20TT_427 : Temperatura cuscinetto posteriore",
    ]

def create_name(modelName,max_length,filepath):
    name = f"{modelName}_{'Slide' if global_var.slide else 'NoSlide'}_{max_length}_{os.path.basename(filepath)[:10]}"
        
    if "Turbine" in filepath and global_var.single_column:
        name = name + f"_col_{global_var.idx_column}"
    elif "Turbine" in filepath and global_var.single_column is False:
        name = name + f"_col_0_{global_var.idx_column}"
    print(name)
    return name


def threshold_func(anomaly_scores):
    return [np.mean(anomaly_scores)+3*np.std(anomaly_scores),
            np.mean(anomaly_scores)+2*np.std(anomaly_scores),
            np.mean(anomaly_scores)+1*np.std(anomaly_scores),
            np.mean(anomaly_scores)+0.5*np.std(anomaly_scores),
            np.percentile(anomaly_scores, 70),
            np.percentile(anomaly_scores, 80),
            np.percentile(anomaly_scores, 90),
            np.percentile(anomaly_scores, 95),
            np.percentile(anomaly_scores, 99),
            np.percentile(anomaly_scores, 99.5),
            iqr_threshold(anomaly_scores),
            np.median(anomaly_scores) + 3 * np.median(np.abs(anomaly_scores - np.median(anomaly_scores))),
            np.median(anomaly_scores) + 5 * np.median(np.abs(anomaly_scores - np.median(anomaly_scores)))]


def def_files():
    return [
        'dati/TSB-UAD-Public/Daphnet/S01R02E0.test.csv@1.out',
        'dati/TSB-UAD-Public/Dodgers/101-freeway-traffic.test.out',
        'dati/TSB-UAD-Public/ECG/MBA_ECG801_data.out',
        'dati/TSB-UAD-Public/Genesis/genesis-anomalies.test.csv@1.out',
        'dati/TSB-UAD-Public/GHL/01_Lev_fault_Temp_corr_seed_11_vars_23.test.csv@4.out',
        'dati/TSB-UAD-Public/IOPS/KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0.test.out',
        'dati/TSB-UAD-Public/KDD21/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.out',
        'dati/TSB-UAD-Public/MGAB/1.test.out',
        'dati/TSB-UAD-Public/MITDB/100.test.csv@1.out',
        'dati/TSB-UAD-Public/NAB/NAB_data_art0_0.out',
        'dati/TSB-UAD-Public/NASA-MSL/C-1.test.out',
        'dati/TSB-UAD-Public/NASA-SMAP/A-1.test.out',
        'dati/TSB-UAD-Public/Occupancy/room-occupancy-0.test.csv@1.out',
        'dati/TSB-UAD-Public/OPPORTUNITY/S1-ADL1.test.csv@16.out',
        'dati/TSB-UAD-Public/SensorScope/stb-2.test.out',
        'dati/TSB-UAD-Public/SMD/machine-1-1.test.csv@1.out',
        'dati/TSB-UAD-Public/SVDB/801.test.csv@1.out',
        'dati/TSB-UAD-Public/YAHOO/Yahoo_A1real_1_data.out',
    ]


##########
##########
## MAIN ##
##########
##########


def main():
    filelist = def_files()
    modellist = ['IForest', 'AutoEncoder', 'KNN_5', 'KNN_10', 'KNN_30']
    results = []

    if not global_var.allfiles:
        filelist = [filelist[global_var.idx_file]]
    if global_var.Turbine:
        if global_var.TurbAgg:
            filelist = ['dati/Turbine/df_2024_agg60_04-10']
        else:
            filelist = ['dati/Turbine/df_2024_04-10']

    for filepath in filelist:
        data, label, max_length = read_dataset(filepath, global_var.full_length)
        
        W_data, label, len_Win = create_windows(data, label, global_var.slide)

        if W_data.shape[0] <= 32:
            print(f"⚠️ Salto {filepath} perchè sono state create 32 o meno finestre")
            continue

        models_to_run = modellist if global_var.allmodels else [modellist[global_var.idx_modello]]

        for modelName in models_to_run:

            name = create_name(modelName, max_length, filepath)
            score, execution_time = create_model(W_data, modelName)
            print(f"Model {modelName} trained in {execution_time:.2f}s")

            score = post_processing_score(score, len_Win, global_var.slide)

            if label is not None:
                metrics_list = calculate_metrics(label, score)

                print(f"Metodo: {modelName}; AUC: {metrics_list[0]['auc']:.4f}")
                for metric in metrics_list:
                    results.append({
                        'file': filepath,
                        'folder': Path(filepath).parts[-2],
                        'model': modelName,
                        'auc': metric['auc'],
                        'precision': metric['precision'],
                        'recall': metric['recall'],
                        'f1': metric['f1'],
                        'threshold': metric['threshold'],
                        'time': execution_time
                    })
                    print(f"Threshold {metric['threshold']:.4f}: Precision: {metric['precision']:.4f}; Recall: {metric['recall']:.4f}; F1: {metric['f1']:.4f}")

            if global_var.show_plot or global_var.save_plot:
                create_plot(W_data, data, score, label, global_var.slide, name, modelName, filepath)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("risultati/RISULTATI_TEMP.csv", index=False)


main()