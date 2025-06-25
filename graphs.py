import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import global_var
import numpy as np


def plot_auc_stripplot(metrics):
    df = pd.DataFrame(metrics)
    df_unique = df.drop_duplicates(subset=['file', 'model'])

    palette = sns.color_palette("hsv", len(df_unique['file'].unique()))
    plt.figure(figsize=(7, 4))
    ax = sns.swarmplot(
        data=df_unique,
        x='auc',
        y='model',
        #hue='model',
        alpha=0.8,
        size=6,
        #palette=palette
        color='black'
    )

    models = df_unique['model'].unique()

    for i, model in enumerate(models):
        ax.axhline(i, color='lightgray', linestyle='-', linewidth=1, zorder=0)

    #ax.set_title("Distribuzione AUC per modello")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Modello")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim([0,1])
    #ax.legend_.remove()
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig("risultati/swarmplot")
    plt.show()



def create_latex_table(df, caption, label, to_ignore=None):
    df = df.drop(df.columns[global_var.metric_to_ignore], axis=1)

    return df.to_latex(
        index=True,
        float_format="%.3f",
        caption=caption,
        label=label,
        column_format='l' + 'r' * df.shape[1],
        na_rep="",
        escape=False,
        bold_rows=False
    )

def create_metric_latex_tables(results, threshold_labels=None, to_ignore=None):
    df = pd.DataFrame(results)
    df['folder'] = df['file'].apply(lambda x: Path(x).parts[-2])
    df['threshold_idx'] = df.groupby(['file', 'model']).cumcount()

    selected_model = df['model'].unique()[global_var.idx_modello]
    df_model = df[df['model'] == selected_model]

    for metric in ['precision', 'recall', 'f1']:
        pivot = df_model.pivot(index='folder', columns='threshold_idx', values=metric)
        if threshold_labels:
            pivot.columns = threshold_labels
        else:
            pivot.columns = [f"T{int(c)}" for c in pivot.columns]
        pivot = pivot.fillna(0).round(3)

        # Aggiungi colonna AUC se la metrica è F1
        if metric == 'f1':
            auc_values = df_model.drop_duplicates(subset=['file', 'model'])[['folder', 'auc']]
            auc_values = auc_values.set_index('folder').reindex(pivot.index).fillna(0).round(3)
            pivot['AUC'] = auc_values['auc']

        mean_row = pivot.mean(numeric_only=True).round(3)
        mean_row.name = 'Media'
        pivot = pd.concat([pivot, pd.DataFrame([mean_row])])

        tex_code = create_latex_table(
            pivot,
            f"{metric.upper()} by Folder \\& Threshold",
            f"tab:{metric}_by_folder",
            to_ignore
        )
        with open(f"risultati/{metric}_table.tex", "w") as f:
            f.write(tex_code)




def create_execution_time_table(results, dataset_lengths):
    df = pd.DataFrame(results)
    df['folder'] = df['file'].apply(lambda x: Path(x).parts[-2])

    df_grouped = df.groupby(['folder', 'model'])['time'].mean().reset_index()
    pivot = df_grouped.pivot(index='folder', columns='model', values='time')

    model_order = ['IForest', 'AutoEncoder', 'KNN_5', 'KNN_10', 'KNN_30']
    pivot = pivot.reindex(columns=model_order)

    pivot = pivot.fillna(0).round(3)

    folders_sorted = sorted(pivot.index.tolist())
    length_map = dict(zip(folders_sorted, dataset_lengths))

    pivot['Lunghezza'] = pivot.index.map(length_map)
    pivot = pivot.sort_values(by='Lunghezza', ascending=False)

    tex_code = create_latex_table(
        pivot,
        caption="Execution Time (s) per dataset e modello, con lunghezza dei dataset",
        label="tab:execution_time"
    )

    with open("risultati/execution_time_table.tex", "w") as f:
        f.write(tex_code)


def plot_execution_time_vs_length(results, lengths):
    df = pd.DataFrame(results)
    df['folder'] = df['file'].apply(lambda x: Path(x).parts[-2])

    df_grouped = df.groupby(['folder', 'model'])['time'].mean().reset_index()

    pivot = df_grouped.pivot(index='folder', columns='model', values='time')

    model_order = ['IForest', 'AutoEncoder', 'KNN_5', 'KNN_10', 'KNN_30']
    pivot = pivot.reindex(columns=model_order)

    pivot['length'] = lengths
    pivot = pivot.sort_values(by='length')

    plt.figure(figsize=(8, 5))
    for model in pivot.columns.drop('length'):
        plt.plot(pivot['length'], pivot[model], marker='o', label=model)

    plt.xlabel("Dataset Length (n)")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Dataset Length")
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("risultati/execution_time_vs_length.png", dpi=300)
    plt.show()


def plot_execution_log_time_vs_length(results, lengths):
    df = pd.DataFrame(results)
    df['folder'] = df['file'].apply(lambda x: Path(x).parts[-2])

    df_grouped = df.groupby(['folder', 'model'])['time'].mean().reset_index()

    pivot = df_grouped.pivot(index='folder', columns='model', values='time')

    model_order = ['IForest', 'AutoEncoder', 'KNN_5', 'KNN_10', 'KNN_30']
    pivot = pivot.reindex(columns=model_order)

    if global_var.Graphs_on_NoSlide is False:
        pivot['length'] = lengths
        pivot = pivot.sort_values(by='length')

    pivot['log_length'] = np.log(pivot['length'])

    plt.figure(figsize=(8, 5))
    for model in model_order:
        if model in pivot:
            plt.plot(pivot['log_length'], np.log(pivot[model]), marker='o', label=model)

    plt.xlabel("log(Dataset Length)")
    plt.ylabel("log(Execution Time)")
    plt.title("Log-Log Plot: Execution Time vs Dataset Length")
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("risultati/execution_time_vs_length_loglog.png", dpi=300)
    plt.show()


def create_auc_latex_table(results):
    df = pd.DataFrame(results)
    df['folder'] = df['file'].apply(lambda x: Path(x).parts[-2])

    df_grouped = df.groupby(['folder', 'model'])['auc'].mean().reset_index()

    pivot = df_grouped.pivot(index='folder', columns='model', values='auc')
    model_order = ['IForest', 'AutoEncoder', 'KNN_5', 'KNN_10', 'KNN_30']
    pivot = pivot.reindex(columns=model_order)
    pivot = pivot.round(3)

    latex_code1 = pivot.to_latex(
        index=True,
        caption="AUC medio per dataset e modello",
        label="tab:auc_by_folder",
        float_format="%.3f",
        column_format="l" + "r" * len(pivot.columns),
        escape=False,
        bold_rows=True
    )

    stats_df = pd.DataFrame({
        "Media AUC": pivot.mean(),
        "Std Dev": pivot.std()
    }).round(3).reset_index().rename(columns={"index": "Modello"})

    latex_code2 = stats_df.to_latex(
        index=False,
        caption="Media AUC per modello",
        label="tab:auc_means",
        float_format="%.3f",
        column_format="lr",
        escape=False
    )

    with open("risultati/auc_table.tex", "w") as f:
        f.write(latex_code1 + "\n\n" + latex_code2)


def main():
    to_ignore = []#7,8]
    if global_var.Graphs_on_temp:
        results = pd.read_csv("risultati/RISULTATI_TEMP.csv")
        threshold_labels = []

    elif global_var.Graphs_on_iF:
        results = pd.read_csv("risultati/RISULTATI_IFOREST.csv")
        threshold_labels = ["$\mu + 3\sigma$","$\mu + 2\sigma$","$\mu + 1\sigma$","$\mu + .5\sigma$","70\%","80\%","90\%","95\%","99\%","99.5\%","IQR","MAD3","MAD5"]
    
    elif global_var.Graphs_on_NoSlide:
        results = pd.read_csv("risultati/RISULTATI_NoSlide.csv")
        threshold_labels = ["$\mu + 3\sigma$","$\mu + 2\sigma$","$\mu + 1\sigma$","$\mu + .5\sigma$","70\%","80\%","90\%","95\%","99\%","99.5\%","IQR","MAD3","MAD5"]

    elif global_var.Graphs_on_Belli:
        results = pd.read_csv("risultati/RISULTATI.csv")
        threshold_labels = ["$\mu + 3\sigma$","$\mu + 2\sigma$","$\mu + 1\sigma$","95\%","99\%","99.5\%","IQR","MAD3","MAD5"]


    if global_var.metric_latex:
        create_metric_latex_tables(results, threshold_labels, to_ignore)
    
    else:
    #if global_var.compare_models and (global_var.Graphs_on_Belli or global_var.Graphs_on_NoSlide or global_var.Graphs_on_temp):
        plot_auc_stripplot(results)

        lengths = [28800,50400,227900,16220,200001,149156,79795,100000,650000,2264,8640,2665,50670,29553,28479,230400,1420]
        create_execution_time_table(results, lengths)

        if global_var.Graphs_on_NoSlide is False:
            plot_execution_log_time_vs_length(results, lengths)

        create_auc_latex_table(results)


main()