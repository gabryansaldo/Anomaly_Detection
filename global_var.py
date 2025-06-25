# Plot
show_plot = False
save_plot = False
show_metrics = True
full_plot = False
plotRange = [1000,1420]

#Graphs.py
metric_latex = True
metric_to_ignore=[1,3,4,5,7,8,12]
  #compare_models = True

Graphs_on_Belli = False
Graphs_on_temp = False
Graphs_on_iF = False     # metti anche idx_modello
Graphs_on_NoSlide = True

# Turbine
Turbine = False
TurbAgg = False
single_column = True
idx_column = 2       # ultima (o unica) colonna da comprendere (max 18)
len_Win_multi = 60

# File
allfiles = False   ###
idx_file = 0
full_length = True
limit_length = 10000

# Finestre
slide = False

# Thresholds
allthresholds = True
idx_threshold = 0    # con allthreshold True idx serve per il grafico

# Modello
allmodels = False   ###
idx_modello = 1
metric_kNN = 'euclidean'     # options: 'euclidean', 'l1', 'l2', 'manhattan', 'mahalanobis', 'minkowski'

