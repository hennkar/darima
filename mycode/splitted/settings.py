import time

# DataFrame names
series_name = 'TOTAL'
dist_id_name = "partition_id"
id_name = "id"
X_name = "time"
Y_name = "demand"

# Data to be used
file_train_path = '/home/ole/IdeaProjects/darima/data/' + series_name + '_train.csv'
file_test_path = '/home/ole/IdeaProjects/darima/data/' + series_name + '_test.csv'

# Save locations
model_saved_file_name = '/home/ole/IdeaProjects/darima/result/darima_model_' + series_name + '_' + time.strftime(
    "%Y-%m-%d-%H:%M:%S",
    time.localtime()) + '.pkl'
coef_saved_file_name = '/home/ole/IdeaProjects/darima/result/darima_coef_' + series_name + '_' + time.strftime(
    "%Y-%m-%d-%H:%M:%S",
    time.localtime()) + '.csv'
forec_saved_file_name = '/home/ole/IdeaProjects/darima//result/darima_forec_' + series_name + '_' + time.strftime(
    "%Y-%m-%d-%H:%M:%S", time.localtime()) + '.csv'

# DARIMA & PySpark
# TODO: back to 2879
horizon = 120
amount_of_dist = 30
num_cores = 2

# ARIMA Model
tol = 80
period = 24
order = [0, 0, 0]
seasonal = [0, 0, 0]
max_p = 5
max_q = 5
max_P = 2
max_Q = 2
max_order = 5
max_d = 2
max_D = 1
allowmean = True
allowdrift = True
method = "CSS"  # Fitting method
approximation = False
stepwise = True
parallel = False
level = 95
