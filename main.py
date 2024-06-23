import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from credit_scrore.models.venera import venera_train_and_prediction, venera_cross_validation
from credit_scrore.models.venera_mse import venera_train_and_prediction_mse, venera_cross_validation_mse, plot_model_view
from credit_scrore.models.mercury import mercury_cross_validation, mercury_train_and_prediction
from credit_scrore.models.mercury_mse import mercury_cross_validation_mse, mercury_train_and_prediction_mse

# Constants
EPOCHS = 100
EPOCHS_VALIDATION = 40
# ['EM', 'gain', 'hyperimpute', 'ice', 'mean', 'median', 'most_frequent', 'nop', 'sinkhorn', 'softimpute']
train_df = pd.read_csv('./credit_scrore/data/credit_score_regression/CreditScore_train.csv')
test_df = pd.read_csv('./credit_scrore/data/credit_score_regression/CreditScore_test.csv')

train_df_hyperimpute = pd.read_csv('./credit_scrore/data/credit_score_regression/train_df_imputed_hyperimpute.csv')
test_df_hyperimpute = pd.read_csv('./credit_scrore/data/credit_score_regression/test_df_imputed_hyperimpute.csv')

train_df_mean = pd.read_csv('./credit_scrore/data/credit_score_regression/train_df_imputed_mean_sklearn.csv')
test_df_mean = pd.read_csv('./credit_scrore/data/credit_score_regression/test_df_imputed_mean_sklearn.csv')

train_df_most_frequent = pd.read_csv('./credit_scrore/data/credit_score_regression/train_df_imputed_most_frequent_sklearn.csv')
test_df_most_frequent = pd.read_csv('./credit_scrore/data/credit_score_regression/test_df_imputed_most_frequent_sklearn.csv')

train_df_median = pd.read_csv('./credit_scrore/data/credit_score_regression/train_df_imputed_median_sklearn.csv')
test_df_median = pd.read_csv('./credit_scrore/data/credit_score_regression/test_df_imputed_median_sklearn.csv')

venera = pd.read_csv('./venera_prediction_normalisation_mse_median.csv')
venera['Id'] = venera['Id'].astype(int)
print(venera.head(10))

train_df_filled = train_df.fillna(0)
test_df_filled = test_df.fillna(0)
data_sets = [[train_df_hyperimpute, test_df_hyperimpute, 'hyperimpute'],[train_df_filled, test_df_filled,
'fill_with_zeroes'], [train_df_mean, test_df_mean, 'mean'], [train_df_most_frequent, test_df_most_frequent,
'most_frequent'], [train_df_median, test_df_median, 'median']]
# data_sets = [[train_df_median, test_df_median, 'median']]
# result = pd.read_csv('./venera_prediction_normalisation_mse_median.csv')
# result.plot()
# plt.show()
columns = ['Model', 'Imputer', 'MAPE']
df_initial = pd.DataFrame(columns=columns)

df_initial.to_csv('model_performance_1.csv', index=False)

for train, test, imputer_name in data_sets:
    print(f'Begin imputer {imputer_name}')
    scaler = MinMaxScaler()
    # Train data normalised
    X = train.drop(columns=['x001', 'y'])
    numeric_columns = X.select_dtypes(include=['int', 'float']).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    y = train['y']
    # Test data normalised
    ids = test['x001']
    new_X = test.drop(columns=['x001', 'y'])
    new_X[numeric_columns] = scaler.fit_transform(new_X[numeric_columns])
    new_y = test['y']

    # With that config the test data is around 15%
    kf = KFold(n_splits=8, shuffle=True, random_state=42)

    # VENERA model
    # venera_cross_validation(kf, X, y, EPOCHS_VALIDATION)
    venera_train_and_prediction(X, y, new_X, new_y, EPOCHS, imputer_name, ids)

    # VENERA MSE model
    # venera_cross_validation_mse(kf, X, y, EPOCHS_VALIDATION)
    # plot_model_view(X.shape[1])
    venera_train_and_prediction_mse(X, y, new_X, new_y, EPOCHS, imputer_name, ids)

    # MERCURY model
    # mercury_cross_validation(kf, X, y, EPOCHS_VALIDATION)
    mercury_train_and_prediction(X, y, new_X, new_y, EPOCHS, imputer_name, ids)

    # MERCURY MSE model
    # mercury_cross_validation_mse(kf, X, y, EPOCHS_VALIDATION)
    mercury_train_and_prediction_mse(X, y, new_X, new_y, EPOCHS, imputer_name, ids)






# print(mean_absolute_percentage_error(result['Predicted'], result['Real']))
