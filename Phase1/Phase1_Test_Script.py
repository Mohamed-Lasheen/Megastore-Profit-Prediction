import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import kendalltau, shapiro
from sklearn.feature_selection import SelectFromModel
from statsmodels.tsa.arima.model import ARIMA
import os


def Split_And_Retrieve_IDs(Dataset):
    Product_ID_Dict = dict()
    Order_ID_Dict = dict()
    Customer_ID_Dict = dict()
    for i in range(Dataset.shape[0]):
        if not pd.isnull(Dataset.iloc[i, 1]):
            Order_ID = int(Dataset.iloc[i, 1].split("-")[2])
            Order_ID_Dict[i] = Order_ID
        if not pd.isnull(Dataset.iloc[i, 5]):
            Customer_ID = int(Dataset.iloc[i, 5].split("-")[1])
            Customer_ID_Dict[i] = Customer_ID
        if not pd.isnull(Dataset.iloc[i, 13]):
            Product_ID = int(Dataset.iloc[i, 13].split("-")[2])
            Product_ID_Dict[i] = Product_ID
    return Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict

def Read_CategoryTree(Dataset):
    Main_Category_List = []
    Sub_Category_List = []
    Category_Tree_Index = Dataset.columns.get_loc('CategoryTree')
    for i in range(Dataset.shape[0]):
        if not pd.isnull(Dataset.iloc[i, Category_Tree_Index]):
            Main_Category_Flag = re.search("MainCategory", Dataset.iloc[i, Category_Tree_Index])
            Sub_Category_Flag = re.search("SubCategory", Dataset.iloc[i, Category_Tree_Index])

            if Main_Category_Flag:
                Main_Category = Dataset.iloc[i, Category_Tree_Index].split(":")[1].split("\'")[1]
            else:
                Main_Category = np.NaN

            if Sub_Category_Flag and Main_Category_Flag:
                Sub_Category = Dataset.iloc[i, Category_Tree_Index].split(":")[2].split("\'")[1]
            elif Sub_Category_Flag:
                Sub_Category = Dataset.iloc[i, Category_Tree_Index].split(":")[1].split("\'")[1]
            else:
                Sub_Category = np.NaN
            Main_Category_List.append(Main_Category)
            Sub_Category_List.append(Sub_Category)
        else:
            Main_Category_List.append(np.NaN)
            Sub_Category_List.append(np.NaN)
    Dataset['Main Category'] = Main_Category_List
    Dataset['Sub Category'] = Sub_Category_List
def Connect_Product_ID_With_Categories(Dataset):
    Categories = pd.DataFrame()
    Main_Category_ID_List = []
    Sub_Category_ID_List = []
    for i in range(Dataset.shape[0]):
        Main_Category_ID_List.append(Dataset.loc[i, 'Product ID'].split("-")[0])
        Sub_Category_ID_List.append(Dataset.loc[i, 'Product ID'].split("-")[1])
    Categories['Main Category ID'] = Main_Category_ID_List
    Categories['Sub Category ID'] = Sub_Category_ID_List
    Categories['Main Category'] = Dataset['Main Category']
    Categories['Sub Category'] = Dataset['Sub Category']
    Categories.to_csv("Categories.csv")

"""                                               Preprocessing                                                      """
def Data_Generalization(Test_Dataset, Full_Dataset, Columns):
    Read_CategoryTree(Full_Dataset)
    isExisting = os.path.exists('Categories.csv')
    if not isExisting:
        Connect_Product_ID_With_Categories(Full_Dataset)
    Categories = pd.read_csv('Categories.csv')
    Read_CategoryTree(Test_Dataset)
    Test_Dataset["Order Date"] = pd.to_datetime(Test_Dataset["Order Date"])
    Test_Dataset["Ship Date"] = pd.to_datetime(Test_Dataset["Ship Date"])
    #Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict = Split_And_Retrieve_IDs(Test_Dataset)
    #Handle_Missing_Values(Test_Dataset, Full_Dataset, Categories, Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict)
    Handle_Missing_Values_In_Test(Test_Dataset)
    Convert_Data_Types(Test_Dataset, Columns)
    Test_Dataset.drop('Country', axis=1, inplace=True)
    #Test_Dataset.drop('Product Name', axis=1, inplace=True)
    return Test_Dataset


###################################################################
def Show_Data(Dataset):
    print(Dataset.info())
    print(Dataset.head())
    print(Dataset.describe())


def Feature_Scaling_Test(Dataset, Columns):
    # Feature scaling
    with open('scaler_mod_phase1.pkl', 'rb') as f:
        Scaler_mod = pickle.load(f)
    for i ,column in enumerate(Columns):
        Data = Dataset.loc[:, column].values.reshape(-1, 1)
        Scaled_Data = Scaler_mod[i].transform(Data)
        # print(Scaled_Data)
        Dataset.loc[:, column] = Scaled_Data

def feature_encoder_transform(dataset):
    """
    Transform Action => apply the trained LabelEncoder models from (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will apply on
    """
    with open('features_categorical_phase1.pkl', 'rb') as f:
        features_categorical = pickle.load(f)
    for feature_obj in features_categorical:
        dataset[feature_obj] = [label if label in features_categorical[feature_obj].classes_ else "Unknown" for label in
                                dataset[feature_obj]]
        dataset[feature_obj] = features_categorical[feature_obj].transform(list(dataset[feature_obj].values))

def Check_And_Remove_Duplicates(Dataset):
    if Dataset.duplicated().any():
        print("num of duplicated values",Dataset.duplicated().sum())
        Dataset.drop_duplicates(inplace=True)
        print("after deleting Duplicates",Dataset.duplicated().sum())
        return Dataset


def Convert_Data_Types(Dataset, Columns):
    for column in Columns["Numerical_Columns"]:
        if column in Columns["Numerical_And_Continuous_Columns"]:
            Dataset[column] = Dataset[column].astype(np.float64)
        else:
            Dataset[column] = Dataset[column].astype(np.int64)

def Handle_Missing_Values_In_Test(Test_Dataset):
    if Test_Dataset["Profit"].isna().any():
        Test_Dataset["Profit"] = Test_Dataset["Profit"].dropna(how='any')

    if Test_Dataset["Discount"].isna().any():
        Test_Dataset["Discount"].fillna(0.2, inplace=True)

    if Test_Dataset["Sales"].isna().any():
        Test_Dataset["Sales"].fillna(54.792, inplace=True)

    if Test_Dataset["Quantity"].isna().any():
        Test_Dataset["Quantity"].fillna(3, inplace=True)

    if Test_Dataset["Country"].isna().any():
        Test_Dataset["Country"].fillna("United States", inplace=True)

    if Test_Dataset["Product Name"].isna().any():
        Test_Dataset["Product Name"].fillna("Staples", inplace=True)

    if Test_Dataset["Main Category"].isna().any():
        Test_Dataset["Main Category"].fillna("Office Supplies", inplace=True)

    if Test_Dataset["Sub Category"].isna().any():
        Test_Dataset["Sub Category"].fillna("Binders", inplace=True)

    if Test_Dataset["Product ID"].isna().any():
        Test_Dataset["Product ID"].fillna("TEC-AC-10003832", inplace=True)

    if Test_Dataset["Order ID"].isna().any():
        Test_Dataset["Order ID"].fillna("CA-2017-100111", inplace=True)

    if Test_Dataset["Customer ID"].isna().any():
        Test_Dataset["Customer ID"].fillna("PP-18955", inplace=True)

    Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict = Split_And_Retrieve_IDs(Test_Dataset)
    Test_Dataset["Order ID"] = dict(sorted(Order_ID_Dict.items())).values()
    Test_Dataset["Customer ID"] = dict(sorted(Customer_ID_Dict.items())).values()
    Test_Dataset["Product ID"] = dict(sorted(Product_ID_Dict.items())).values()

    if Test_Dataset["Segment"].isna().any():
        Test_Dataset["Segment"].fillna("Consumer", inplace=True)

    if Test_Dataset["Customer Name"].isna().any():
        Test_Dataset["Customer Name"].fillna("Paul Prost", inplace=True)

    if Test_Dataset["Ship Mode"].isna().any():
        Test_Dataset["Ship Mode"].fillna("Standard Class", inplace=True)

    if Test_Dataset["Order Date"].isna().any():
        Test_Dataset["Order Date"].fillna(pd.to_datetime("2017-09-02 00:00:00"), inplace=True)

    if Test_Dataset["Ship Date"].isna().any():
        Test_Dataset["Ship Date"].fillna(pd.to_datetime("2017-09-26 00:00:00"), inplace=True)

    if Test_Dataset["Postal Code"].isna().any():
        Test_Dataset["Postal Code"].fillna(10035, inplace=True)

    if Test_Dataset["City"].isna().any():
        Test_Dataset["City"].fillna("New York City", inplace=True)

    if Test_Dataset["State"].isna().any():
        Test_Dataset["State"].fillna("California", inplace=True)

    if Test_Dataset["Region"].isna().any():
        Test_Dataset["Region"].fillna("West", inplace=True)
    if Test_Dataset["Row ID"].isna().any():
        Test_Dataset["Row ID"].fillna(5007.5, inplace=True)

def boxplot(Numerical, plot_cols):
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.flatten()

    # Plot boxplots for each column
    for i, col in enumerate(plot_cols):
        sns.boxplot(x=Numerical[col], ax=axs[i])
        axs[i].set_title(col)

    plt.tight_layout()
    plt.show()

def Check_Outliers_Number(Data_train):
    Z_Score = np.abs((Data_train-Data_train.mean())/Data_train.std())
    Threshold = 3
    num_out = len(Z_Score[(Z_Score > Threshold).any(axis=1)])
    print("num of outliers:= ", num_out)


def Check_Duplicates(Data_train):
    print("num of duplicated values", Data_train.duplicated().sum())
    Data_train.drop_duplicates(inplace=True)
    print("after deleting Duplicates", Data_train.duplicated().sum())
    return Data_train

def overfit(model, X_train, Y_Train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, Y_Train, cv=5)

    # calculate the mean and standard deviation of the training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot the learning curves
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    plt.xlabel('Number of training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.show()

def Time_Series_Analysis(df):
    ts = df.set_index('Order Date')['Profit']
    # visualize the data
    plt.plot(ts)
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.show()

    # fit ARIMA model to the data
    model = ARIMA(ts, order=(1, 1, 1))
    results = model.fit()

    # make predictions on future values
    future_dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
    future_df = pd.DataFrame(index=future_dates, columns=['Profit'])
    predictions = results.predict(start=len(ts), end=len(ts)+len(future_df)-1, typ='levels')
    future_df['Profit'] = predictions

    # plot residual errors
    residuals = pd.DataFrame(results.resid)
    residuals.plot(title='Residuals')
    plt.show()
    residuals.plot(kind='kde', title='Density Plot of Residuals')
    plt.show()

    # summarize model fit
    print(results.summary())


###################################################################
########################## preprocessing ##########################
###################################################################
def Handle_Outliers(df, columns):
    Numerical_Dataframe = df[columns['Numerical_Columns']]

    boxplot(Numerical_Dataframe, columns['Numerical_Columns'])
     #outliers
    for col in Numerical_Dataframe:
        #if col != "Profit":
            # calculate interquartile range
        q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        Column_Index = df.columns.get_loc(col)
        #outliers = ((df[col] < lower) | (df[col] > upper))
        for i in range(df.shape[0]):
            if df.iloc[i, Column_Index] < lower:
                df.iloc[i, Column_Index] = lower
            if df.iloc[i, Column_Index] > upper:
                df.iloc[i, Column_Index] = upper

    Numerical_Dataframe = df[columns['Numerical_Columns']]

    boxplot(Numerical_Dataframe, columns['Numerical_Columns'])

    return df
def Remove_Outliers(df,columns):
    Numerical_Dataframe = df[columns['Numerical_Columns']]
    for col in Numerical_Dataframe:
        # calculate interquartile range
        q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = ( ( df[col] < lower) | (df[col] > upper) )
        index_label = df[outliers].index
        df.drop(index_label, inplace=True)
        # Drop rows based on the outlier mask
        df = df.drop(df[outliers].index)
        print(f'Number of outliers in {col}: {len(index_label)}')
    # Numerical_Dataframe = df[columns['Numerical_Columns']]
    return df
def Detect_Outliers(df,columns):
    Numerical_Dataframe = df[columns['Numerical_Columns']]
    for col in Numerical_Dataframe:
        # calculate interquartile range
        q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = ( ( df[col] < lower) | (df[col] > upper) )
        index_label = df[outliers].index

        print(f'Number of outliers in {col}: {len(index_label)}')
    # Numerical_Dataframe = df[columns['Numerical_Columns']]
    return df

def Preprocess_Test(df, columns):
    missing_val = df.isnull().sum()
    print("Null Values In Test : ",missing_val)
    print("Null Values In Test : ",missing_val[missing_val==True].index.tolist())
    # df=Handle_Outliers(df,columns)
    # df=Remove_Outliers(df,columns)
    print("#"*100)
    print("Num Of Outliers In Test : ")
    df=Detect_Outliers(df,columns)
    print("#"*100)
    Numerical_Dataframe = df[columns['Numerical_Columns']]
    feature_encoder_transform(df)
    Feature_Scaling_Test(df, Numerical_Dataframe)
    return df
def Data_Compare (Data,Full_Dataset,Split_Columns,str):
    print("#"*100)
    print("#"*50,str,"#"*50)
    print(Data)
    Data = Data_Generalization(Data, Full_Dataset, Split_Columns)
    print("#"*100)
    print("#"*50,str," After Handling","#"*50)
    print(Data)
    return Data
"""                                                Function Call                                                     """

# Read Data
Full_Dataset = pd.read_csv('megastore-regression-dataset.csv')
Test_Dataset = pd.read_csv('megastore-regression-dataset.csv')
# Test_Dataset = pd.read_csv('regtst.csv')
ID_Columns = ['Order ID', 'Customer ID', 'Product ID']
Numerical_And_Discrete_Columns = ['Postal Code', 'Quantity']
Numerical_And_Continuous_Columns = ['Sales', 'Discount', 'Profit']
Numerical_Columns = ['Row ID', 'Order ID', 'Customer ID', 'Postal Code',
                     'Product ID', 'Sales', 'Quantity', 'Discount', 'Profit']
Categorical_Columns = ['Ship Mode', 'Customer Name', 'Segment', 'City', 'State', 'Region',
                       'Main Category', 'Sub Category', 'Product Name']

Split_Columns = dict()
Split_Columns["Categorical_Columns"] = Categorical_Columns
Split_Columns["Numerical_Columns"] = Numerical_Columns
Split_Columns["Numerical_And_Discrete_Columns"] = Numerical_And_Continuous_Columns
Split_Columns["Numerical_And_Continuous_Columns"] = Numerical_And_Continuous_Columns
Split_Columns["ID_Columns"] = ID_Columns
# Train_Data, Test_Data = train_test_split(Test_Dataset, test_size=0.2, shuffle=True, random_state=42)

Test_Data=pd.read_csv('megastore-regression-dataset.csv')
Test_Data=Data_Compare(Test_Data,Full_Dataset,Split_Columns,"Test_Data")
Test_Data= Preprocess_Test(Test_Data, Split_Columns)


X_test = Test_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region', 'Main Category',
                    'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Test = Test_Data['Profit']

print('1- linear regression')
# Fit a linear regression model
Linear_Regression_Model = LinearRegression()

File_Name_Linear_Regression_Model = 'linear_reg_model.sav'
Load_Linear_Regression_Model = pickle.load(open(File_Name_Linear_Regression_Model, 'rb'))

Y_Prediction_Linear_Regression = Load_Linear_Regression_Model.predict(X_test)
Linear_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction_Linear_Regression)
print("Linear Regression MSE: ", Linear_Regression_MSE)
r2 = r2_score(Y_Test, Y_Prediction_Linear_Regression)
print("R-squared score Y_T&Y_Prediction:", r2)


###############################################################

###########################
print()
print('2- Random Forest Regressor')
Random_Forest_Regression_Model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=123)
File_Name_RFR_Model = 'RandomForestRegressor_model.sav'
Load_RFR_Model = pickle.load(open(File_Name_RFR_Model, 'rb'))
Y_Prediction = Load_RFR_Model.predict(X_test)
# Evaluate the Random_Forest_Regression_Model on the testing data
Random_Forest_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction)
print("Random Forest Regression MSE:", Random_Forest_Regression_MSE)
# plt.scatter(Y_Test, Y_Prediction)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.show()

r2 = r2_score(Y_Test, Y_Prediction)
print("R-squared score Y_T&Y_Prediction:", r2)


   ######################################################
# TODO  encode fun to efect in data set
print('3- Decision tree Algorithm')
# max_depth:= level of the tree
Decision_Tree_Regressor = DecisionTreeRegressor(max_depth=5)

File_Name_Decision_Tree_Model = 'Decision tree_model.sav'

Load_Decision_Tree_Model = pickle.load(open(File_Name_Decision_Tree_Model, 'rb'))

Y_Prediction_For_Decision_Tree = Load_Decision_Tree_Model.predict(X_test)

Decision_Tree_MSE = mean_squared_error(Y_Test, Y_Prediction_For_Decision_Tree)
print("Decision tree MSE : ", Decision_Tree_MSE)


r2 = r2_score(Y_Test, Y_Prediction_For_Decision_Tree)
print("R-squared score Y_T&Y_Prediction:", r2)
###########################################################
print('4- XGBRegressor Algorithm')
File_Name_XGBRegressor_model = 'XGBRegressor_model.sav'
# pickle.dump(XGBRegressor_model, open(File_Name_XGBRegressor_model, 'wb'))
Load_XGBRegressor_model = pickle.load(open(File_Name_XGBRegressor_model, 'rb'))
Y_Prediction_For_XGBRegressor = Load_XGBRegressor_model.predict(X_test)
XGBRegressor_MSE = mean_squared_error(Y_Test, Y_Prediction_For_XGBRegressor)
print("XGBRegressor_model MSE : ", XGBRegressor_MSE)

r1 = Load_XGBRegressor_model.score(X_test, Y_Test)
print("LR-squared score X_Test & Y_test:", r1)
r2 = r2_score(Y_Test, Y_Prediction_For_XGBRegressor)
print("R-squared score Y_T&Y_Prediction:", r2)
#############################################################################
print('5- Polynomial regression')
with open('poly.pkl', 'rb') as f:
    poly = pickle.load(f)
X_test_poly = poly.transform(X_test)
File_Name_poly_model = 'poly_model.sav'
# pickle.dump(Linear_Regression_Model, open(File_Name_poly_model, 'wb'))
Load_Polynomial_model = pickle.load(open(File_Name_poly_model, 'rb'))
# Predict values for test set
Y_Prediction_Linear_Regression = Load_Polynomial_model.predict(X_test_poly)
Linear_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction_Linear_Regression)
print("Linear Regression MSE: ", Linear_Regression_MSE)

r2 = r2_score(Y_Test, Y_Prediction_Linear_Regression)
print("R-squared score Y_T&Y_Prediction:", r2)


#####################################################################################
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
def plts(i,Y_Prediction_mod,Y_Test,str):
    axs[i].scatter(Y_Prediction_mod, Y_Test, color='gray')
    axs[i].plot(Y_Test, Y_Test, color='blue', linestyle='--')
    axs[i].set_xlabel('Predicted Profit')
    axs[i].set_ylabel('Actual Profit')
    axs[i].set_title(str)
plts(0,Y_Prediction_Linear_Regression,Y_Test,'Linear Regression Prediction')
plts(1,Y_Prediction,Y_Test,'Random Forest Regression Prediction')
plts(2,Y_Prediction_For_Decision_Tree,Y_Test,'Decision Tree Regression Prediction')
# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)
# Show the plot
plt.show()
