import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
import seaborn as sns
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
    Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict = Split_And_Retrieve_IDs(Test_Dataset)
    #Handle_Missing_Values(Test_Dataset, Full_Dataset, Categories, Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict)
    Test_Dataset= Handle_Missing_Values_In_Test(Test_Dataset)
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
    with open('scaler_mod.pkl', 'rb') as f:
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
    with open('features_categorical.pkl', 'rb') as f:
        features_categorical1 = pickle.load(f)
    for feature_obj in features_categorical1:
        dataset[feature_obj] = [label if label in features_categorical1[feature_obj].classes_ else "Unknown" for label in
                                dataset[feature_obj]]
        dataset[feature_obj] = features_categorical1[feature_obj].transform(list(dataset[feature_obj].values))

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
            print(Dataset[column])
            Dataset[column] = Dataset[column].astype(np.int64)

def Handle_Missing_Values_In_Test(Test_Dataset):
    if Test_Dataset["ReturnCategory"].isna().any():
        Test_Dataset["ReturnCategory"] = Test_Dataset["ReturnCategory"].dropna(how='any')

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
    return Test_Dataset

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

def boxplot(Numerical, plot_cols):
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    axs = axs.flatten()

    # Plot boxplots for each column
    for i, col in enumerate(plot_cols):
        sns.boxplot(x=Numerical[col], ax=axs[i])
        axs[i].set_title(col)

    plt.tight_layout()
    plt.show()

###################################################################
########################## preprocessing ##########################
###################################################################
def Handle_Outliers(df, columns):
    Numerical_Dataframe = df[columns['Numerical_Columns']]

    boxplot(Numerical_Dataframe, columns['Numerical_Columns'])
     #outliers
    for col in Numerical_Dataframe:
        #if col != "ReturnCategory":
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
Full_Dataset = pd.read_csv('megastore-classification-dataset.csv')

Test_Data = pd.read_csv('megastore-classification-dataset.csv')
# Test_Data = pd.read_csv('TSt.csv')

ID_Columns = ['Order ID', 'Customer ID', 'Product ID']
Numerical_And_Discrete_Columns = ['Postal Code', 'Quantity']
Numerical_And_Continuous_Columns = ['Sales', 'Discount']
Numerical_Columns = ['Row ID', 'Order ID', 'Customer ID', 'Postal Code',
                     'Product ID', 'Sales', 'Quantity', 'Discount']
Categorical_Columns = ['Ship Mode', 'Customer Name', 'Segment', 'City', 'State', 'Region',
                       'Main Category', 'Sub Category', 'Product Name','Row ID', 'Order ID', 'Customer ID', 'Postal Code',
                     'Product ID', 'Sales', 'Quantity', 'Discount']

Split_Columns = dict()
Split_Columns["Categorical_Columns"] = Categorical_Columns
Split_Columns["Numerical_Columns"] = Numerical_Columns
Split_Columns["Numerical_And_Discrete_Columns"] = Numerical_And_Continuous_Columns
Split_Columns["Numerical_And_Continuous_Columns"] = Numerical_And_Continuous_Columns
Split_Columns["ID_Columns"] = ID_Columns

Test_Data=Data_Compare(Test_Data,Full_Dataset,Split_Columns,"Test_Data")

Test_Data= Preprocess_Test(Test_Data, Split_Columns)

X_test = Test_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region', 'Main Category',
                    'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Test = Test_Data['ReturnCategory']
####################################################################################
print("1-DecisionTreeClassifier")

start_time = time.time()
dtc = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_split=10) # 84% accuracy after slect the best hyperparm =Accuracy_tree: 0.8561601000625391


File_Name_Decision_Tree_Model = 'Decision tree_model.sav'
# Make predictions on the testing set
start_time = time.time()
Load_Decision_Tree_Model = pickle.load(open(File_Name_Decision_Tree_Model, 'rb'))
y_pred = Load_Decision_Tree_Model.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_Test, y_pred)
testing_time_Tre = time.time() - start_time
print("     tst_time:  ",testing_time_Tre)
print(f"    Accuracy_tree: {accuracy}")
# Precision, F1-score
DecTree_precision,DecTree_recall,DecTree_f1_score,DecTree_support=score(Y_Test, y_pred, average='macro')
print ('    Decision tree - Precision : {}'.format(DecTree_precision))
print ('    Decision tree - F1-score  : {}'.format(DecTree_f1_score))
#############################################################################
print("2-Logistic Regression")

lr = LogisticRegression(C=10,max_iter=100,solver="newton-cg")
File_Name_Logistic_Regression_Model = 'Logistic Regression.sav'

# Make predictions on the test set using the logistic regression model
start_time = time.time()
Load_Logistic_Regression_Model = pickle.load(open(File_Name_Logistic_Regression_Model, 'rb'))
y_pred_lr = Load_Logistic_Regression_Model.predict(X_test)
#y_pred_lr = lr.predict(X_test)
testing_time_lr = time.time() - start_time
print("     tst_time:  ",testing_time_lr)
# Measure the classification accuracy of the logistic regression model
accuracy_lr = accuracy_score(Y_Test, y_pred_lr)
print(f"    Accuracy_log: {accuracy_lr}")#0.4046278924327705
LogReg_precision,LogReg_recall,LogReg_f1_score,LogReg_support=score(Y_Test, y_pred_lr, average='macro')
print ('    LogReg - Precision : {}'.format(LogReg_precision))
print ('    LogReg - F1-score  : {}'.format(LogReg_f1_score))
#############################################################################
# Train the KNN model and measure training time
print("3-KNN")
knn = KNeighborsClassifier(n_neighbors=7,p=1,weights="distance")#0.4834271419637273
File_Name_KNN_Model = 'KNN.sav'

# Make predictions on the test set using the KNN model
start_time = time.time()
Load_KNN_Model = pickle.load(open(File_Name_KNN_Model, 'rb'))
y_pred_knn = Load_KNN_Model.predict(X_test)
# y_pred_knn = knn.predict(X_test)
testing_time_knn = time.time() - start_time
print("     tst_time:  ",testing_time_knn)

# Measure the classification accuracy of the KNN model
accuracy_knn = accuracy_score(Y_Test, y_pred_knn)
print("    Accuracy_KNN",accuracy_knn)
# Precision, F1-score
Knn_precision,Knn_recall,Knn_f1_score,Knn_support=score(Y_Test, y_pred_knn, average='macro')
print ('    KNN - Precision : {}'.format(Knn_precision))
print ('    KNN - F1-score  : {}'.format(Knn_f1_score))


print("Random forest classification:")
start_time = time.time()

File_Name_rfc_Model = 'RandomForestClassifier.sav'
Load_rfc_Model = pickle.load(open(File_Name_rfc_Model, 'rb'))
training_time_rf = time.time() - start_time

start_time = time.time()
rfc_predict = Load_rfc_Model.predict(X_test)
testing_time_rf = time.time() - start_time
accuracy_rf=accuracy_score(Y_Test, rfc_predict)
print('Accuracy random forest:', accuracy_rf)
rf_precision,rf_recall,rf_f1_score,rf_support=score(Y_Test, rfc_predict, average='macro')
print ('Random Forest - Precision : {}'.format(rf_precision))
print ('Random Forest - F1-score  : {}'.format(rf_f1_score))


######################################################
# Generate bar graphs to show the results
models = ['Logistic Regression', 'DecisionTreeClassifier','KNN']
accuracy_scores = [accuracy_lr, accuracy,accuracy_knn]
# training_times = [training_time_lr, training_time_tre,training_time_knn]
testing_times = [testing_time_lr, testing_time_Tre,testing_time_knn]

plt.bar(models, accuracy_scores)
plt.title('Classification Accuracy')
plt.show()

# plt.bar(models, training_times)
# plt.title('Total Training Time')
# plt.show()

plt.bar(models, testing_times)
plt.title('Total Test Time')
plt.show()





















# models = [
#     {'name': 'Model 1', 'algorithm': DecisionTreeClassifier()},
#     {'name': 'Model 2', 'algorithm': LogisticRegression()}
#     # {'name': 'Model 3', 'algorithm': SVC(kernel='linear')}
# ]
# #'Logistic Regression', 'SVM'
# # Train and evaluate each model or configuration
# accuracy_values = []
# training_times = []
# test_times = []
# for model in models:
#     start_time = time.time()
#     model['algorithm'].fit(X_train, Y_Train)
#     end_time = time.time()
#     training_time = end_time - start_time
#
#     start_time = time.time()
#     y_pred = model['algorithm'].predict(X_test)
#     end_time = time.time()
#     test_time = end_time - start_time
#
#     accuracy = accuracy_score(Y_Test, y_pred)
#
#     accuracy_values.append(accuracy)
#     training_times.append(training_time)
#     test_times.append(test_time)
#
# # Plot the classification accuracy bar graph
# plt.bar([m['name'] for m in models], accuracy_values)
# plt.title('Classification Accuracy')
# plt.xlabel('Models')
# plt.ylabel('Accuracy')
# plt.show()
#
# # Plot the total training time bar graph
# plt.bar([m['name'] for m in models], training_times)
# plt.title('Total Training Time')
# plt.xlabel('Models')
# plt.ylabel('Time (seconds)')
# plt.show()
#
# # Plot the total test time bar graph
# plt.bar([m['name'] for m in models], test_times)
# plt.title('Total Test Time')
# plt.xlabel('Models')
# plt.ylabel('Time (seconds)')
# plt.show()
"""
# WARNING: get the top features from the 2 output of the corr
# high_corr_features :  Index(['Sales', 'Quantity', 'Discount', 'Profit'], dtype='object')
# Selected categorical features:  ['Region', 'State', 'Main Category', 'City', 'Ship Mode']
X = Train_Data[['Sales', 'Quantity', 'Discount', 'Region', 'State', 'Main Category', 'City', 'Ship Mode']]

Y = Train_Data[['Profit']]

# train a random forest classifier
regressor = RandomForestRegressor()
regressor.fit(X, Y)

# select the top 5 features based on feature importance scores
sfm = SelectFromModel(regressor, threshold=-np.inf, max_features=3)
sfm.fit(X, Y)

# get the indices of the selected features
feature_indices = sfm.get_support(indices=True)

# get the names of the selected features
selected_features = [list(X.columns)[i] for i in feature_indices]

print('Selected features:', selected_features)
"""
# Create subplots
# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# def plts(i,Y_Prediction_mod,Y_Test,str):
#     axs[i].scatter(Y_Prediction_mod, Y_Test, color='gray')
#     axs[i].plot(Y_Test, Y_Test, color='blue', linestyle='--')
#     axs[i].set_xlabel('Predicted ReturnCategory')
#     axs[i].set_ylabel('Actual ReturnCategory')
#     axs[i].set_title(str)
# plts(0,Y_Prediction_Linear_Regression,Y_Test,'Linear Regression Prediction')
# plts(1,Y_Prediction,Y_Test,'Random Forest Regression Prediction')
# plts(2,Y_Prediction_For_Decision_Tree,Y_Test,'Decision Tree Regression Prediction')
# # Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.3)
# # Show the plot
# plt.show()
