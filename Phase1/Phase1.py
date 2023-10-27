import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
Scaler_mod=[]

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
# split CategoryTree coulmn into 2 coulmn main, sub
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
        Connect_Product_ID_With_Categories(Full_Dataset) #creating full category data to fill missing values
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

# def Feature_Scaling_Train(Dataset, Columns):
#     # Feature scaling
#     for i ,column in enumerate(Columns):
#         Is_It_Normal = True
#         stat, p = shapiro(list(Dataset.loc[:, column]))
#         if p <= 0.05:
#             Is_It_Normal = False
#         if Is_It_Normal:
#             Scaler_mod.append(StandardScaler())
#             Data = Dataset.loc[:, column].values.reshape(-1, 1)
#             Scaled_Data = Scaler_mod[i].fit_transform(Data)
#             # print(Scaled_Data)
#             Dataset.loc[:, column] = Scaled_Data
#         else:
#             Scaler_mod.append( MinMaxScaler())
#             Data = Dataset.loc[:, column].values.reshape(-1, 1)
#             Scaled_Data =Scaler_mod[i].fit_transform(Data)
#             # print(Scaled_Data)
#             Dataset.loc[:, column] = Scaled_Data
# def Feature_Scaling_Test(Dataset, Columns):
#     # Feature scaling
#     for i ,column in enumerate(Columns):
#         Data = Dataset.loc[:, column].values.reshape(-1, 1)
#         Scaled_Data = Scaler_mod[i].transform(Data)
#         # print(Scaled_Data)
#         Dataset.loc[:, column] = Scaled_Data
def Feature_Scaling_Train(Dataset, Columns):
    # Feature scaling
    for i, column in enumerate(Columns):
        Scaler_mod.append(MinMaxScaler())
        Data = Dataset.loc[:, column].values.reshape(-1, 1)
        Scaled_Data =Scaler_mod[i].fit_transform(Data)
        # print(Scaled_Data)
        Dataset.loc[:, column] = Scaled_Data
    with open('scaler_mod_phase1.pkl', 'wb') as f:
        pickle.dump(Scaler_mod, f)

def Feature_Scaling_Test(Dataset, Columns):
    # Feature scaling
    with open('scaler_mod_phase1.pkl', 'rb') as f:
        Scaler_mod1 = pickle.load(f)
    for i ,column in enumerate(Columns):
        Data = Dataset.loc[:, column].values.reshape(-1, 1)
        Scaled_Data = Scaler_mod[i].transform(Data)
        # print(Scaled_Data)
        Dataset.loc[:, column] = Scaled_Data

features_categorical={"Ship Mode":None,  "Segment":None, "City":None, "State":None, "Region":None,"Main Category":None, "Sub Category":None}

def feature_encoder_train(dataset):
    """
    Fit Action => train LabelEncoder models based on (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will train from
    """
    for feature_obj in features_categorical:
        lbl_model = LabelEncoder()
        list_vals = list(dataset[feature_obj].values)
        list_vals.append('Unknown')
        lbl_model.fit(list_vals)
        features_categorical[feature_obj] = lbl_model
    with open('features_categorical_phase1.pkl', 'wb') as f:
        pickle.dump(features_categorical, f)

def feature_encoder_transform(dataset):
    """
    Transform Action => apply the trained LabelEncoder models from (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will apply on
    """
    with open('features_categorical_phase1.pkl', 'rb') as f:
        features_categorical1 = pickle.load(f)
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

def Handle_Missing_Sub_Category(Missing_Dataset, Category):
    Missing_Rows = Missing_Dataset[Missing_Dataset['Sub Category'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Main_Category_Index = Missing_Dataset.columns.get_loc('Main Category')
    Product_ID_Index = Missing_Dataset.columns.get_loc('Product ID')
    Sub_Category_Index = Missing_Dataset.columns.get_loc('Sub Category')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        if not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Product_ID_Index]):
            Sub_Abbreviation = Missing_Rows.iloc[Index_Of_Missing, Product_ID_Index].split("-")[1]
            Found_Dataset = Category[Category["Sub Category ID"] == Sub_Abbreviation]
            Sub_Category_Found_Index = Found_Dataset.columns.get_loc('Sub Category')
            Missing_Dataset.iloc[Value_Of_Missing, Sub_Category_Index] = Found_Dataset.iloc[0, Sub_Category_Found_Index]
        else:
            if Missing_Dataset.iloc[Value_Of_Missing, Main_Category_Index] != "Unavailable Main Category":
                Main_Category = Missing_Dataset.iloc[Value_Of_Missing, Main_Category_Index]
                Found_Dataset = Category[Category["Main Category"] == Main_Category]
                Missing_Dataset.iloc[Value_Of_Missing, Sub_Category_Index] = Found_Dataset['Sub Category'].mode()
            else:
                Missing_Dataset.loc[Value_Of_Missing, ['Sub Category']] = Full_Dataset.mode()['Sub Category'][0]

def Handle_Missing_Main_Category(Dataset, Category):
    Missing_Rows = Dataset[Dataset['Main Category'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Product_ID_Index = Dataset.columns.get_loc('Product ID')
    Main_Category_Index = Dataset.columns.get_loc('Main Category')
    Sub_Category_Index = Dataset.columns.get_loc('Sub Category')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        if not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Product_ID_Index]):
            Main_Abbreviation = Missing_Rows.iloc[Index_Of_Missing, Product_ID_Index].split("-")[0]
            Found_Dataset = Category[Category["Main Category ID"] == Main_Abbreviation]
            Main_Category_Index_Found = Found_Dataset.columns.get_loc('Main Category')
            Dataset.iloc[Value_Of_Missing, Main_Category_Index] = Found_Dataset.iloc[0, Main_Category_Index_Found]
        else:
            if not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Sub_Category_Index]):
                Sub_Category = Missing_Rows.iloc[Index_Of_Missing, Sub_Category_Index]
                Found_Dataset = Category[Category["Sub Category"] == Sub_Category]
                Main_Category_In_Categories = Found_Dataset.columns.get_loc('Main Category')
                Dataset.iloc[Value_Of_Missing, Main_Category_Index] = Found_Dataset.iloc[1, Main_Category_In_Categories]
            else:
                Dataset.loc[Value_Of_Missing, ['Main Category']] = Full_Dataset.mode()['Main Category'][0]

def Handle_Missing_Region(Missing_Dataset, Full_Dataset):
    """
    Fill out empty regions
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Region'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Postal_Code_Index = Missing_Dataset.columns.get_loc('Postal Code')
    Region_Index = Missing_Dataset.columns.get_loc('Region')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Found_Value = Full_Dataset[(Full_Dataset["Postal Code"] == Missing_Dataset.iloc[Value_Of_Missing,
                                                                                        Postal_Code_Index])]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                Missing_Dataset.iloc[Value_Of_Missing, Region_Index] = Full_Dataset.iloc[Value_Of_Found, Region_Index]
                break
        else:
            Missing_Dataset.iloc[Value_Of_Missing, Region_Index] = Full_Dataset.mode()['Region'][0]

def Handle_Missing_State(Missing_Dataset, Full_Dataset):
    """
    Fill out empty states
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['State'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Postal_Code_Index = Missing_Dataset.columns.get_loc('Postal Code')
    State_Index = Missing_Dataset.columns.get_loc('State')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Found_Value = Full_Dataset[(Full_Dataset["Postal Code"] == Missing_Dataset.iloc[Value_Of_Missing,
                                                                                        Postal_Code_Index])]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                Missing_Dataset.iloc[Value_Of_Missing, State_Index] = Full_Dataset.iloc[Value_Of_Found, State_Index]
                break
        else:
            Missing_Dataset.iloc[Value_Of_Missing, State_Index] = Full_Dataset.mode()['State'][0]

def Handle_Missing_City(Missing_Dataset, Full_Dataset):
    """
    Fill out empty cities
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['City'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Postal_Code_Index = Missing_Dataset.columns.get_loc('Postal Code')
    City_Index = Missing_Dataset.columns.get_loc('City')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Found_Value = Full_Dataset[(Full_Dataset["Postal Code"] == Missing_Dataset.iloc[Value_Of_Missing,
                                                                                        Postal_Code_Index])]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                Missing_Dataset.iloc[Value_Of_Missing, City_Index] = Full_Dataset.iloc[Value_Of_Found, City_Index]
                break
        else:
            Missing_Dataset.iloc[Value_Of_Missing, City_Index] = Full_Dataset.mode()['City'][0]

def Handle_Missing_Postal_Code(Missing_Dataset, Full_Dataset):
    """
    Fill out empty Postal Codes
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Postal Code'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    City_Index = Missing_Dataset.columns.get_loc('City')
    State_Index = Missing_Dataset.columns.get_loc('State')
    Region_Index = Missing_Dataset.columns.get_loc('Region')
    Postal_Code_Index = Missing_Dataset.columns.get_loc('Postal Code')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Flag = False
        Found_Value = Full_Dataset[(Full_Dataset["City"] == Missing_Dataset.iloc[Value_Of_Missing, City_Index]) &
                                   (Full_Dataset["State"] == Missing_Dataset.iloc[Value_Of_Missing, State_Index]) &
                                   (Full_Dataset["Region"] == Missing_Dataset.iloc[Value_Of_Missing, Region_Index])]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                if Value_Of_Found not in Missing_Rows_Indices and not pd.isnull(Found_Value.iloc[Index_Of_Found,
                Postal_Code_Index]):
                    Missing_Dataset.iloc[Value_Of_Missing, Postal_Code_Index] = Full_Dataset.iloc[Value_Of_Found,
                    Postal_Code_Index]
                    Flag = True
                    break
        if Flag is False and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, City_Index]) \
                         and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, State_Index])\
                         and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Region_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Postal_Code_Index] = Full_Dataset.mode()['Postal Code'][0]
        elif Flag is False and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, City_Index])\
                           and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, State_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Postal_Code_Index] = Full_Dataset[Full_Dataset['Region'] ==
            Missing_Dataset.iloc[Value_Of_Missing, Region_Index]].mode()['Postal Code'][0]
        elif Flag is False and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, City_Index]) \
                           and pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Region_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Postal_Code_Index] = Full_Dataset[Full_Dataset['State'] ==
            Missing_Dataset.iloc[Value_Of_Missing, State_Index]].mode()['Postal Code'][0]
        else:
            Missing_Dataset.iloc[Value_Of_Missing, Postal_Code_Index] = Full_Dataset[Full_Dataset['City'] ==
            Missing_Dataset.iloc[Value_Of_Missing, City_Index]].mode()['Postal Code'][0]

def Handle_Missing_Order_Date(Missing_Dataset):
    """
    Fill out empty Order Dates
    :param Missing_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Order Date'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Ship_Mode_Index = Missing_Dataset.columns.get_loc('Ship Mode')
    Order_Date_Index = Missing_Dataset.columns.get_loc('Order Date')
    Ship_Date_Index = Missing_Dataset.columns.get_loc('Ship Date')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        if pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Ship_Date_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Date_Index] = pd.to_datetime('today').date()
        if re.search("Same Day", Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Order_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing,
            Ship_Date_Index]
        elif re.search("First Class", Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Order_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing,
            Ship_Date_Index] - pd.Timedelta(days=2)
        elif re.search("Second Class", Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Order_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing,
            Ship_Date_Index] - pd.Timedelta(days=3)
        else:
            Missing_Dataset.iloc[Value_Of_Missing, Order_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing,
            Ship_Date_Index] - pd.Timedelta(days=4)

def Handle_Missing_Ship_Date(Missing_Dataset):
    """
    Fill out empty Ship Dates
    :param Missing_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Ship Date'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Ship_Mode_Index = Missing_Dataset.columns.get_loc('Ship Mode')
    Order_Date_Index = Missing_Dataset.columns.get_loc('Order Date')
    Ship_Date_Index = Missing_Dataset.columns.get_loc('Ship Date')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        if pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Order_Date_Index]):
            Missing_Dataset.iloc[Value_Of_Missing, Order_Date_Index] = pd.to_datetime('today').date()
        if Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index] == "Same Day":
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing, 2]
        elif Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index] == "First Class":
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing, 2] + \
                                                                      pd.Timedelta(days=2)
        elif Missing_Rows.iloc[Index_Of_Missing, Ship_Mode_Index] == "Second Class":
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing, 2] + \
                                                                      pd.Timedelta(days=3)
        else:
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Date_Index] = Missing_Dataset.iloc[Value_Of_Missing, 2] + \
                                                                      pd.Timedelta(days=4)

def Handle_Missing_Ship_Mode(Missing_Dataset, Full_Dataset):
    """
    Fill out empty Ship Modes
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Ship Mode'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Ship_Mode_Index = Missing_Dataset.columns.get_loc('Ship Mode')
    Order_Date_Index = Missing_Dataset.columns.get_loc('Order Date')
    Ship_Date_Index = Missing_Dataset.columns.get_loc('Ship Date')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        if not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, Ship_Date_Index]) and not pd.isnull(
                Missing_Rows.iloc[Index_Of_Missing, Order_Date_Index]):
            Difference = (Missing_Rows.iloc[Index_Of_Missing, Ship_Date_Index] -
                          Missing_Rows.iloc[Index_Of_Missing, Order_Date_Index]).days
            if Difference == 0:
                Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "Same Day"
            elif 1 <= Difference < 2:
                Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "First Class"
            elif 2 <= Difference <= 3:
                choice = np.random.randint(2, size=1)
                if Difference == 2:
                    if choice == 0:
                        Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "Second Class"
                    else:
                        Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "First Class"
                else:
                    choice = np.random.randint(2, size=1)
                    if choice == 0:
                        Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "Standard Class"
                    else:
                        Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "Second Class"
            else:
                Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = "Standard Class"
        else:
            Missing_Dataset.iloc[Value_Of_Missing, Ship_Mode_Index] = Full_Dataset.mode()['Ship Mode'][0]

def Handle_Missing_Segment(Missing_Dataset, Full_Dataset):
    Missing_Rows = Missing_Dataset[Missing_Dataset['Segment'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Customer_ID_Index = Missing_Dataset.columns.get_loc('Customer ID')
    Segment_Index = Missing_Dataset.columns.get_loc('Segment')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Flag = False
        Found_Value = Full_Dataset[Full_Dataset["Customer ID"] == Missing_Dataset.iloc[Value_Of_Missing,
                                                                                       Customer_ID_Index]]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                if Value_Of_Found not in Missing_Rows_Indices and not pd.isnull(Found_Value.iloc[Index_Of_Found,
                Customer_ID_Index]) and not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, 5]):
                    Missing_Dataset.iloc[Value_Of_Missing, Segment_Index] = \
                        Full_Dataset.iloc[Value_Of_Found, Segment_Index]
                    Flag = True
                    break
        if Flag is False:
            Missing_Dataset.iloc[Value_Of_Missing, 6] = Full_Dataset.mode()['Segment'][0]

def Handle_Missing_Customer_Name(Missing_Dataset, Full_Dataset):
    """
    Fill out empty Customer names
    :param Missing_Dataset:
    :param Full_Dataset:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Customer Name'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Customer_ID_Index = Missing_Dataset.columns.get_loc('Customer ID')
    Customer_Name_Index = Missing_Dataset.columns.get_loc('Customer Name')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Flag = False
        Found_Value = Full_Dataset[Full_Dataset["Customer ID"] == Missing_Dataset.iloc[Value_Of_Missing,
                                                                                       Customer_ID_Index]]
        Found_Value_Indices = list(Found_Value.index)
        if Found_Value_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Value_Indices):
                if Value_Of_Found not in Missing_Rows_Indices and not pd.isnull(Found_Value.iloc[Index_Of_Found,
                Customer_ID_Index]) and not pd.isnull(Missing_Rows.iloc[Index_Of_Missing, 5]):
                    Missing_Dataset.iloc[Value_Of_Missing, Customer_Name_Index] = \
                        Full_Dataset.iloc[Value_Of_Found, Customer_Name_Index]
                    Flag = True
                    break
        if Flag is False:
            Missing_Dataset.iloc[Value_Of_Missing, 6] = Full_Dataset.mode()['Customer Name'][0]

def Handle_Missing_Order_ID(Missing_Dataset, Order_ID_Dict):
    """
    Fill out empty Order IDs
    :param Missing_Dataset:
    :param Order_ID_Dict:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset["Order ID"].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        New_ID = int(max(Order_ID_Dict.values())) + 1
        Order_ID_Dict[Value_Of_Missing] = New_ID
    return Order_ID_Dict

def Handle_Missing_Product_ID(Missing_Dataset, Product_ID_Dict):
    """
    Fill out empty Product IDs
    :param Missing_Dataset:
    :param Product_ID_Dict:
    :return:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset["Product ID"].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        New_ID = int(max(Product_ID_Dict.values())) + 1
        Product_ID_Dict[Value_Of_Missing] = New_ID
    return Product_ID_Dict

def Handle_Missing_Customer_ID(Missing_Dataset, Full_Dataset, Customer_ID_Dict):
    """
    Fill out empty Customer IDs
    :param Missing_Dataset:
    :param Full_Dataset:
    :param Customer_ID_Dict:
    :return Customer_ID_Dict:
    """
    Missing_Rows = Missing_Dataset[Missing_Dataset['Customer ID'].isna()]
    Missing_Rows_Indices = list(Missing_Rows.index)
    Customer_Name_Index = Missing_Dataset.columns.get_loc('Customer Name')
    Customer_ID_Index = Missing_Dataset.columns.get_loc('Customer ID')
    for Index_Of_Missing, Value_Of_Missing in enumerate(Missing_Rows_Indices):
        Flag = False
        Found_Rows = Full_Dataset[Full_Dataset["Customer Name"] == Missing_Dataset.iloc[Index_Of_Missing,
        Customer_Name_Index]]
        Found_Rows_Indices = list(Found_Rows.index)
        if Found_Rows_Indices:
            for Index_Of_Found, Value_Of_Found in enumerate(Found_Rows_Indices):
                if Value_Of_Found not in Missing_Rows_Indices and Found_Rows.iloc[Index_Of_Found, Customer_ID_Index] != \
                        np.NaN and Found_Rows.iloc[Index_Of_Found, 6] != np.NaN:
                    Customer_ID_Dict[Value_Of_Missing] = Customer_ID_Dict[Value_Of_Found]
                    Flag = True
                    break
        if Flag is False:
            New_ID = int(max(Customer_ID_Dict.values())) + 1
            Customer_ID_Dict[Value_Of_Missing] = New_ID
    return Customer_ID_Dict

def Handle_Missing_Values(Test_Dataset, Full_Dataset, Category, Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict):
    """
    Check for missing values and how to handle them

    :param Test_Dataset:
    :param Full_Dataset:
    :param Category:
    :param Order_ID_Dict:
    :param Customer_ID_Dict:
    :param Product_ID_Dict:
    :return:
    """
    print("Product Name: ", Test_Dataset.mode()["Product Name"][0])
    print("Customer Name: ", Test_Dataset.mode()["Customer Name"][0])
    print("Customer ID: ", Test_Dataset.mode()["Customer ID"][0])
    print("Main Category: ", Test_Dataset.mode()["Main Category"][0])
    print("Sub Category: ", Test_Dataset.mode()["Sub Category"][0])
    print("Product ID: ", Test_Dataset.mode()["Product ID"][0])
    print("Segment: ", Test_Dataset.mode()["Segment"][0])
    print("Ship Mode: ", Test_Dataset.mode()["Ship Mode"][0])
    print("Order ID: ", Test_Dataset.mode()["Order ID"][0])
    print("Postal Code: ", Test_Dataset.mode()["Postal Code"][0])
    print("State: ", Test_Dataset.mode()["State"][0])
    print("City: ", Test_Dataset.mode()["City"][0])
    print("Region: ", Test_Dataset.mode()["Region"][0])
    print("Order Date: ", Test_Dataset.mode()["Order Date"][0])
    print("Ship Date: ", Test_Dataset.mode()["Ship Date"][0])
    print("Sales: ", Test_Dataset["Sales"].median())
    print("Discount: ", Test_Dataset["Discount"].median())
    print("Quantity: ", Test_Dataset["Quantity"].median())

    if Test_Dataset["Profit"].isna().any():
        Test_Dataset["Profit"] = Test_Dataset["Profit"].dropna(how='any')
    if Test_Dataset["Discount"].isna().any():
        # Dataset["Discount"].fillna(Dataset["Discount"].mean(), inplace=True)
        Test_Dataset["Discount"].fillna(0.155729, inplace=True)

    if Test_Dataset["Sales"].isna().any():
        # Dataset["Sales"].fillna(Dataset["Sales"].mean(), inplace=True)
        Test_Dataset["Sales"].fillna(233.382151, inplace=True)

    if Test_Dataset["Quantity"].isna().any():
        # Dataset["Quantity"].fillna(Dataset.mode()['Quantity'][0], inplace=True)
        Test_Dataset["Quantity"].fillna(3, inplace=True)

    if Test_Dataset["Country"].isna().any():
        Test_Dataset["Quantity"].fillna("United States")

    if Test_Dataset["Product Name"].isna().any():
        Test_Dataset["Product Name"].fillna("Unavailable Product Name", inplace=True)

    if Test_Dataset["Main Category"].isna().any():
        Handle_Missing_Main_Category(Test_Dataset, Category)
    if Test_Dataset["Sub Category"].isna().any():
        Handle_Missing_Sub_Category(Test_Dataset, Category)
    if Test_Dataset["Product ID"].isna().any():
        Product_ID_Dict = Handle_Missing_Product_ID(Test_Dataset, Product_ID_Dict)
    Test_Dataset["Product ID"] = dict(sorted(Product_ID_Dict.items())).values()

    if Test_Dataset["Segment"].isna().any():
        # Dataset["Segment"].fillna(Dataset.mode()['Segment'][0], inplace=True)
        Handle_Missing_Segment(Test_Dataset, Full_Dataset)

    if Test_Dataset["Customer Name"].isna().any():
        Handle_Missing_Customer_Name(Test_Dataset, Full_Dataset)

    if Test_Dataset["Customer ID"].isna().any():
        Customer_ID_Dict = Handle_Missing_Customer_ID(Test_Dataset, Customer_ID_Dict)
    Test_Dataset["Customer ID"] = dict(sorted(Customer_ID_Dict.items())).values()

    if Test_Dataset["Ship Mode"].isna().any():
        Handle_Missing_Ship_Mode(Test_Dataset, Full_Dataset)
    if Test_Dataset["Order Date"].isna().any():
        Handle_Missing_Order_Date(Test_Dataset)
    if Test_Dataset["Ship Date"].isna().any():
        Handle_Missing_Ship_Date(Test_Dataset)

    if Test_Dataset["Postal Code"].isna().any():
        Handle_Missing_Postal_Code(Test_Dataset, Full_Dataset)
    if Test_Dataset["City"].isna().any():
        Handle_Missing_City(Test_Dataset, Full_Dataset)
    if Test_Dataset["State"].isna().any():
        Handle_Missing_State(Test_Dataset, Full_Dataset)
    if Test_Dataset["Region"].isna().any():
        Handle_Missing_Region(Test_Dataset, Full_Dataset)
    if Test_Dataset["Row ID"].isna().any():
        Test_Dataset["Row ID"].fillna(5007.5, inplace=True)

    if Test_Dataset["Order ID"].isna().any():
        Order_ID_Dict = Handle_Missing_Order_ID(Test_Dataset, Order_ID_Dict)
    Test_Dataset["Order ID"] = dict(sorted(Order_ID_Dict.items())).values()

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

def Numerical_Correlation(Numerical_Dataframe): #persona
    corr_matrix = Numerical_Dataframe.corr()
    relevant_features = corr_matrix['Profit'].abs().sort_values(ascending=False)[:5]
    corr_subset = Numerical_Dataframe[relevant_features.index].corr()
    # Identify the features with high correlation with the target variable 0.5=> sales
    high_corr_features = list(corr_matrix.index[abs(corr_matrix['Profit']) > 0.05])
    # Print the highly correlated features
    print("high_corr_features : ", high_corr_features) # if need to know  what name of this feature
    # Create heatmap plot of the correlation matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Top {} Features and Profit'.format(5))
    plt.show()

def Categorical_Correlation(Categorical_Dataframe, Y_Train):
    correlations = [kendalltau(Categorical_Dataframe.iloc[:, i], Y_Train)[0] for i in range(Categorical_Dataframe.shape[1])]
    # Sort features by the absolute value of their correlation coefficient
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    # Select the top 5 features with highest correlation coefficients
    selected_features = Categorical_Dataframe.columns[sorted_indices[:5]]
    # Display the selected features
    print("Selected categorical features: ", selected_features.tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(Categorical_Dataframe.columns[sorted_indices], np.array(correlations)[sorted_indices], color='b')
    ax.set_xticklabels(Categorical_Dataframe.columns[sorted_indices], rotation=45, ha='right')
    ax.set_xlabel('Categorical Feature')
    ax.set_ylabel("Kendall's Rank Correlation Coefficient")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

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
    # df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.set_index('Order Date', inplace=True)
    monthly_profit = df['Profit'].resample('M').sum()
    # Plot the monthly profit values over time
    plt.plot(monthly_profit)
    plt.title('Monthly Profit')
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.show()
    # Decompose the monthly profit values into trend, seasonal, and residual components
    decomposition = seasonal_decompose(monthly_profit, model='additive', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # Plot the decomposed components
    plt.subplot(411)
    plt.plot(monthly_profit, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    # Fit an ARIMA model to the monthly profit values
    model = ARIMA(monthly_profit, order=(1, 1, 1))
    results = model.fit()
    # Use the fitted model to forecast future profit values
    forecast = results.forecast(12)
    # Plot the actual monthly profit values and the forecasted values
    plt.plot(monthly_profit, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title('Monthly Profit Forecast')
    plt.xlabel('Month')
    plt.ylabel('Profit')
    plt.legend(loc='best')
    plt.show()


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
def preprocessing_train(df,columns):
    Numerical_Dataframe = df[columns['Numerical_Columns']]
    missing_val = df.isnull().sum()
    print("Null Values in Train : ",missing_val)
    print("Null Values in Train : ",missing_val[missing_val==True].index.tolist())
    boxplot(Numerical_Dataframe, columns['Numerical_Columns'])
    # df=Handle_Outliers(df,columns)
    # df=Remove_Outliers(df,columns)
    print("#"*100)
    print("Num Of Outliers In Train : ")
    df=Detect_Outliers(df,columns)
    print("#"*100)

    Numerical_Dataframe = df[columns['Numerical_Columns']]
    # boxplot(Numerical_Dataframe, columns['Numerical_Columns'])

    df = Check_Duplicates(df)
    feature_encoder_train(df)
    feature_encoder_transform(df)

    Numerical_Dataframe = df[columns['Numerical_Columns']]
    Categorical_Dataframe = df[columns['Categorical_Columns']]

    Feature_Scaling_Train(df, Numerical_Dataframe)
    Numerical_Correlation(Numerical_Dataframe)
    Categorical_Correlation(Categorical_Dataframe, df['Profit'])
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
Train_Data, Test_Data = train_test_split(Test_Dataset, test_size=0.2, shuffle=True, random_state=42)
# Test_Data=pd.read_csv('m1.csv')

Train_Data=Data_Compare(Train_Data,Full_Dataset,Split_Columns,"Train_Data")
Test_Data=Data_Compare(Test_Data,Full_Dataset,Split_Columns,"Test_Data")

Train_Data = preprocessing_train(Train_Data, Split_Columns)
Test_Data= Preprocess_Test(Test_Data, Split_Columns)
Time_Series_Analysis(Train_Data)

'''X_train =train_data[['Sales', 'Discount', 'Region', 'Quantity']]
Y_Train=train_data['Profit']
X_test=test_data[['Sales', 'Discount', 'Region', 'Quantity']]
Y_Test=test_data['Profit']'''
X_train = Train_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region',
                      'Main Category', 'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Train = Train_Data['Profit']
X_test = Test_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region', 'Main Category',
                    'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Test = Test_Data['Profit']

# Selected features: ['Sales', 'Quantity', 'Discount', 'Main Category', 'City']

# high_corr_features :  Index(['Sales', 'Quantity', 'Discount', 'Profit'], dtype='object')
# Selected categorical features:  ['Region', 'State', 'Main Category', 'City', 'Ship Mode']
###################################################################
#####################################################################################
print('1- Polynomial regression')
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
with open('poly.pkl', 'wb') as f:
    pickle.dump(poly, f)
X_test_poly = poly.transform(X_test)


Linear_Regression_Model = LinearRegression()
Linear_Regression_Model.fit(X_train_poly, Y_Train)
File_Name_poly_model = 'poly_model.sav'
pickle.dump(Linear_Regression_Model, open(File_Name_poly_model, 'wb'))
# Load_XGBRegressor_model = pickle.load(open(File_Name_XGBRegressor_model, 'rb'))
# Predict values for test set
Y_Prediction_Linear_Regression = Linear_Regression_Model.predict(X_test_poly)

# Evaluate model performance on test set
Linear_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction_Linear_Regression)
print("Linear Regression MSE: ", Linear_Regression_MSE)

r2 = r2_score(Y_Test, Y_Prediction_Linear_Regression)
print("R-squared score Y_T&Y_Prediction:", r2)

r1 = Linear_Regression_Model.score(X_test_poly, Y_Test)
print("LR-squared score X_Test & Y_test:", r1)

r3 = Linear_Regression_Model.score(X_train_poly, Y_Train)
print("LR-squared score X_train &Y_Train:", r3)

# Evaluate model accuracy through cross-validation
scores = cross_val_score(Linear_Regression_Model, X_train_poly, Y_Train, cv=5)# mean accuracy is high but the standard deviation is also high(overfitting)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#####################################################################################
# print('1- linear regression')
# # Fit a linear regression model
# Linear_Regression_Model = LinearRegression()
# Linear_Regression_Model.fit(X_train, Y_Train)
# File_Name_Linear_Regression_Model = 'linear_reg_model.sav'
# pickle.dump(Linear_Regression_Model,open(File_Name_Linear_Regression_Model, 'wb'))
# # Load_Linear_Regression_Model = pickle.load(open(File_Name_Linear_Regression_Model, 'rb'))
# Y_Prediction_Linear_Regression = Linear_Regression_Model.predict(X_test)
# scores = cross_val_score(Linear_Regression_Model, X_train, Y_Train, cv=5)
# # Print the mean accuracy and standard deviation of the scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)
#
# # Y_Prediction_Linear_Regression = Load_Linear_Regression_Model.predict(X_test)
# Linear_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction_Linear_Regression)
# print("Linear Regression MSE: ", Linear_Regression_MSE)
# r2 = r2_score(Y_Test, Y_Prediction_Linear_Regression)
# print("R-squared score Y_T&Y_Prediction:", r2)
# r1 = Linear_Regression_Model.score(X_test, Y_Test)
# print("LR-squared score X_Test & Y_test:", r1)
# r3 = Linear_Regression_Model.score(X_train, Y_Train)
# print("LR-squared score X_train &Y_Train:", r3)
# overfit(Linear_Regression_Model, X_train, Y_Train)

# plt.scatter(X_test, Y_Test, color='gray')
# plt.plot(X_test, Y_Prediction_Linear_Regression, color='red', linewidth=2)
# plt.xlabel('Input Feature')
# plt.ylabel('Target Variable')
# plt.title('Linear Regression Prediction')
# plt.show()

###############################################################

###########################
print()
print('2- Random Forest Regressor')
Random_Forest_Regression_Model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=123)
Random_Forest_Regression_Model.fit(X_train, Y_Train)
File_Name_RFR_Model = 'RandomForestRegressor_model.sav'
pickle.dump(Random_Forest_Regression_Model, open(File_Name_RFR_Model, 'wb'))
# Load_RFR_Model = pickle.load(open(File_Name_RFR_Model, 'rb'))
# Y_Prediction = Load_RFR_Model.predict(X_test)
Y_Prediction = Random_Forest_Regression_Model.predict(X_test)
scores = cross_val_score(Random_Forest_Regression_Model, X_train, Y_Train, cv=5)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)

# Evaluate the Random_Forest_Regression_Model on the testing data
Random_Forest_Regression_MSE = mean_squared_error(Y_Test, Y_Prediction)
print("Random Forest Regression MSE:", Random_Forest_Regression_MSE)
# plt.scatter(Y_Test, Y_Prediction)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.show()
r1 = Random_Forest_Regression_Model.score(X_test, Y_Test)
print("LR-squared score X_Test & Y_test:", r1)
r2 = r2_score(Y_Test, Y_Prediction)
print("R-squared score Y_T&Y_Prediction:", r2)
r3 = Random_Forest_Regression_Model.score(X_train, Y_Train)
print("LR-squared score X_train &Y_Train:", r3)
# overfit(Random_Forest_Regression_Model, X_train, Y_Train)
#######################################################
print('3- XGBRegressor Algorithm')
XGBRegressor_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

# Train the model on the training data
XGBRegressor_model.fit(X_train, Y_Train)
File_Name_XGBRegressor_model = 'XGBRegressor_model.sav'
pickle.dump(XGBRegressor_model, open(File_Name_XGBRegressor_model, 'wb'))
# Load_XGBRegressor_model = pickle.load(open(File_Name_XGBRegressor_model, 'rb'))
# Make predictions on the test data
Y_Prediction_For_XGBRegressor = XGBRegressor_model.predict(X_test)
scores = cross_val_score(XGBRegressor_model, X_train, Y_Train, cv=10)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)
XGBRegressor_MSE = mean_squared_error(Y_Test, Y_Prediction_For_XGBRegressor)
print("XGBRegressor_model MSE : ", XGBRegressor_MSE)

r1 = XGBRegressor_model.score(X_test, Y_Test)
print("LR-squared score X_Test & Y_test:", r1)
r2 = r2_score(Y_Test, Y_Prediction_For_XGBRegressor)
print("R-squared score Y_T&Y_Prediction:", r2)
r3 = XGBRegressor_model.score(X_train, Y_Train)
print("LR-squared score X_train &Y_Train:", r3)
# overfit(XGBRegressor_model, X_train, Y_Train)
 ######################################################
# TODO  encode fun to efect in data set
# print('3- Decision tree Algorithm')
# # max_depth:= level of the tree
# Decision_Tree_Regressor = DecisionTreeRegressor()
# Decision_Tree_Regressor.fit(X_train, Y_Train)
# File_Name_Decision_Tree_Model = 'Decision tree_model.sav'
# pickle.dump(Decision_Tree_Regressor, open(File_Name_Decision_Tree_Model, 'wb'))
# # Load_Decision_Tree_Model = pickle.load(open(File_Name_Decision_Tree_Model, 'rb'))
# Y_Prediction_For_Decision_Tree = Decision_Tree_Regressor.predict(X_test)
# scores = cross_val_score(Decision_Tree_Regressor, X_train, Y_Train, cv=10)
# # Print the mean accuracy and standard deviation of the scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)
#
# # Y_Prediction_For_Decision_Tree = Load_Decision_Tree_Model.predict(X_test)
#
# Decision_Tree_MSE = mean_squared_error(Y_Test, Y_Prediction_For_Decision_Tree)
# print("Decision tree MSE : ", Decision_Tree_MSE)
#
# r1 = Decision_Tree_Regressor.score(X_test, Y_Test)
# print("LR-squared score X_Test & Y_test:", r1)
# r2 = r2_score(Y_Test, Y_Prediction_For_Decision_Tree)
# print("R-squared score Y_T&Y_Prediction:", r2)
# r3 = Decision_Tree_Regressor.score(X_train, Y_Train)
# print("LR-squared score X_train &Y_Train:", r3)
# overfit(Decision_Tree_Regressor, X_train, Y_Train)

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
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
def plts(i,Y_Prediction_mod,Y_Test,str):
    axs[i].scatter(Y_Prediction_mod, Y_Test, color='gray')
    axs[i].plot(Y_Test, Y_Test, color='blue', linestyle='--')
    axs[i].set_xlabel('Predicted Profit')
    axs[i].set_ylabel('Actual Profit')
    axs[i].set_title(str)
plts(0,Y_Prediction_Linear_Regression,Y_Test,'Polynomial Regression Prediction')
plts(1,Y_Prediction,Y_Test,'Random Forest Regression Prediction')
plts(2,Y_Prediction_For_XGBRegressor,Y_Test,'XGBRegressor Prediction')
# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)
# Show the plot
plt.show()
