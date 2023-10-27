import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from seaborn import heatmap
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import *
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kendalltau, shapiro
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
import os
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, SMOTEN
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
"""                                               Preprocessing                                                      """
Scaler_mod=[]
# ctrl+shift+- to close all functions

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

def Data_Generalization(Test_Dataset, Full_Dataset, Columns):
    Read_CategoryTree(Full_Dataset)
    isExisting = os.path.exists('Categories.csv')
    if not isExisting:
        Connect_Product_ID_With_Categories(Full_Dataset)
    Categories = pd.read_csv('Categories.csv')
    Read_CategoryTree(Test_Dataset)
    Test_Dataset["Order Date"] = pd.to_datetime(Test_Dataset["Order Date"])
    Test_Dataset["Ship Date"] = pd.to_datetime(Test_Dataset["Ship Date"])
    #Handle_Missing_Values(Test_Dataset, Full_Dataset, Categories, Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict) # mohamed
    Handle_Missing_Values_In_Test(Test_Dataset)
    Convert_Data_Types(Test_Dataset, Columns)
    Test_Dataset.drop('Country', axis=1, inplace=True)
    #Test_Dataset.drop('Product Name', axis=1, inplace=True)
    return Test_Dataset

def Show_Data(Dataset):
    print(Dataset.info())
    print(Dataset.head())
    print(Dataset.describe())
####################################################################################################################################
"""                                               Encoding,Scaling,Duplicates                                                     """
####################################################################################################################################

def Feature_Scaling_Train(Dataset, Columns):
    # Feature scaling
    for i, column in enumerate(Columns):
        Scaler_mod.append(MinMaxScaler())
        Data = Dataset.loc[:, column].values.reshape(-1, 1)
        Scaled_Data =Scaler_mod[i].fit_transform(Data)
        # print(Scaled_Data)
        Dataset.loc[:, column] = Scaled_Data
    with open('scaler_mod.pkl', 'wb') as f:
        pickle.dump(Scaler_mod, f)

def Feature_Scaling_Test(Dataset, Columns):
    # Feature scaling
    with open('scaler_mod.pkl', 'rb') as f:
        Scaler_mod1 = pickle.load(f)
    for i ,column in enumerate(Columns):
        Data = Dataset.loc[:, column].values.reshape(-1, 1)
        Scaled_Data = Scaler_mod[i].transform(Data)
        # print(Scaled_Data)
        Dataset.loc[:, column] = Scaled_Data

# features_categorical={"Ship Mode":None,  "Segment":None, "City":None, "State":None, "Region":None,"Main Category":None, "Sub Category":None,"ReturnCategory":None}
features_categorical = {"Ship Mode": None,  "Segment": None, "City": None, "State": None, "Region": None, "Customer Name": None, "Product Name": None, "Main Category": None, "Sub Category": None, "ReturnCategory": None}

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
    with open('features_categorical.pkl', 'wb') as f:
        pickle.dump(features_categorical, f)

def feature_encoder_transform(dataset):
    """
    Transform Action => apply the trained LabelEncoder models from (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will apply on
    """
    with open('features_categorical.pkl', 'rb') as f:
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

#######################################################################################################################################
"""                                                    Data Handling                                                """
#######################################################################################################################################

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

def Handle_Missing_Values_In_Test(Test_Dataset):
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
    print("Row ID: ", Test_Dataset["Row ID"].median())

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

    # if Test_Dataset["ReturnCategory"].isna().any():
    #     Test_Dataset["ReturnCategory"] = Test_Dataset["ReturnCategory"].dropna(how='any')
    #
    # if Test_Dataset["Discount"].isna().any():
    #     Test_Dataset["Discount"].fillna(0.2, inplace=True)
    #
    # if Test_Dataset["Sales"].isna().any():
    #     Test_Dataset["Sales"].fillna(54.792, inplace=True)
    #
    # if Test_Dataset["Quantity"].isna().any():
    #     Test_Dataset["Quantity"].fillna(3, inplace=True)
    #
    # if Test_Dataset["Country"].isna().any():
    #     Test_Dataset["Country"].fillna("United States", inplace=True)
    #
    # if Test_Dataset["Product Name"].isna().any():
    #     Test_Dataset["Product Name"].fillna("Staples", inplace=True)
    #
    # if Test_Dataset["Main Category"].isna().any():
    #     Test_Dataset["Main Category"].fillna("Office Supplies", inplace=True)
    #
    # if Test_Dataset["Sub Category"].isna().any():
    #     Test_Dataset["Sub Category"].fillna("Binders", inplace=True)
    #
    # if Test_Dataset["Product ID"].isna().any():
    #     Test_Dataset["Product ID"].fillna("TEC-AC-10003832", inplace=True)
    #
    # if Test_Dataset["Order ID"].isna().any():
    #     Test_Dataset["Order ID"].fillna("CA-2017-100111", inplace=True)
    #
    # if Test_Dataset["Customer ID"].isna().any():
    #     Test_Dataset["Customer ID"].fillna("PP-18955", inplace=True)
    #
    # Order_ID_Dict, Customer_ID_Dict, Product_ID_Dict = Split_And_Retrieve_IDs(Test_Dataset)
    # Test_Dataset["Order ID"] = dict(sorted(Order_ID_Dict.items())).values()
    # Test_Dataset["Customer ID"] = dict(sorted(Customer_ID_Dict.items())).values()
    # Test_Dataset["Product ID"] = dict(sorted(Product_ID_Dict.items())).values()
    #
    # if Test_Dataset["Segment"].isna().any():
    #     Test_Dataset["Segment"].fillna("Consumer", inplace=True)
    #
    # if Test_Dataset["Customer Name"].isna().any():
    #     Test_Dataset["Customer Name"].fillna("Paul Prost", inplace=True)
    #
    # if Test_Dataset["Ship Mode"].isna().any():
    #     Test_Dataset["Ship Mode"].fillna("Standard Class", inplace=True)
    #
    # if Test_Dataset["Order Date"].isna().any():
    #     Test_Dataset["Order Date"].fillna(pd.to_datetime("2017-09-02 00:00:00"), inplace=True)
    #
    # if Test_Dataset["Ship Date"].isna().any():
    #     Test_Dataset["Ship Date"].fillna(pd.to_datetime("2017-09-26 00:00:00"), inplace=True)
    #
    # if Test_Dataset["Postal Code"].isna().any():
    #     Test_Dataset["Postal Code"].fillna("10035.0", inplace=True)
    #
    # if Test_Dataset["City"].isna().any():
    #     Test_Dataset["City"].fillna("New York City", inplace=True)
    #
    # if Test_Dataset["State"].isna().any():
    #     Test_Dataset["State"].fillna("California", inplace=True)
    #
    # if Test_Dataset["Region"].isna().any():
    #     Test_Dataset["Region"].fillna("West", inplace=True)

    if Test_Dataset["ReturnCategory"].isna().any():
         Test_Dataset["ReturnCategory"] = Test_Dataset["ReturnCategory"].dropna(how='any')
    if Test_Dataset["Discount"].isna().any():
        # Dataset["Discount"].fillna(Dataset["Discount"].mean(), inplace=True)
        Test_Dataset["Discount"].fillna(0.2, inplace=True)

    if Test_Dataset["Sales"].isna().any():
        # Dataset["Sales"].fillna(Dataset["Sales"].mean(), inplace=True)
        Test_Dataset["Sales"].fillna(54.792, inplace=True)

    if Test_Dataset["Quantity"].isna().any():
        # Dataset["Quantity"].fillna(Dataset.mode()['Quantity'][0], inplace=True)
        Test_Dataset["Quantity"].fillna(3, inplace=True)

    if Test_Dataset["Country"].isna().any():
        Test_Dataset["Country"].fillna("United States", inplace=True)

    if Test_Dataset["Product Name"].isna().any():
        Test_Dataset["Product Name"].fillna("Staples", inplace=True)
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

    if Test_Dataset["Order ID"].isna().any():
        Order_ID_Dict = Handle_Missing_Order_ID(Test_Dataset, Order_ID_Dict)
    Test_Dataset["Order ID"] = dict(sorted(Order_ID_Dict.items())).values()

#######################################################################################################################################
"""                                                                                                   """
#######################################################################################################################################
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

def Numerical_Correlation(Numerical_Dataframe, Y_Train):
    Numerical_Columns = ['Row ID', 'Order ID', 'Customer ID', 'Postal Code',
                         'Product ID', 'Sales', 'Quantity', 'Discount']
    correlations = [kendalltau(Numerical_Dataframe.iloc[:, i], Y_Train)[0] for i in
                    range(Numerical_Dataframe.shape[1])]
    print("\nNumerical features:", Numerical_Columns)
    print("Numerical correlation:", correlations)
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    for index in sorted_indices:
        print(Numerical_Columns[index], "Correlation:", correlations[index])
    # Select the top 3 features with highest correlation coefficients
    selected_features = Numerical_Dataframe.columns[sorted_indices[:3]]
    # Display the selected features
    print("Selected numerical features: ", selected_features.tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(Numerical_Dataframe.columns[sorted_indices], np.array(correlations)[sorted_indices], color='b')
    ax.set_xticklabels(Numerical_Dataframe.columns[sorted_indices], rotation=45, ha='right')
    ax.set_xlabel('Numerical Feature')
    ax.set_ylabel("Kendall's Rank Correlation Coefficient")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

def Categorical_Correlation(Categorical_Dataframe, Y_Train):
    Categorical_Columns = Categorical_Dataframe.columns.to_list()
    selector = SelectKBest(chi2, k=7)
    chi_sq = selector.fit(Categorical_Dataframe, Y_Train)
    sorted_indices = np.argsort(np.abs(chi_sq.scores_))[::-1]
    selected_features = Categorical_Dataframe.columns[sorted_indices[:7]]
    print("\nCategorical features:")
    for index in sorted_indices:
        print(Categorical_Columns[index], "Correlation:", chi_sq.scores_[index])
    # Display the selected features
    print("Selected categorical features: ", selected_features.tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(Categorical_Dataframe.columns[sorted_indices], np.array(chi_sq.scores_)[sorted_indices], color='b')
    ax.set_xticklabels(Categorical_Dataframe.columns[sorted_indices], rotation=45, ha='right')
    ax.set_xlabel('Categorical Feature')
    ax.set_ylabel("Chi-squared Correlation")
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
def preprocessing_train(df,columns):  #1- check outlers 2- encoder 3- scaling  4- cheack bais   5- slection
    Numerical_Dataframe = df[columns['Numerical_Columns']]
    missing_val = df.isnull().sum()
    print("Null Values in Train : ",missing_val)
    print("Null Values in Train : ",missing_val[missing_val==True].index.tolist())
    boxplot(Numerical_Dataframe, columns['Numerical_Columns'])
    # df=Handle_Outliers(df,columns)
    df=Remove_Outliers(df,columns)
    print("#"*100)
    print("Num Of Outliers In Train : ")
    df=Detect_Outliers(df,columns)
    print("#"*100)

    Numerical_Dataframe = df[columns['Numerical_Columns']]
    # boxplot(Numerical_Dataframe, columns['Numerical_Columns'])

    df = Check_Duplicates(df)
    feature_encoder_train(df) # 0 1
    feature_encoder_transform(df)

    Numerical_Dataframe = df[columns['Numerical_Columns']]
    Categorical_Dataframe = df[columns['Categorical_Columns']]

    Feature_Scaling_Train(df, Numerical_Dataframe)
    # data_visualization(df)
    X = df.drop(['ReturnCategory'], axis=1, inplace=False)
    Y = df['ReturnCategory']
    # X, Y = under_sample(X, Y)
    X, Y = over_sample(X, Y)
    # X, Y = near_miss(X, Y)
    # X, Y = smote(X, Y)
    df_sampled = X.join(Y)
    print(df_sampled.head(20))
    Numerical_Dataframe = df_sampled[columns['Numerical_Columns']]
    df_cat = df_sampled[columns['Categorical_Columns']]
    Numerical_Correlation(Numerical_Dataframe, df_sampled['ReturnCategory'])
    Categorical_Correlation(df_cat, df_sampled['ReturnCategory'])
    # data_visualization(df_sampled)
    return df_sampled

def Hyperparam(x_test,y_test,model,parms,x_train, y_train,x):# selection for best hyperparamter
    if x==0:
        grid_search = RandomizedSearchCV(model, param_distributions=parms, n_iter=100, cv=5, random_state=42)
    else:
        grid_search = GridSearchCV(model, param_grid=parms, cv=5)
    grid_search.fit(x_train, y_train)
    # Print the best hyperparameters and accuracy score
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Accuracy score: ", grid_search.score(x_test, y_test))

def Preprocess_Test(df, columns):  # scaling + encoding
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

def over_sample(X, Y):
    sm = RandomOverSampler()
    X_over_sampled, Y_over_sampled = sm.fit_resample(X, Y)
    print("Over sampling:")
    print("X_balanced shape is ", X_over_sampled.shape)
    print("y_balanced shape is ", Y_over_sampled.shape)
    print(Y_over_sampled.value_counts())
    return X_over_sampled, Y_over_sampled

def under_sample(X, Y):
    sm = RandomUnderSampler()
    X_under_sampled, Y_under_sampled = sm.fit_resample(X, Y)
    print("Under sampling:")
    print("X_balanced shape is ", X_under_sampled.shape)
    print("y_balanced shape is ", Y_under_sampled.shape)
    print(Y_under_sampled.value_counts())
    return X_under_sampled, Y_under_sampled

def smote(X, Y):
    smote = SMOTEN(random_state=42)
    # fit predictor and target variable
    X_copy = X.drop(['Order Date', 'Ship Date'], axis=1)
    x_smote, y_smote = smote.fit_resample(X_copy, Y)
    return x_smote, y_smote

def near_miss(X, Y):
    nm = NearMiss()
    X_copy = X.drop(['Order Date', 'Ship Date', 'CategoryTree'], axis=1)
    x_nm, y_nm = nm.fit_resample(X_copy, Y)
    return x_nm, y_nm

def data_visualization(df):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    sns.set_style('whitegrid')
    fig.suptitle("Count plot for various categorical features")
    ax1 = sns.countplot(ax=axes[0, 0], data=df, x='Segment')
    ax2 = sns.countplot(ax=axes[0, 1], data=df, x='City')
    ax3 = sns.countplot(ax=axes[1, 0], data=df, x='State')
    ax4 = sns.countplot(ax=axes[1, 1], data=df, x='Region')
    for label in ax1.containers:
        ax1.bar_label(label)
    for label in ax2.containers:
        ax2.bar_label(label)
    for label in ax3.containers:
        ax3.bar_label(label)
    for label in ax4.containers:
        ax4.bar_label(label)
    plt.show()
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    sns.set_style('whitegrid')
    ax1 = sns.countplot(ax=axes[0, 0], data=df, x='Ship Mode')
    ax2 = sns.countplot(ax=axes[0, 1], data=df, x='Main Category')
    ax3 = sns.countplot(ax=axes[1, 0], data=df, x='Sub Category')
    ax4 = sns.countplot(ax=axes[1, 1], data=df, x='ReturnCategory')
    for label in ax1.containers:
        ax1.bar_label(label)
    for label in ax2.containers:
        ax2.bar_label(label)
    for label in ax3.containers:
        ax3.bar_label(label)
    for label in ax4.containers:
        ax4.bar_label(label)
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    ax1 = sns.histplot(ax=axes[0, 0], data=df['Row ID'])
    ax2 = sns.histplot(ax=axes[0, 1], data=df['Order ID'])
    ax3 = sns.histplot(ax=axes[1, 0], data=df['Customer ID'])
    ax4 = sns.histplot(ax=axes[1, 1], data=df['Product ID'])
    plt.show()
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    ax1 = sns.histplot(ax=axes[0], data=df['Sales'])
    ax2 = sns.histplot(ax=axes[1], data=df['Quantity'])
    ax3 = sns.histplot(ax=axes[2], data=df['Discount'])
    plt.show()

def plot_confusion_matrix(y_test, pred):
    plt.figure(figsize=(10, 6))
    fx = sns.heatmap(metrics.confusion_matrix(y_test, pred), annot=True, fmt=".2f", cmap="GnBu")
    fx.set_title('Confusion Matrix \n')
    fx.set_xlabel('\n Predicted Values\n')
    fx.set_ylabel('Actual Values\n')
    fx.xaxis.set_ticklabels(['Medium Profit', 'Low Profit', 'Low Loss', 'High Profit', 'High Loss'])
    fx.yaxis.set_ticklabels(['Medium Profit', 'Low Profit', 'Low Loss', 'High Profit', 'High Loss'])
    plt.show()
"""                                                Function Call                                                     """

# Read Data
Full_Dataset = pd.read_csv('megastore-classification-dataset.csv')
Test_Dataset = pd.read_csv('megastore-classification-dataset.csv')

ID_Columns = ['Order ID', 'Customer ID', 'Product ID']
Numerical_And_Discrete_Columns = ['Postal Code', 'Quantity']
Numerical_And_Continuous_Columns = ['Sales', 'Discount']
Numerical_Columns = ['Row ID', 'Order ID', 'Customer ID', 'Postal Code',
                     'Product ID', 'Sales', 'Quantity', 'Discount']
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

'''X_train =train_data[['Sales', 'Discount', 'Region', 'Quantity']]
Y_Train=train_data['Profit']
X_test=test_data[['Sales', 'Discount', 'Region', 'Quantity']]
Y_Test=test_data['Profit']'''
X_train = Train_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region',
                      'Main Category', 'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Train = Train_Data['ReturnCategory']
X_test = Test_Data[['Ship Mode', 'Postal Code', 'City', 'State', 'Region', 'Main Category',
                    'Sub Category', 'Sales', 'Quantity', 'Discount']]

Y_Test = Test_Data['ReturnCategory']
#Discount', 'Region', 'State', 'Main Category', 'City' accurcy=0.52 (Accuracy_tree: 0.5259537210756723)

###################################################################
# check hyperparm
params_L = {'C': [0.001, 0.01, 0.1, 1, 10],
          'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'max_iter': [100, 500, 1000]}
lr = LogisticRegression()
params_T = {'max_depth': [5, 10, 15],
          'min_samples_split': [2, 5, 10],
          'criterion': ['gini', 'entropy']}

# Create a decision tree classifier object
dtc = DecisionTreeClassifier()
param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}
knn = KNeighborsClassifier()
params_RF = {
    'n_estimators': list(range(10, 200)),
    'max_depth': list(range(1, 50)),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
# Hyperparam(X_test,Y_Test,lr,params_L,X_train,Y_Train,1)# Best hyperparameters:  {'C': 10, 'max_iter': 100, 'solver': 'newton-cg'}Accuracy score:  0.6823014383989994
# Hyperparam(X_test,Y_Test,dtc,params_T,X_train,Y_Train,1) #Best hyperparameters:  {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 10}
# Hyperparam(X_test,Y_Test,knn,param_grid,X_train,Y_Train,1)#Best hyperparameters:  {'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
# Hyperparam(X_test,Y_Test,rf,params_RF,X_train,Y_Train,0)#Best hyperparameters:
# Perform a random search over the hyperparameter grid
# random_search = RandomizedSearchCV(rf, param_distributions=params_RF, n_iter=100, cv=5, random_state=42)
# File_Name_Logistic_Regression_Model = 'RF.sav'
# random_search.fit(X_train,Y_Train)
# pickle.dump(rf, open(File_Name_Logistic_Regression_Model, 'wb'))
#
#
# # Print the best hyperparameters and corresponding score
# print("Best Hyperparameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)
# ######################################################################################################################################################################
# logistic=>
# Tree    =>
# KNN     =>
# RF      =>
""" When i use hyperparmter """
""" with outliers
tst 0.0019989013671875 
Accuracy_tree: 0.8567854909318324
tst 0.0009732246398925781
Accuracy_log: 0.6823014383989994
Accuracy_KNN 0.535334584115072
"""
""" remove outliers in train data
tst 0.002036571502685547
Accuracy_tree: 0.7854909318323953
tst 0.0010020732879638672
Accuracy_log: 0.7473420888055035
Accuracy_KNN 0.5090681676047529
############################################
1-DecisionTreeClassifier
tst 0.002961874008178711
Accuracy_tree: 0.7892432770481551
2-Logistic Regression
tst 0.002000093460083008
Accuracy_log: 0.7473420888055035
KNN
Accuracy_KNN 0.4790494058786742
"""
"""Remove outliers in both train and test
tst 0.0030019283294677734
Accuracy_tree: 0.8701195219123506
tst 0.0010004043579101562
Accuracy_log: 0.7625498007968128
"""
print("1-DecisionTreeClassifier")

start_time = time.time()
# dtc = DecisionTreeClassifier(criterion="entropy",max_depth=10,min_samples_split=10) # 84% accuracy after slect the best hyperparm =Accuracy_tree: 0.8561601000625391
dtc = DecisionTreeClassifier(criterion="entropy",max_depth=15,min_samples_split=2) # 84% accuracy after slect the best hyperparm =Accuracy_tree: 0.8561601000625391
scores = cross_val_score(dtc, X_train, Y_Train, cv=5)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)

# Train the model on the training set
dtc.fit(X_train, Y_Train)
File_Name_Decision_Tree_Model = 'Decision tree_model.sav'
pickle.dump(dtc, open(File_Name_Decision_Tree_Model, 'wb'))
training_time_tre = time.time() - start_time
# Make predictions on the testing set
start_time = time.time()
#Load_Decision_Tree_Model = pickle.load(open(File_Name_Decision_Tree_Model, 'rb'))
#y_pred = Load_Decision_Tree_Model.predict(X_test)
y_pred = dtc.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(Y_Test, y_pred)
testing_time_Tre = time.time() - start_time
print("tst",testing_time_Tre)
print(f"Accuracy_tree: {accuracy}")
# Precision, F1-score
DecTree_precision,DecTree_recall,DecTree_f1_score,DecTree_support=score(Y_Test, y_pred, average='macro')
print ('Decision tree - Precision : {}'.format(DecTree_precision))
print ('Decision tree - F1-score  : {}'.format(DecTree_f1_score))
# Confusion Matrix
plot_confusion_matrix(Y_Test, y_pred)

#############################################################################
print("2-Logistic Regression")
start_time = time.time()
lr = LogisticRegression(C=10,max_iter=100,solver="newton-cg")
lr.fit(X_train, Y_Train)
File_Name_Logistic_Regression_Model = 'Logistic Regression.sav'
pickle.dump(lr, open(File_Name_Logistic_Regression_Model, 'wb'))
training_time_lr = time.time() - start_time
scores = cross_val_score(lr, X_train, Y_Train, cv=5)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)

# Make predictions on the test set using the logistic regression model
start_time = time.time()
#Load_Logistic_Regression_Model = pickle.load(open(File_Name_Logistic_Regression_Model, 'rb'))
#y_pred_lr = Load_Logistic_Regression_Model.predict(X_test)
y_pred_lr = lr.predict(X_test)
testing_time_lr = time.time() - start_time
print("tst",testing_time_lr)
# Measure the classification accuracy of the logistic regression model
accuracy_lr = accuracy_score(Y_Test, y_pred_lr)
print(f"Accuracy_log: {accuracy_lr}")#0.4046278924327705
LogReg_precision,LogReg_recall,LogReg_f1_score,LogReg_support=score(Y_Test, y_pred_lr, average='macro')
print ('LogReg - Precision : {}'.format(LogReg_precision))
print ('LogReg - F1-score  : {}'.format(LogReg_f1_score))
plot_confusion_matrix(Y_Test, y_pred_lr)
#############################################################################
# Train the KNN model and measure training time
print("KNN")
start_time = time.time()
# knn = KNeighborsClassifier(n_neighbors=7,p=1,weights="distance")#0.4834271419637273
knn = KNeighborsClassifier(n_neighbors=3,p=1,weights="distance")
knn.fit(X_train, Y_Train)
File_Name_KNN_Model = 'KNN.sav'
pickle.dump(knn, open(File_Name_KNN_Model, 'wb'))
training_time_knn = time.time() - start_time
scores = cross_val_score(knn, X_train, Y_Train, cv=5)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)

# Make predictions on the test set using the KNN model
start_time = time.time()
#Load_KNN_Model = pickle.load(open(File_Name_KNN_Model, 'rb'))
#y_pred_knn = Load_KNN_Model.predict(X_test)
y_pred_knn = knn.predict(X_test)
testing_time_knn = time.time() - start_time
# Measure the classification accuracy of the KNN model
accuracy_knn = accuracy_score(Y_Test, y_pred_knn)
print("Accuracy_KNN",accuracy_knn)
# Precision, F1-score
Knn_precision,Knn_recall,Knn_f1_score,Knn_support=score(Y_Test, y_pred_knn, average='macro')
print ('KNN - Precision : {}'.format(Knn_precision))
print ('KNN - F1-score  : {}'.format(Knn_f1_score))
plot_confusion_matrix(Y_Test, y_pred_knn)
######################################################
# Best Hyperparameters: {'n_estimators': 189, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 18}
# Best Score: 0.9495745633676668
print("Random forest classification:")
start_time = time.time()
rfc = RandomForestClassifier(n_estimators=129,min_samples_split=2,min_samples_leaf=1,max_features=None,max_depth=29,random_state=42)
rfc.fit(X_train, Y_Train)
File_Name_rfc_Model = 'RandomForestClassifier.sav'
pickle.dump(rfc, open(File_Name_rfc_Model, 'wb'))
training_time_rf = time.time() - start_time
scores = cross_val_score(rfc, X_train, Y_Train, cv=5)
# Print the mean accuracy and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# mean accuracy is high but the standard deviation is also high(overfitting)

start_time = time.time()
rfc_predict = rfc.predict(X_test)
testing_time_rf = time.time() - start_time
accuracy_rf=accuracy_score(Y_Test, rfc_predict)
print('Accuracy random forest:', accuracy_rf)
rf_precision,rf_recall,rf_f1_score,rf_support=score(Y_Test, rfc_predict, average='macro')
print ('Random Forest - Precision : {}'.format(rf_precision))
print ('Random Forest - F1-score  : {}'.format(rf_f1_score))
plot_confusion_matrix(Y_Test, rfc_predict)

######################################################
# Generate bar graphs to show the results
models = ['Logistic Regression', 'DecisionTreeClassifier','KNN','RandomForestClassification']
accuracy_scores = [accuracy_lr, accuracy,accuracy_knn,accuracy_rf]
training_times = [training_time_lr, training_time_tre,training_time_knn,training_time_rf]
testing_times = [testing_time_lr, testing_time_Tre,testing_time_knn,testing_time_rf]

plt.bar(models, accuracy_scores)
plt.title('Classification Accuracy')
plt.show()

plt.bar(models, training_times)
plt.title('Total Training Time')
plt.show()

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
