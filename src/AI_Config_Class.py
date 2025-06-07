import  pandas               as pd
import  numpy                as np
import  matplotlib.pyplot    as plt
from    sklearn.preprocessing       import LabelEncoder
from    sklearn.model_selection     import train_test_split
from    sklearn.neighbors           import KNeighborsClassifier
from    sklearn.metrics             import accuracy_score, precision_score, recall_score
from    sklearn.decomposition       import PCA


class AI_Config_Class:
    def __init__(self):
        self.label_encolder = LabelEncoder()
        self.data           = pd.read_csv("https://www.sciencebuddies.org/ai/colab/breastcancer.csv?t=AQVCB3SEUt1Ppj49TSARt_r7dw4yEr1BX1UMsPy_nxHB8A")
        
        self.numerical_colums = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                                 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                                 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                                 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
                                 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                                 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
                                 'symmetry_worst', 'fractal_dimension_worst']
        self.__config()
        
        self.x = self.data.drop('diagnosis', axis=1)
        self.y = self.data['diagnosis']
        self.k = 5

    
    def __config(self):
        self.data.drop('id', axis=1, inplace=True)
        self.data[self.numerical_colums] = (
            self.data[self.numerical_colums] - self.data[self.numerical_colums].min()) / (
            self.data[self.numerical_colums].max() - self.data[self.numerical_colums].min())
        self.data['diagnosis'].unique()
        self.data['diagnosis'] = self.label_encolder.fit_transform(self.data['diagnosis'])
    
    
    def __IA_Model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(x_train, y_train)
        
        y_pred = knn.predict(x_test)
        accuracy    = accuracy_score(y_test, y_pred)
        precision   = precision_score(y_test, y_pred)
        recall      = recall_score(y_test, y_pred)
        
        print("accuracy: "  , accuracy)
        print("precision: " , precision)
        print("recall: "    , recall)
    
    
    def AI_Show_Result(self):
        #self.__config()
        self.__IA_Model()