import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

def clean_data(df_raw):
    """Handel mixed type and fill missing value for raw dataset
    
    Args:
        df_raw=pd.DataFrame(), Raw dataset of Arvato
    Returns:
        df=pd.DataFrame(), cleaned dataset
    """
    
    df = df_raw.copy()
    # Map categorical value(1A,1B,2A, etc.) to numerical value
    CAMEO_DEU_2015_value = df.groupby(['CAMEO_DEU_2015']).count()['LNR'].index
    CAMEO_DEU_2015_value_map = {}
    for i, item in enumerate(CAMEO_DEU_2015_value):
        # Conside XX as missing value, fill with 0
        if item== 'XX':
            CAMEO_DEU_2015_value_map[item] = 0
        else:
            CAMEO_DEU_2015_value_map[item] = i + 1 
            
    df['CAMEO_DEU_2015'] = df['CAMEO_DEU_2015'].apply(lambda x: CAMEO_DEU_2015_value_map.get(x))
    df['CAMEO_DEU_2015'].fillna(0, inplace=True)
    
    # Conside X as missing value, fill with 0
    df['CAMEO_DEUG_2015'] = np.where(df['CAMEO_DEUG_2015']=='X', 0, df['CAMEO_DEUG_2015'])
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype('float')
    df['CAMEO_DEUG_2015'].fillna(0, inplace=True)
    
    # Conside XX as missing value, fill with 0
    df['CAMEO_INTL_2015'] = np.where(df['CAMEO_INTL_2015']=='XX', 0, df['CAMEO_INTL_2015'])
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].astype('float')
    df['CAMEO_INTL_2015'].fillna(0, inplace=True)
    
    # Already has encoded columns contain branch info.
    df.drop(['D19_LETZTER_KAUF_BRANCHE'], axis=1, inplace=True)
    
    # Only keep year
    df['EINGEFUEGT_AM'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    df['EINGEFUEGT_AM'].fillna(df['EINGEFUEGT_AM'].mode()[0], inplace=True)
    
    # Map O and W to 1 and 2, fill Nan with 0
    df['OST_WEST_KZ'] = np.where(
        df['OST_WEST_KZ']=='O',
        1,
        np.where(df['OST_WEST_KZ']=='W', 2, 0)
    )
    
    kind_col = ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4']
    for col in kind_col:
        df[col] = np.where(df[col]>0, 1, 0)

    df['ALTER_KIND'] = (df['ALTER_KIND1'] + df['ALTER_KIND2']
                            + df['ALTER_KIND3'] + df['ALTER_KIND4'])
    df.drop(kind_col, axis=1, inplace = True)  
    
    df.drop(['EXTSEL992', 'KK_KUNDENTYP'], axis=1, inplace = True)
    
    df.fillna(0, inplace=True)
    
    df_missing = (df.isna().sum()).sum()
    
    print(f'Dataset has {df_missing} missing value.')
    
    return df


        
def build_model():
    """GridSearchCV for XGBoost hyper-parameter tunning
    
    Returns:
        cv: model with best estimator 
    """
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('clf', XGBClassifier())
    ])
    

    parameters = {
        'clf__gamma': [0.7], #[i/10 for i in range(0,10)],
        'clf__max_depth': [i for i in range(3,7)],
        'clf__min_child_weight': [i for i in range(1,5)],
        'clf__subsample': [i/10 for i in range(7,11)],
        'clf__colsample': [i/10 for i in range(7,11)]
    }
        
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1, scoring='roc_auc')
    
    return cv