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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc



def clean_data(df_raw):
    """Clean raw dataset
    1. convert mixed type to int
    2. fill missing value
    3. cap outlier
    4. drop columns with too many identical value
    
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
    
    # cap outlier
    kind_col = ['ALTER_KIND1', 'ALTER_KIND2', 'ALTER_KIND3', 'ALTER_KIND4']
    for col in kind_col:
        df[col] = np.where(df[col]>0, 1, 0)

    df['ALTER_KIND'] = (df['ALTER_KIND1'] + df['ALTER_KIND2']
                            + df['ALTER_KIND3'] + df['ALTER_KIND4'])
    df.drop(kind_col, axis=1, inplace = True)  
    
    df.drop(['EXTSEL992', 'KK_KUNDENTYP'], axis=1, inplace = True)
    
    df.fillna(0, inplace=True)
    
    outlier_cols = ['ANZ_HAUSHALTE_AKTIV', 'ANZ_STATISTISCHE_HAUSHALTE']
    for col in outlier_cols:
        df[col] = np.where(df[col]>9, 9, df[col])
    
    # drop columns with too many identical value
    drop_col = ['D19_TELKO_ONLINE_QUOTE_12', 'D19_VERSI_ONLINE_QUOTE_12',
       'TITEL_KZ', 'ANZ_TITEL', 'SOHO_KZ', 'D19_BANKEN_LOKAL',
       'ANZ_HH_TITEL', 'D19_TELKO_ANZ_12', 'D19_DIGIT_SERV',
       'D19_BIO_OEKO', 'D19_TIERARTIKEL', 'D19_NAHRUNGSERGAENZUNG',
       'D19_GARTEN', 'D19_BANKEN_ONLINE_QUOTE_12', 'D19_LEBENSMITTEL',
       'D19_WEIN_FEINKOST', 'D19_BANKEN_ANZ_12', 'D19_ENERGIE',
       'D19_TELKO_ANZ_24', 'D19_BANKEN_REST', 'D19_VERSI_ANZ_12',
       'HH_DELTA_FLAG', 'UNGLEICHENN_FLAG', 'D19_BILDUNG', 'ALTER_KIND',
       'D19_BEKLEIDUNG_GEH', 'D19_RATGEBER', 'ANZ_KINDER',
       'D19_SAMMELARTIKEL', 'D19_BANKEN_ANZ_24', 'D19_FREIZEIT',
       'KBA05_SEG6', 'D19_BANKEN_GROSS', 'D19_VERSI_ANZ_24', 'D19_SCHUHE',
       'D19_HANDWERK', 'D19_TELKO_REST', 'D19_SOZIALES',
       'D19_DROGERIEARTIKEL', 'D19_KINDERARTIKEL', 'D19_LOTTO',
       'D19_KOSMETIK', 'VHA', 'D19_REISEN', 'D19_VERSAND_REST',
       'KBA05_ANTG4', 'D19_BANKEN_DIREKT', 'D19_TELKO_MOBILE',
       'GREEN_AVANTGARDE', 'D19_HAUS_DEKO']
    
    df.drop(drop_col, axis=1, inplace = True)  
    
    df_missing = (df.isna().sum()).sum()
    
    print(f'Dataset has {df_missing} missing value.')
    
    return df


        
def build_model(model):
    """GridSearchCV for XGBoost hyper-parameter tunning
    Arg:
        model=str, model used in pipeline
    
    Returns:
        cv: model with best estimator 
    """
    
    model_dict={
        'XGBoost': XGBClassifier(),
        'Random Forecast': RandomForestClassifier(),
        'Logistic': LogisticRegression()
    }
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('clf', model_dict.get(model))
    ])
    
    if model=='XGBoost':
        parameters = {
            'clf__gamma': [i/10 for i in range(6,9)],
            'clf__min_child_weight': [i for i in range(1,3)],
            'clf__subsample': [i/10 for i in range(7,9)],
            'clf__colsample': [i/10 for i in range(7,9)]
        }
    elif model=='Random Forecast':
        parameters = {
            'clf__min_impurity_decrease': [0.1, 0.2, 0.5],
            'clf__min_samples_leaf': [i*10 for i in range(3,7)],
        }
                                           
    elif model=='Logistic':
        parameters = {
            'clf__penalty': ['l1', 'l2'], 
        }
        
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1, scoring='roc_auc')
    
    return cv

def save_model_result(model):
    """Check top cv results and save the model"""
    cv_results = pd.DataFrame(model.cv_results_).sort_values(by='rank_test_score')
    model_name = 'src/'+ type(model.best_estimator_[2]).__name__ + '_clf_mdl.plk'
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
    return cv_results

def model_evaluation(X_train, X_test, y_train, y_test, model):
    """ Model Evaluation with in-sample and off-sample test
        Metrics: precision, recall, accuracy, auc
    """
    y_train_pred = model.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn)/ (tn+ fp+ fn+ tp)
    
    fpr, tpr, thresholds = roc_curve(y_train, y_train_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    print(f'In-Sample Test\n'
         + f'precision: {precision}, \nrecall: {recall}, \naccuracy: {accuracy}, \nauc: {auc}')
    
    y_test_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn)/ (tn+ fp+ fn+ tp)
    
    fpr, tpr, thresholds = roc_curve(y_train, y_train_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    print(f'Off-Sample Test\n'
         + f'precision: {precision}, \nrecall: {recall}, \naccuracy: {accuracy}, \nauc: {auc}\n')
    
    # draw roc chart
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
