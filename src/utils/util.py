import pandas as pd
import numpy as np


def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),columns=['dup_cnt'])
    # print(old_feature_name_df.groupby('column_name').cumcount())
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(),feature_dup_df,how='outer')
    # print(old_feature_name_df)
    # print(feature_dup_df)
    # print(new_feature_name_df)
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name','dup_cnt']].apply(lambda x : x[0]+'_'+str(x[1]) if x[1]>0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(['index'],axis=1)
    return new_feature_name_df

def get_human_dataset():

    feature_name_df = pd.read_csv('../../dataset/human_activity/features.txt',sep='\s+',header=None,names=['column_index','column_name'])

    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:,1].values.tolist()
    X_train = pd.read_csv('../../dataset/human_activity/train/X_train.txt',sep='\s+',names=feature_name)
    X_test = pd.read_csv('../../dataset/human_activity/test/X_test.txt',sep='\s+',names=feature_name)

    y_train = pd.read_csv('../../dataset/human_activity/train/y_train.txt',sep='\s+',names=['action'],header=None)
    y_test = pd.read_csv('../../dataset/human_activity/test/y_test.txt',sep='\s+',names=['action'],header=None)

    return X_train,X_test,y_train,y_test


from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행.
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def get_clf_eval(y_test,pred=None,pred_proba=None):
    confusion = confusion_matrix(y_test,pred)
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    if pred_proba is None :
        print("오차 행렬")
        print(confusion)
        print("정확도 :{0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}, F1:{3:.4f}".format(accuracy,precision,recall,f1))
    else :
        roc_auc = roc_auc_score(y_test, pred_proba)
        print("오차 행렬")
        print(confusion)
        print("정확도 :{0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}, F1:{3:.4f}, AUC:{4:.4f}".format(accuracy, precision, recall, f1,
                                                                                       roc_auc))


def precision_recall_curve_plot(y_test,pred_proba_c1):
    precisions,recalls,thresholds = precision_recall_curve(y_test,pred_proba_c1)

    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds,precisions[0:threshold_boundary],linestyle='--',label='precision')
    plt.plot(thresholds,recalls[0:threshold_boundary],label='recall')

    start,end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))

    plt.xlabel("Threshold value");plt.ylabel("Precision and Recall value")
    plt.legend();plt.grid()
    plt.show()

from sklearn.preprocessing import Binarizer
def get_eval_by_threshold(y_test,pred_proba,thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba)
        custom_predict = binarizer.transform(pred_proba)
        print("임계값:",custom_threshold)
        get_clf_eval(y_test,custom_predict)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_preprocessed_df(df=None):
    df_copy=df.copy()
    # scaler = StandardScaler()
    amount_n = np.log1p(df_copy['Amount'])#scaler.fit_transform(df_copy['Amount'].values.reshape(-1,1))
    df_copy.insert(0,'Amount_Scaled',amount_n)
    df_copy.drop(['Time','Amount'],axis=1,inplace=True)

    outlier_index=get_outlier(df=df_copy,column='V14')
    df_copy.drop(outlier_index,axis=0,inplace=True)
    return df_copy

def get_train_test_dataset(df=None):
    df_copy=get_preprocessed_df(df)
    X_features=df_copy.iloc[:,:-1]
    y_target = df_copy.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.3,random_state=0,stratify=y_target)
    return X_train,X_test,y_train,y_test

def get_model_train_eval(model,ftr_train=None,ftr_test=None,tgt_train=None,tgt_test=None):
    model.fit(ftr_train,tgt_train)
    pred=model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]

    get_clf_eval(tgt_test,pred,pred_proba)

def get_outlier(df=None,column=None,weight=1.5):
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    iqr = quantile_75-quantile_25
    iqr_weight = iqr*weight
    lowest_val = quantile_25-iqr_weight
    highest_val = quantile_75+iqr_weight
    outlier_index = fraud[(fraud<lowest_val)|(fraud>highest_val)].index
    return outlier_index