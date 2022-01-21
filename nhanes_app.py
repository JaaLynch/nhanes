import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import wget
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost, shap
from lifelines.utils import concordance_index

st.title('NHANES Modeling Dashboard')

url = 'https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx'
st.markdown("Documentation for continuous NHANES [CDC website](%s)" % url)

# Downloads a single XPT file for a given path
def download_data(
    source_path = 'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT',
):
    print('\nDownloading file: ')
    print('     '+ source_path+'/n')
    
    data_file = source_path.rsplit('/',1)[1]
    new_path = 'data/'+data_file
    if os.path.exists(new_path):
        os.remove(new_path)
    wget.download(source_path, new_path)
    file_load_state = st.text('Downloaded file: '+source_path)

# Loop through the provided a list of paths
def batch_download_data(files):
    for file in files:
        data_file = file.rsplit('/',1)[1]
        new_path = 'data/'+data_file
        if not os.path.exists(new_path):
            download_data(file)

@st.cache(allow_output_mutation=True)
def get_file_list(source='Demographics'):
    page = requests.get('https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component='+source)
    soup = BeautifulSoup(page.content, 'html.parser')
    file_list = []
    for a in soup.find_all('a', href=True):
        file_list.append(a['href'])
    file_list = ['https://wwwn.cdc.gov'+file for file in file_list if '.XPT' in file]
    
    return file_list

def get_field_list(doc_list = ['https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/DEMO_D.htm']):
    field_list = []
    for doc in doc_list:
        page = requests.get(doc)
        soup = BeautifulSoup(page.content, 'html.parser')
        soup1 = soup.find("div", {"id": "CodebookLinks"})
        soup2 = soup.find("ul", {"id": "CodebookLinks"})
        if soup1 is not None:
            for a in soup1.find_all('a', href=True):
                field_list.append(a.text)
        if soup2 is not None:
            for a in soup2.find_all('a', href=True):
                field_list.append(a.text)
    field_list = list(set(field_list))
    field_list = sorted(field_list)
    return field_list


scrape_state = st.text('Scraping data...')
files_demog = get_file_list(source='Demographics')
files_diet = get_file_list(source='Dietary')
files_exam = get_file_list(source='Examination')
files_lab = get_file_list(source='Laboratory')
files_quest = get_file_list(source='Questionnaire')
files_mort = [
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_1999_2000_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2001_2002_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2003_2004_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2005_2006_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2007_2008_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2009_2010_MORT_2015_PUBLIC.dat', 
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2011_2012_MORT_2015_PUBLIC.dat',
     'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_2013_2014_MORT_2015_PUBLIC.dat'
]
scrape_state.text('Scraping data...Done! The required metadata has been retrieved.')

if st.checkbox('Check to create field list with usefull names'):
    name_scrape_state = st.text('Gathering unique field names...')
    files = files_demog + files_diet + files_exam + files_lab + files_quest + files_mort
    doc_list = [file.replace('XPT','htm') for file in files]
    field_list = get_field_list(doc_list)
    pd.DataFrame(field_list, columns=['field_list']).to_csv('data/field_list.csv', index=False)
    name_scrape_state.text('Gathering unique field names... Done!')

if os.path.isfile('data/field_list.csv'):
    field_list = pd.read_csv('data/field_list.csv')
    field_list = field_list.field_list.values.tolist()

## ---
st.subheader('Download data from CDC website')

if st.checkbox('Check to download: Demographic'):
    data_load_state = st.text('Loading data...')
    data = batch_download_data(files_demog)
    data_load_state.text("Loading data... Done!")

if st.checkbox('Check to download: Dietary'):
    data_load_state = st.text('Loading data...')
    data = batch_download_data(files_diet)
    data_load_state.text("Loading data... Done!")

if st.checkbox('Check to download: Laboratory'):
    data_load_state = st.text('Loading data...')
    data = batch_download_data(files_lab)
    data_load_state.text("Loading data... Done!")

if st.checkbox('Check to download: Questionnaire'):
    data_load_state = st.text('Loading data...')
    data = batch_download_data(files_quest)
    data_load_state.text("Loading data... Done!")

if st.checkbox('Check to download: Linked Mortality'):
    data_load_state = st.text('Loading data...')
    data = batch_download_data(files_mort)
    data_load_state.text("Loading data... Done!")

if st.checkbox('Check to download: Examination'):
    data_load_state = st.text('Loading data...')

    skip_file = 'https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/PAXMIN_G.XPT'
    if skip_file in files_exam: files_exam.remove(skip_file)

    skip_file = 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/PAXMIN_H.XPT'
    if skip_file in files_exam: files_exam.remove(skip_file)

    data = batch_download_data(files_exam)
    data_load_state.text("Loading data... Done!")

## ---
st.subheader('Raw Data: Sample and Histograms')

if st.checkbox('Check to view sample data and histograms'):

    source = st.selectbox(
        'Select Data Source: ',
        ('Demographic','Dietary','Examination','Laboratory','Questionnaire')
    )

    if source == 'Demographic':
        files = files_demog
    elif source == 'Dietary':
        files = files_diet
    elif source == 'Examination':
        files = files_exam
    elif source == 'Laboratory':
        files = files_lab
    elif source == 'Questionnaire':
        files = files_quest
    elif source == 'Mortality':
        files = files_mort
    else:
        files = None

    table_1 = st.selectbox(
        'Select Table:',
        tuple(files)
    )
    url = table_1.replace('.XPT','.htm')
    st.markdown("[Open table documentation from CDC](%s)" % url)

    df = pd.read_sas('data/'+table_1.rsplit('/',1)[1])

    st.dataframe(df.head())

    col_list = []
    for field in field_list:
        for col in df.columns:
            if col in field:
                col_list.append(field)

    value = st.selectbox(
        'Select column for graphs:',
        tuple(sorted(list(set(col_list))))
    )

    plot_col = ''
    for col in df.columns:
        if col in value:
            plot_col = col

    fig = plt.figure(figsize=(10,4))
    sns.histplot(data=df, x=plot_col)
    st.pyplot(fig)

    if st.checkbox('Check to show multiperiod histogram'):
        files_local = ['data/'+file.rsplit('/',1)[1] for file in files]
        periods = [file.split("/")[-2] for file in files]
        df_agg = pd.DataFrame()
        for period, file_local in zip(periods, files_local):
            df_tmp = pd.read_sas(file_local)
            if plot_col in df_tmp.columns:
                df_agg_tmp = pd.DataFrame(
                    {
                        plot_col: df_tmp[plot_col],
                        'period': period
                    }
                )
                df_agg = pd.concat([df_agg, df_agg_tmp])

        fig = plt.figure(figsize=(10,4))
        sns.histplot(data=df_agg, x=plot_col, hue='period', multiple='dodge')
        st.pyplot(fig)

## ---
st.subheader('Create required intermediate tables')

def create_multiperiod_tables():
    files_full = glob.glob('data/*.XPT')
    files_full = sorted(files_full)
    files_short = [file.rsplit('/',1)[1] for file in files_full]
    files_short = [file.rsplit('_',1)[0] for file in files_short]
    files_short = [file.replace('.XPT','') for file in files_short]
    
    files_full = [x for _, x in sorted(zip(files_short, files_full))]
    files_short = sorted(files_short)
    
    file_short = ''
    df = pd.DataFrame()
    for file_short, file_full, next_file_short in zip(files_short, files_full, files_short[1:]):
        outfile = 'data/main_' + file_short + '.csv'
        if not os.path.exists(outfile):  
            df_tmp = pd.read_sas(file_full).copy()
            df_tmp['data_source'] = file_full
            df = df.append(df_tmp)
            if next_file_short != file_short:
                df.to_csv(outfile, index=False)
                print('Created file: ', outfile)
                df = pd.DataFrame()

def create_column_name_map(files):
    data_col_map = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, low_memory=False, nrows=0)
        df_tmp = pd.DataFrame(df.columns, columns=['column_name']) 
        df_tmp['table'] = file
        data_col_map = pd.concat([data_col_map, df_tmp])
    return data_col_map

def stack_mortality(files_mort):
    df_mort = pd.DataFrame()
    for file in files_mort:
        colspecs = [(0, 14), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (21, 22), (22, 26), (26, 34), (34, 42), (42, 45), (45, 48)]
        names = ['publicid', 'eligstat', 'mortstat', 'ucod_leading', 'diabetes', 'hyperten', 'dodqtr', 'dodyear', 'wgt_new', 'sa_wgt_new', 'permth_int', 'permth_exm']
        df_tmp = pd.read_fwf(file, colspecs=colspecs, header=None, names=names)
        df_tmp['data_source'] = file.rsplit('/', 1)[1]
        df_mort = pd.concat([df_mort, df_tmp])
    df_mort.to_csv('data/main_mort.csv', index=False)
    return df_mort

if st.checkbox('Check to create multiperiod tables'):
    data_load_state = st.text('Combining data...')
    create_multiperiod_tables()
    data_load_state.text("Combining data... Done!")

if st.checkbox('Check to create multiperiod mortality'):
    data_load_state = st.text('Combining data...')
    if not os.path.exists('data/main_mort.csv'): stack_mortality(files_mort)
    data_load_state.text("Combining data... Done!")

## ---
st.subheader('Mortality Modeling')

if st.checkbox('Check to show available features.json'):
    with open('features.json', 'r') as f:
        feature_meta = json.load(f)
    st.json(feature_meta)

def create_modeling_data(feature_meta):
    df = pd.DataFrame()
    for feature in feature_meta['features']:
        df_tmp = pd.read_csv(feature['table_name'])
        df_tmp = df_tmp.rename(columns={'SEQN':'id'})
        df_tmp = df_tmp.rename(columns={'publicid':'id'})
        df_tmp['id'] = df_tmp['id'].astype(int)

        df_tmp = df_tmp.rename(columns={feature['column_name']:feature['alias']})
        if feature['fillna']:
            df_tmp[feature['alias']] = df_tmp[feature['alias']].fillna(feature['fillna'])
            
        if feature['alias'] == 'id':
            df[feature['alias']] = df_tmp[feature['alias']]
        else:
            cols = ['id']+[feature['alias']]
            print(cols)
            if feature['encoding']:
                df_tmp[feature['alias']] = df_tmp[feature['alias']].replace(ast.literal_eval(feature['encoding']))
                if feature['encoding_names']:
                    enc_col = feature['alias']+'_values'
                    cols = cols + [enc_col]
                    df_tmp[enc_col] = df_tmp[feature['alias']].replace(ast.literal_eval(feature['encoding_names']))
            if feature['type']:
                print(feature['alias'])
                df_tmp[feature['alias']] = df_tmp[feature['alias']].astype(feature['type'])
            df = pd.merge(df, df_tmp[cols], on='id', how='left') 
            print(df.shape)
            
    return df

st.text('Edit features.json to customize available data fields')

if st.checkbox('Check to create modeling data'):

    write_state = st.text('Creating data/modeling.p...')

    with open('features.json', 'r') as f:
        feature_meta = json.load(f)
    
    df = create_modeling_data(feature_meta)

    # Population Filter
    df = df[df['mort_elig']==1]

    # Replace with more robust create features function
    df['bp_sys_ave'] = df[['BPXSY1','BPXSY2', 'BPXSY3','BPXSY4']].mean(axis=1)
    df['exposure_months'] = df['exposure_months'].replace({'.':0}).fillna(0).astype('int')
    df['duration'] = (df['exposure_months'].fillna(0)/12).astype(int)+1
    df['mort_elig'] = df['mort_elig'].astype(int)
    df['death'] = df['death'].astype(int)
    df['pir'] = df['pir'].round(1)
    
    bins = [0,10,20,30,40,50,60,70,80,90,100]
    labels = ["00-09","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-99"]
    df['age_band_10'] = pd.cut(df['age'], bins, labels=labels, include_lowest=True, right=False)

    bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    labels = [
        "00-04","05-09","10-14","15-19","20-24","25-29",
        "30-34","35-39","40-44","45-49","50-54","55-59",
        "60-64","65-69","70-74","75-79","80-84","85-89","90-94","95-99"
    ]
    df['age_band_5'] = pd.cut(df['age'], bins, labels = labels, include_lowest = True, right=False)

    df['cohort_5'] = df['gender_values'].astype('str')+"-"+df['age_band_5'].astype('str')
    df['cohort_10'] = df['gender_values'].astype('str')+"-"+df['age_band_10'].astype('str')
    # End feature engineering

    st.dataframe(df.head())

    df = df.reset_index()

    df.to_pickle('data/modeling.p')
    write_state.text('Creating data/modeling.p...Done!')
    st.text('Shape of df: '+ str(df.shape))
    st.text('Number of deaths: '+ str(df.death.fillna(0).replace({".":0}).astype('str').astype('int').sum()))

if st.checkbox('Check to explore modeling data'):
    df = pd.read_pickle('data/modeling.p')
    st.dataframe(df.head(100))

    col = st.selectbox(
        'Select field to show frequency distribution',
        tuple(df.columns[1:]),
        index=23
    )

    if str(df[col].dtype) != "float64":
        df[col] = df[col].astype(str)
    
    fig = plt.figure(figsize=(10,6))
    sns.histplot(data=df, x=col)
    st.pyplot(fig)

    df['Is Null'] = df[col].isnull()
    dist = df.groupby(['Is Null']).agg(
        count = ('id','count'),
        exposure_months = ('exposure_months', 'sum'),
        death = ('death', 'sum')
        )
    dist = dist.reset_index()
    dist['Frequency'] = dist['count'] / dist['count'].sum()
    dist['Qx'] = dist['death'] / (dist['exposure_months'] / 12) * 1000

    st.dataframe(
        dist.head()
    )
    col_tmp = col
    if str(df[col_tmp].dtype) == "float64":
        col_tmp = col+'_bin'
        df[col_tmp] = pd.qcut(df[col], q=[0,0.25,0.5,0.75,1])
    dist = df.groupby([col_tmp]).agg(
        count = ('id','count'),
        exposure_months = ('exposure_months', 'sum'),
        death = ('death', 'sum')
        )
    dist = dist.reset_index()
    dist['Frequency'] = dist['count'] / dist['count'].sum()
    dist['Qx'] = dist['death'] / (dist['exposure_months'] / 12) * 1000

    st.dataframe(dist.head(50))

    fig = plt.figure(figsize=(10,6))
    sns.barplot(data=dist, x=col_tmp, y='Qx', palette='Blues')
    st.pyplot(fig)

if st.checkbox('Check to create model'):
    text = st.text('Reading data...')
    df = pd.read_pickle('data/modeling.p')
    text.text('Reading data...Done!')

    # Population Filters
    # None for now
    if st.checkbox('Check to show parameters'):
        with open('params.json', 'r') as f:
            params = json.load(f)
        st.json(params)
    st.text('Edit params.json to customize model parameters')

    features = st.multiselect(
        'Select model features',
        list(df.columns),
        default=['age','gender']
    )

    num_boost_round = st.slider('Number of training iterations:', min_value=0, max_value=10000, step=100)

    train_button = st.button('Train Model')
    if train_button:
        my_bar = st.progress(0)
        text = st.text('Training...')

        X = df[features]
        y = np.where(df['death']==1, df['exposure_months'], df['exposure_months']*-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        xgb_full = xgboost.DMatrix(X, label=y)
        xgb_train = xgboost.DMatrix(X_train, label=y_train)
        xgb_test = xgboost.DMatrix(X_test, label=y_test)

        def update_bar(env):
            percent_complete = round(env.iteration/num_boost_round,2)
            my_bar.progress(percent_complete)

        with open('params.json', 'r') as f:
            params = json.load(f)

        model = xgboost.train(
            params, 
            xgb_train, 
            num_boost_round = num_boost_round, 
            evals = [(xgb_test, "test")],
            verbose_eval=100,
            early_stopping_rounds=5,
            callbacks = [update_bar]
        )
        my_bar.progress(100)
        text.text('Training...Done!')
        
        st.text("Best Iteration: "+str(model.best_iteration))

        c_index = concordance_index(df['exposure_months'], model.predict(xgb_full), df['death'])
        st.text("C-Index: "+str(round(1-c_index,5)))

        outfile = 'model/model.json'
        model.save_model(outfile)

        text = st.text("Calculating shap values...")
        cols=[feature+'_shap' for feature in features]
        shap_values = shap.TreeExplainer(model).shap_values(X)
        shap_values = pd.DataFrame(shap_values, columns=cols)

        df = pd.concat([df, shap_values], axis=1)
        text.text("Calculating shap values...Done!")

        df['pred'] = model.predict(xgb_full)
        df['pred_pct'] = df.pred.rank(pct=True)
        df['pred_pct_cohort_5'] = df.groupby("cohort_5")["pred"].rank(pct=True)
        df['pred_pct_cohort_10'] = df.groupby("cohort_10")["pred"].rank(pct=True)
        df['risk_score'] = df['pred_pct_cohort_10']

        file = 'expected_tables/2015-vbt-unismoke-alb-anb.xlsx'
        df_m_2015 = pd.read_excel(file, sheet_name = '2015 Male Unismoke ANB', skiprows=2)
        df_m_2015 = df_m_2015[list(set(df_m_2015.columns) - set(['Ult.','Att. Age']))]
        df_m_2015 = df_m_2015.melt(id_vars = ['Iss. Age'], value_name='Mortality Rate per 1,000', var_name='Duration')
        df_f_2015 = pd.read_excel(file, sheet_name = '2015 Female Unismoke ANB', skiprows=2)
        df_f_2015 = df_f_2015[list(set(df_f_2015.columns) - set(['Ult.','Att. Age']))]
        df_f_2015 = df_f_2015.melt(id_vars = ['Iss. Age'], value_name='Mortality Rate per 1,000', var_name='Duration')
        df_m_2015["gender"] = 0
        df_f_2015["gender"] = 1
        expected = pd.concat([df_m_2015, df_f_2015])
        expected['Duration'] = expected['Duration'].astype('int')
        expected['gender'] = expected['gender'].astype('int')
        expected = expected.rename(columns={'Iss. Age':'age', 'Duration':'duration'})

        df = pd.merge(df, expected, how='left', on=['age','gender','duration'])
        
        df['expected_deaths'] = (df['exposure_months'] / 12) * df['Mortality Rate per 1,000'] / 1000

        bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95,1.0]
        labels = ["00-19","20-39","40-59","60-79","80-89","90-94","95-100"]
        df['risk_group'] = pd.cut(df['pred_pct_cohort_10'], bins, labels=labels, include_lowest=True)

        st.dataframe(df.head())

        text = st.text('Saving data...')
        df.to_pickle('model/df_model.p')
        text.text('Saving data...Done! Data saved to: model/df_model.p')

st.subheader('Model Understanding and Evaluation')

if st.checkbox('Risk Score Distribution'):
    df = pd.read_pickle('model/df_model.p')

    col1, col2 = st.columns(2)
    with col1:
        pir_low = st.slider("Poverty to Income Ratio Min:", 0, 5, value=2, step=1)
    with col2:
        pir_high = st.slider("Poverty to Income Ratio Max: ", 0, 5, value=5, step=1)

    col1, col2 = st.columns(2)
    with col1:
        eth_low = st.slider("Race/Ethnicity Min:", 0, 5, value=3, step=1)
    with col2:
        eth_high = st.slider("Race/Ethnicity Max: ", 0, 5, value=4, step=1)

    df = df[(df['pir']>=pir_low) & (df['pir']<=pir_high)]
    df = df[(df['race']>=eth_low) & (df['race']<=eth_high)]

    fig = plt.figure(figsize=(10,6))
    sns.histplot(data=df, x="risk_score")
    st.pyplot(fig)

if st.checkbox('Actual To Expected by Risk Score Group'):
    df = pd.read_pickle('model/df_model.p')

    ae = df.groupby(['gender','risk_group']).agg(
        life_years = ('exposure_months','sum'),
        expected_deaths = ('expected_deaths','sum'), 
        actual_deaths = ('death','sum')
    )
    ae['ae'] = ae['actual_deaths']/ae['expected_deaths']
    ae['life_years'] = ae['life_years']/12
    ae = ae.reset_index()

    fig = plt.figure(figsize=(10,6))
    sns.barplot(data=ae, x='risk_group', y='ae', hue='gender')
    st.pyplot(fig)

if st.checkbox('SHAP Importance'):
    df = pd.read_pickle('model/df_model.p')
    
    # This one just for the order :(
    cols = [col for col in df.columns if "_shap" in col]
    df_shap = pd.DataFrame()
    for col in cols:
        df_tmp = pd.DataFrame()
        df_tmp["Importance"] = df[col].abs()
        df_tmp["Feature"] = col.replace('_shap','')
        df_shap = pd.concat([df_shap, df_tmp], axis=0)
    df_shap = df_shap.groupby(['Feature']).agg(Importance = ('Importance','sum')).reset_index()
    df_shap = df_shap.sort_values(by=['Importance'], ascending=False)
    order = list(df_shap['Feature'])
    order = [col for col in order if col not in ['age', 'gender']]

    col1, col2 = st.columns(2)
    with col1:
        age_low = st.slider("Age Min:",0,100, value=0, step=5)
    with col2:
        age_high = st.slider("Age Max: ",0,100, value=100, step=5)

    col1, col2 = st.columns(2)
    with col1:
        gender_low = st.slider("Gender Min:", 0, 1, value=0, step=1)
    with col2:
        gender_high = st.slider("Gender Max: ", 0, 1, value=1, step=1)

    df = df[(df['age']>=age_low) & (df['age']<=age_high)]
    df = df[(df['gender']>=gender_low) & (df['gender']<=gender_high)]

    cols = [col for col in df.columns if "_shap" in col]
    df_shap = pd.DataFrame()
    for col in cols:
        df_tmp = pd.DataFrame()
        df_tmp["Importance"] = df[col].abs()
        df_tmp["Feature"] = col.replace('_shap','')
        df_shap = pd.concat([df_shap, df_tmp], axis=0)

    df_shap = df_shap.groupby(['Feature']).agg(Importance = ('Importance','sum')).reset_index()
    df_shap = df_shap.sort_values(by=['Importance'], ascending=False)
    
    fig = plt.figure(figsize=(10,3))
    sns.barplot(data=df_shap, x='Feature', y='Importance', palette='Blues_r', order=["age","gender"])
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,len(order)/2))
    sns.barplot(data=df_shap, x='Importance', y='Feature', palette='Blues_r', order=order)
    st.pyplot(fig)

if st.checkbox('SHAP Dependency'):
    df = pd.read_pickle('model/df_model.p')

    cols = [col.replace('_shap','') for col in df.columns if '_shap' in col]
    extra_cols = ['', '', '', '',]

    col1, col2, col3 = st.columns(3)

    with col1:
        col = st.selectbox(
            'Select column for SHAP: ',
            tuple(cols),
            index=0
        )
    with col2:
        col_2 = st.selectbox(
            'Select column for color: ',
            tuple(cols),
            index=1
        )
    with col3:
        s = st.slider('Select dot size', min_value=10, max_value=70, value=20, step=5)

    fig = plt.figure(figsize=(10,6))
    scr = sns.scatterplot(
        data=df,
        x=col, 
        y=col+'_shap',
        hue=col_2,
        s=s
    )
    st.pyplot(fig)

