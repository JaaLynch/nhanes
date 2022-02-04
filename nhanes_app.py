from this import d
from tracemalloc import stop
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import wget
import os
import glob
import json
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost, shap
from lifelines.utils import concordance_index
import re
from matplotlib import cm
import random

def plot_hist(df, plot_col, figsize=(10,4), hue=None, multiple='dodge'):
    fig = plt.figure(figsize=figsize)
    sns.histplot(data=df, x=plot_col, hue=hue, multiple=multiple)
    st.pyplot(fig)

def null_count_df(df, plot_col):
    df['Is Null'] = df[plot_col].isnull()
    dist = df.groupby(['Is Null']).agg(
        count = (df.columns[0],'count')
        )
    dist = dist.reset_index()
    dist['Frequency'] = dist['count'] / dist['count'].sum()
    st.dataframe(dist)

def null_deaths_df(col):
    df['Is Null'] = df[col].isnull()
    dist = df.groupby(['Is Null']).agg(
        count = ('SEQN','count'),
        exposure_months = ('exposure_months', 'sum'),
        death = ('death', 'sum')
        )
    dist = dist.reset_index()
    dist['Frequency'] = dist['count'] / dist['count'].sum()
    dist['Qx'] = dist['death'] / (dist['exposure_months'] / 12) * 1000
    return dist

def get_df_mort(file = 'data/main_mort.csv'):
    df_mort = pd.read_csv(file)
    df_mort = df_mort[df_mort['eligstat']==1]
    df_mort = df_mort.rename(columns={
        'publicid':'SEQN', 
        'permth_int':'exposure_months', 
        'mortstat':'death'
        }
    )
    df_mort['death'] = df_mort['death'].astype(int)
    df_mort['exposure_months'] = df_mort['exposure_months'].astype(float)
    return df_mort

@st.cache
def c_index_on_df(model_df_file):
    df = pd.read_pickle(model_df_file)
    c_index = concordance_index(df['exposure_months'], df['pred'], df['death'])
    return c_index

def create_filter_widgets(cols, key='test'):
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_col = st.selectbox(
            'First filter column',
            tuple(cols),
            key= key+'0'
        )
    with col2:
        filt_low = st.slider(
            filter_col+' min',
            df[filter_col].min(),
            df[filter_col].max(),
            value = df[filter_col].min(),
            key = key+'1'
        )
    with col3:
        filt_high = st.slider(
            filter_col+' max',
            df[filter_col].min(),
            df[filter_col].max(),
            value = df[filter_col].max(),
            key = key+'2'
        )
    return filter_col, filt_low, filt_high

def age_race_filters(key):
        col1, col2 = st.columns(2)
        with col1:
            pir_low = st.slider("Poverty to Income Ratio Min:", 0, 5, value=0, step=1,key=key+'0')
            eth_low = st.slider("Race/Ethnicity Min:", 0, 5, value=0, step=1)   
        with col2:
            pir_high = st.slider("Poverty to Income Ratio Max: ", 0, 5, value=5, step=1, key=key+'1')
            eth_high = st.slider("Race/Ethnicity Max: ", 0, 5, value=5, step=1)

        return pir_low, pir_high, eth_low, eth_high

def calc_c_index(df, exposure_col = 'exposure_months', death_col = 'death', pred_col = 'pred', train_test_col='train_test', features=['age','gender']):
    c_index = 1-concordance_index(df[exposure_col], df[pred_col], df[death_col])
    deaths = df['death'].sum()
    
    df_in = df[df[train_test_col] == 'Train']
    df_out = df[df[train_test_col] == 'Test']
    c_index_in = 1-concordance_index(df_in[exposure_col], df_in[pred_col], df_in[death_col])
    c_index_out = 1-concordance_index(df_out[exposure_col], df_out[pred_col], df_out[death_col])
    deaths_in = df_in['death'].sum()
    deaths_out =  df_out['death'].sum()
    
    count_in = df_in.shape[0] 
    count_out = df_out.shape[0]
    count = df.shape[0]

    count_in = df_in.shape[0] 
    count_out = df_out.shape[0]
    count = df.shape[0]

    null_count_in = df_in[df_in[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})].shape[0] 
    null_count_out = df_out[df_out[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})].shape[0]
    null_count = df[df[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})].shape[0]

    null_deaths = df[df[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})]['death'].sum()
    null_deaths_in = df_in[df_in[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})]['death'].sum()
    null_deaths_out = df_out[df_out[features].isna().replace([True,False], [0, 1]).product(axis=1).replace({0:True, 1:False})]['death'].sum()

    tmp = pd.DataFrame(
        [
            ["Train", round(c_index_in*100,1), count_in, deaths_in, null_count_in, null_deaths_in],
            ["Test", round(c_index_out*100,1), count_out, deaths_out, null_count_out, null_deaths_out],
            ["Total", round(c_index*100,1), count, deaths, null_count, null_deaths]
        ], columns=['Sample','C-Index','Count', 'Deaths','Null Count', 'Null Deaths'])
    st.dataframe(tmp)

def create_sample_df(df, wt_col='wt_mec', k=1000):
    population = df['id'].to_list()
    weights = df[wt_col].to_list()
    id_sample = random.choices(
        population = population,
        weights = weights,
        k=k
    )
    id_sample = pd.DataFrame(id_sample, columns=['id'])
    df = pd.merge(id_sample, df, on='id', how='left')
    return df

@st.cache
def get_model_names():
    model_dfs = glob.glob('model/model_df_*.p')
    model_names = [model_df.replace('model/model_df_','').replace('.p','') for model_df in model_dfs]
    model_names = sorted(model_names)
    return model_names



## -----------------
## -----------------
## -----------------
## -----------------
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


## ---------------
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

## ------
st.subheader('Raw Data: Sample and Histograms')

if st.checkbox('Check to view sample data and histograms'):

    # Creating the drop downs
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

    def explore_raw_data(table_1):
        df = pd.read_sas('data/'+table_1.rsplit('/',1)[1])

        st.dataframe(df.head())

        field_list = pd.read_csv('data/field_list.csv')
        field_list = field_list.field_list.values.tolist()

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

        plot_hist(df, plot_col)
        null_count_df(df, plot_col)

    explore_raw_data(table_1)

## ---
st.subheader('Create required intermediate tables')

def create_multiperiod_tables():
    files_full = glob.glob('data/*.XPT')
    files_full = sorted(files_full)
    files_short = [file.rsplit('/',1)[1] for file in files_full]
    files_short = [file.replace('.XPT','') for file in files_short]
    files_short = [file[:-2] if file[-2:-1]=="_" else file for file in files_short]
    files_full = [x for _, x in sorted(zip(files_short, files_full))]
    files_short = sorted(files_short)

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

    # to create the stacked intermediate table for RX
    # Woops move this out of mortality :(
    df = pd.read_csv('data/main_RXQ_RX.csv', low_memory=False)
    df = df[(df['RXD030']==1)|(df['RXDUSE']==1)]
    df = df.groupby(['SEQN']).agg(
        count1 = ('RXD295','max'),
        count2 = ('RXDCOUNT','max'),
        days1 = ('RXD260','sum'),
        days2 = ('RXDDAYS','sum')
    ).reset_index()
    df['rx_script_count'] = df[['count1','count2']].sum(axis=1)
    df['rx_total_days'] = df[['days1','days2']].sum(axis=1)
    df = df.rename(columns={'SEQN':'id'})
    df = df.drop(columns=['count1','count2','days1','days2'])
    df.to_csv('data/main_mort_rx.csv', index=False)

    df = pd.read_csv('data/main_RXQ_RX.csv', low_memory=False)
    df = df[(df['RXD030']==1)|(df['RXDUSE']==1)]
    df['rx_script_count'] = df[['RXD295','RXDCOUNT']].sum(axis=1)
    df['rx_total_days'] = df[['RXD260','RXDDAYS']].sum(axis=1)
    tmp = df.groupby(['RXDDRGID','FDACODE1']).agg(count=('SEQN','count')).reset_index()
    tmp = tmp.sort_values(by='count', ascending=False)
    tmp = tmp[tmp['FDACODE1']!="b''"]
    tmp = tmp[tmp['RXDDRGID']!="b''"]
    tmp['rx_class'] = tmp['FDACODE1'].str[2:4]
    tmp = tmp.drop(columns=['FDACODE1','count'])
    df = pd.merge(df, tmp, on='RXDDRGID', how='left')
    tmp = df
    cols = ['SEQN','rx_class']
    tmp[cols] = tmp[cols].fillna(".").astype(str)
    tmp = tmp.groupby(cols).agg(
        rx_script_count=('SEQN','count'),
        rx_total_days=('rx_total_days', 'sum')
    ).reset_index()
    keep_classes = ['03','05', '06', '09','10','17', '19']
    tmp['rx_class'] = np.where(tmp['rx_class'].isin(keep_classes), tmp['rx_class'],'00')
    tmp = tmp.pivot_table(index='SEQN',columns='rx_class',values=['rx_script_count','rx_total_days'], aggfunc='sum')
    tmp.columns = list(map("_".join, tmp.columns))
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={'SEQN':'id'})
    tmp.to_csv('data/main_mort_rx_class.csv', index=False)

    return df_mort, df

if st.checkbox('Check to create multiperiod tables'):
    data_load_state = st.text('Combining data...')
    create_multiperiod_tables()
    data_load_state.text("Combining data... Done!")

if st.checkbox('Check to create multiperiod mortality'):
    data_load_state = st.text('Combining data...')
    if not os.path.exists('data/main_mort.csv'): stack_mortality(files_mort)
    data_load_state.text("Combining data... Done!")

if st.checkbox('Explore multiperiod tables'):
    multiperiod_tables = sorted(glob.glob('data/main_*.csv'))
    multiperiod_table = st.selectbox(
        'Select multiperiod table',
        tuple(multiperiod_tables)
    )

    df = pd.read_csv(multiperiod_table)
    df_mort = get_df_mort(file='data/main_mort.csv')
    df = pd.merge(df, df_mort, on='SEQN', how='left')

    multiperiod_table_col = st.selectbox(
        'Select column',
        tuple(sorted(df.columns))
    )

    plot_hist(df, multiperiod_table_col)
    dist = null_deaths_df(multiperiod_table_col)
    st.dataframe(dist)


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
        if feature['fillna'] is not None:
            df_tmp[feature['alias']] = df_tmp[feature['alias']].fillna(feature['fillna'])
            
        if feature['alias'] == 'id':
            df[feature['alias']] = df_tmp[feature['alias']]
        else:
            cols = ['id']+[feature['alias']]
            print(cols)
            if feature['encoding'] is not None:
                df_tmp[feature['alias']] = df_tmp[feature['alias']].replace(ast.literal_eval(feature['encoding']))
                if feature['encoding_names']:
                    enc_col = feature['alias']+'_values'
                    cols = cols + [enc_col]
                    df_tmp[enc_col] = df_tmp[feature['alias']].replace(ast.literal_eval(feature['encoding_names']))
            if feature['type'] is not None:
                print(feature['alias'])
                df_tmp[feature['alias']] = df_tmp[feature['alias']].astype(feature['type'])
            df = pd.merge(df, df_tmp[cols], on='id', how='left') 
            print(df.shape)
            
    return df

st.text('Edit features.json to customize available data fields')

if st.checkbox('Check to create modeling data'):

    start = st.button('Create')
    if start:

        write_state = st.text('Creating data/modeling.p...')

        with open('features.json', 'r') as f:
            feature_meta = json.load(f)
        
        df = create_modeling_data(feature_meta)

        # Population Filter
        df = df[df['mort_elig']==1]
        df = df[df['age']>18]

        # Replace with more robust create features function
        df['bp_sys_ave'] = df[['BPXSY1','BPXSY2', 'BPXSY3','BPXSY4']].mean(axis=1)
        df['exposure_months'] = df['exposure_months'].replace({'.':0}).fillna(0).astype('int')
        df['duration'] = (df['exposure_months'].fillna(0)/12).astype(int)+1
        df['mort_elig'] = df['mort_elig'].astype(int)
        df['death'] = df['death'].astype(int)
        df['pir'] = df['pir'].round(1)
        df['pir_bin'] = df['pir'].round(0)
        
        bins = [0,10,20,30,40,50,60,70,80,90,100]
        labels = ["00-09","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-99"]
        df['age_band_10'] = pd.cut(df['age'], bins, labels=labels, include_lowest=True, right=False)
        df['age_band_10_values'] = pd.cut(df['age'], bins, labels=bins[:-1], include_lowest=True, right=False).fillna(0).astype(int)

        bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        labels = [
            "00-04","05-09","10-14","15-19","20-24","25-29",
            "30-34","35-39","40-44","45-49","50-54","55-59",
            "60-64","65-69","70-74","75-79","80-84","85-89","90-94","95-99"
        ]
        df['age_band_5'] = pd.cut(df['age'], bins, labels = labels, include_lowest = True, right=False)
        df['age_band_5_values'] = pd.cut(df['age'], bins, labels = bins[:-1], include_lowest = True, right=False).fillna(0).astype(int)

        df['cohort_5'] = df['gender_values'].astype('str')+"-"+df['age_band_5'].astype('str')
        df['cohort_10'] = df['gender_values'].astype('str')+"-"+df['age_band_10'].astype('str')
        
        #Sphere: The sphere (SPH) on your prescription indicates the lens power you need to see clearly. 
        #  A minus (-) symbol next to this number means you’re nearsighted, and a plus (+) symbol means 
        #  the prescription is meant to correct farsightedness.
        #Cylinder: The cylinder (CYL) number indicates the lens power needed to correct astigmatism. 
        #  If this column is blank, it means you don’t have an astigmatism.
        #Axis: An axis number will also be included if you have an astigmatism. 
        #  This number shows the angle of the lens that shouldn’t feature a cylinder power to correct your astigmatism.

        df['ex_vis_gl'] = df[['ex_vis_gl_n', 'ex_vis_gl_f']].max(axis=1)
        df['ex_vis_ac'] = df[['ex_vis_ac_r','ex_vis_ac_l']].max(axis=1)
        df['ex_vis_ker_cyl'] = df[['ex_vis_ker_cyl_r','ex_vis_ker_cyl_l']].max(axis=1)
        df['ex_vis_ker_rad'] = df[['ex_vis_ker_rad_r','ex_vis_ker_rad_l']].max(axis=1)
        df['ex_vis_ker_axs'] = df[['ex_vis_ker_axs_r','ex_vis_ker_axs_l']].max(axis=1)
        df['ex_vis_ker_pow'] = df[['ex_vis_ker_pow_r','ex_vis_ker_pow_l']].max(axis=1)
        df['ex_vis_axs'] = df[['ex_vis_axs_r','ex_vis_axs_l']].max(axis=1)
        
        df['ex_vis_cyl'] = df[['ex_vis_cyl_r','ex_vis_cyl_l']].sum(axis=1)
        
        df['ex_vis_sph_min'] = df[['ex_vis_sph_r','ex_vis_sph_l']].min(axis=1)
        df['ex_vis_sph_max'] = df[['ex_vis_sph_r','ex_vis_sph_l']].max(axis=1)

        df['ex_armc_to_arml'] = df['ex_armc'] / (df['ex_arml'])
        df['ex_leg_to_ht'] = df['ex_leg'] / df['ex_ht']
        df['ex_bmi_calc'] = df['ex_ht'] / (df['ex_wt'] ** 2)

        df['qs_alc_cnt_avg'] = np.where(df['qs_alc_cnt_avg']>36, 36, df['qs_alc_cnt_avg'])
        df['qs_alc_nday_5p']=np.where(df['qs_alc_nday_5p_new'].isnull(),df['qs_alc_nday_5p_old'],df['qs_alc_nday_5p_new'])
        df['qs_alc_nday_5p_life']=np.where(df['qs_alc_nday_5p_life_new'].isnull(),df['qs_alc_nday_5p_life_old'],df['qs_alc_nday_5p_life_new'])
        
        df['qs_hear_gen'] = np.where(df['qs_hear_gen_new'].isnull(),df['qs_hear_gen_med'],df['qs_hear_gen_new'])
        df['qs_hear_gen'] = np.where(df['qs_hear_gen'].isnull(),df['qs_hear_gen_old'],df['qs_hear_gen'])

        df['qs_hear_gun'] = np.where(df['qs_hear_gun_new'].isnull(),df['qs_hear_gun_med'],df['qs_hear_gun_new'])
        df['qs_hear_gun'] = np.where(df['qs_hear_gun'].isnull(),df['qs_hear_gun_old'],df['qs_hear_gun'])

        df['qs_drm_sun_resp'] = np.where(df['qs_drm_sun_resp_new'].isna(),df['qs_drm_sun_resp_old'],df['qs_drm_sun_resp_new'])
        df['qs_drm_sun_resp'] = np.where((df['age']<60)&(df['age']>=20), df['qs_drm_sun_resp'], np.nan)

        df['qs_drm_moles'] = np.where(df['qs_drm_moles_new'].isna(),df['qs_drm_moles_old'],df['qs_drm_moles_new'])
        df['qs_drm_moles'] = np.where((df['age']<60)&(df['age']>=20), df['qs_drm_moles'], np.nan)

        df['qs_drugs'] = np.where(df['qs_drugs_new'].isna(),df['qs_drugs_old'],df['qs_drugs_new'])
        df['qs_drugs'] = np.where((df['age']<60)&(df['age']>=20), df['qs_drugs'], np.nan)

        df['qs_health_ins'] = np.where(df['qs_health_ins_new'].isna(),df['qs_health_ins_old'],df['qs_health_ins_new'])
        df['qs_health_ntimes'] = np.where(df['qs_health_ntimes_new'].isna(),df['qs_health_ntimes_old'],df['qs_health_ntimes_new'])

        df['qs_kidney'] = np.where(df['qs_kidney_new'].isna(),df['qs_kidney_old'],df['qs_kidney_new'])

        df['qs_alc_any'] = np.where(df['qs_alc_any_new'].isna(),df['qs_alc_any_old'],df['qs_alc_any_new'])

        df['qs_alc_any'] = np.where(df['qs_alc_any_new'].isnull(),df['qs_alc_any_med'],df['qs_alc_any_new'])
        df['qs_alc_any'] = np.where(df['qs_alc_any'].isnull(),df['qs_alc_any_old'],df['qs_alc_any'])

        df['lab_alt'] = df[['lab_alt_new','lab_alt_med','lab_alt_old']].sum(axis=1, min_count=1)
        df['lab_ast'] = df[['lab_ast_new','lab_ast_med','lab_ast_old']].sum(axis=1, min_count=1)
        df['lab_bicarb'] = df[['lab_bicarb_new','lab_bicarb_med','lab_bicarb_old']].sum(axis=1, min_count=1)
        df['lab_chol'] = df[['lab_chol_new','lab_chol_med','lab_chol_old']].sum(axis=1, min_count=1)
        df['lab_ggt'] = df[['lab_ggt_new','lab_ggt_med','lab_ggt_old']].sum(axis=1, min_count=1)
        df['lab_glucose'] = df[['lab_glucose_new','lab_glucose_med','lab_glucose_old']].sum(axis=1, min_count=1)
        df['lab_phosphorus'] = df[['lab_phosphorus_new','lab_phosphorus_med','lab_phosphorus_old']].sum(axis=1, min_count=1)
        df['lab_protein'] = df[['lab_protein_new','lab_protein_med','lab_protein_old']].sum(axis=1, min_count=1)
        df['lab_sodium'] = df[['lab_sodium_new','lab_sodium_med','lab_sodium_old']].sum(axis=1, min_count=1)
        df['lab_triglyc'] = df[['lab_triglyc_new','lab_triglyc_med','lab_triglyc_old']].sum(axis=1, min_count=1)
        df['lab_uric_acid'] = df[['lab_uric_acid_new','lab_uric_acid_med','lab_uric_acid_old']].sum(axis=1, min_count=1)
        df['lab_ldh'] = df[['lab_ldh_new','lab_ldh_med','lab_ldh_old']].sum(axis=1, min_count=1)

        df['qs_sm_years'] =  df[['age','qs_sm_cig_age_start']].max(axis=1) - df['qs_sm_cig_age_start']
        
        # Attach rx_script_count and rx_total_days, and by class
        tmp = pd.read_csv('data/main_mort_rx.csv')
        df = pd.merge(df,tmp, on='id', how='left')
        df['rx_total_days'] = np.where(df['rx_total_days']>100000,100000,df['rx_total_days'])
        tmp = pd.read_csv('data/main_mort_rx_class.csv')
        df = pd.merge(df,tmp, on='id', how='left')

        # End feature engineering
        df = df.reset_index()

        st.dataframe(df.head())

        df.to_pickle('data/modeling.p')
        write_state.text('Creating data/modeling.p...Done!')
        st.text('Shape of df: '+ str(df.shape))
        st.text('Number of deaths: '+ str(df.death.fillna(0).replace({".":0}).astype('str').astype('int').sum()))

if st.checkbox('Check to explore modeling data'):
    df = pd.read_pickle('data/modeling.p')

    st.dataframe(df.head(100))

    col = st.selectbox(
        'Select field to show frequency distribution',
        tuple(sorted(df.columns[1:])),
        index=23
    )

    if str(df[col].dtype) != "float64":
        df[col] = df[col].astype(str)
    
    fig = plt.figure(figsize=(10,6))
    sns.histplot(data=df, x=col)
    st.pyplot(fig)

    if str(df[col].dtype) != "float64":
        df[col] = df[col].astype(str)
    if (str(df[col].dtype) != "float64") & (len(df[col].unique())>100):
        df[col] = df[col].astype(str)


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
        df[col_tmp] = pd.qcut(df[col], q=[0,0.25,0.5,0.75,1], duplicates='drop')
    if (str(df[col].dtype) == "float64") & (len(df[col].unique())<101):
        df[col_tmp] = df[col].astype(str)
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
    #if st.checkbox('Check to show parameters'):
    #    with open('params.json', 'r') as f:
    #        params = json.load(f)
    #    st.json(params)
    #st.text('Edit params.json to customize model parameters')

    features = st.multiselect(
        'Select model features',
        sorted(df.columns),
        default=['age_band_10_values','gender']
    )

    url = 'https://xgboost.readthedocs.io/en/stable/parameter.html'
    st.markdown("[XGBoost Hyperparameter Documentation](%s)" % url)

    col1, col2 = st.columns(2)
    with col1:
        num_boost_round = st.slider('Number of training iterations:', min_value=0, max_value=10000, value=5000, step=100)
        learning_rate = st.slider('Learning Rate x 10,000', 0, 100, value=10, step=1)
        learning_rate = learning_rate / 10000
        subsample = st.slider('XGB Subsample', 0.0, 1.0, 0.7, step=0.1)
        min_child_weight = st.slider('Min Child Weight', 0, 200, 1, 1)
        early_stop_rounds = st.slider('Early Stopping Rounds', 2, 8, 5, 1)
        model_name = st.text_input('Enter a name for your model:')
        if not model_name: model_name = 'blank'
        train_button = st.button('Train Model')

    with col2:
        wts = st.slider('Weighted Sample Size x 1,000', 0, 200, 0, step=20)
        wts = wts * 1000
        #if wts > 0: df = create_sample_df(df, wt_col='wt_mec', k=wts)  
        max_depth = st.slider('Max Depth', 2, 10, 6, step=1)
        gamma = st.slider('Gamma', 0, 100, 0, 1)
        max_delta_step = st.slider('Max Delta Step', 0, 10, 0, 1)
        shap_calc = st.select_slider('Calculate SHAP', tuple(['Yes','No']))

    params = {
        "eta":learning_rate,
        "max_depth": max_depth,
        "objective": "survival:cox",
        "subsample": subsample,
        "min_child_weight":min_child_weight,
        "gamma": gamma,
        "max_delta_step": max_delta_step
    }       

    if train_button:
        my_bar = st.progress(0)
        text = st.text('Training...')

        X = df
        y = np.where(df['death']==1, df['exposure_months'], df['exposure_months']*-1)

        test_size = 0.4
        k = int(wts*(1-test_size))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if wts > 0: 
            X_train = create_sample_df(X_train, wt_col='wt_mec', k=k)
            X_test = create_sample_df(X_test, wt_col='wt_mec', k=wts-k) 
            
            y_train = np.where(X_train['death']==1, X_train['exposure_months'], X_train['exposure_months']*-1)
            y_test = np.where(X_test['death']==1, X_test['exposure_months'], X_test['exposure_months']*-1) 

            X = pd.concat([X_train, X_test]) 
            y = np.concatenate((y_train, y_test))

        df = X.reset_index()

        xgb_full = xgboost.DMatrix(X[features], label=y)
        xgb_train = xgboost.DMatrix(X_train[features], label=y_train)
        xgb_test = xgboost.DMatrix(X_test[features], label=y_test)

        df['train_test'] = np.where(df['id'].isin(X_test.id.unique()), 'Test', 'Train')

        def update_bar(env):
            percent_complete = round(env.iteration/num_boost_round,2)
            my_bar.progress(percent_complete)

        #with open('params.json', 'r') as f:
        #    params = json.load(f)

        model = xgboost.train(
            params, 
            xgb_train, 
            num_boost_round = num_boost_round, 
            evals = [(xgb_test, "test")],
            verbose_eval=100,
            early_stopping_rounds=early_stop_rounds,
            callbacks = [update_bar]
        )
        my_bar.progress(100)
        text.text('Training...Done!')
        
        st.text("Best Iteration: "+str(model.best_iteration))

        df['pred'] = model.predict(xgb_full)
        df['pred_pct'] = df.pred.rank(pct=True)
        df['pred_pct_cohort_5'] = df.groupby("cohort_5")["pred"].rank(pct=True)
        df['pred_pct_cohort_10'] = df.groupby("cohort_10")["pred"].rank(pct=True)
        df['risk_score'] = df['pred_pct_cohort_10']
        
        calc_c_index(df, features=features)

        if shap_calc == "Yes":
            text = st.text("Calculating shap values...")
            cols=[feature+'_shap' for feature in features]
            shap_values = shap.TreeExplainer(model).shap_values(X[features])
            shap_values = pd.DataFrame(shap_values, columns=cols)

            df = pd.concat([df, shap_values], axis=1)
            text.text("Calculating shap values...Done!")

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

        st.text("Number of deaths in model: "+str(df.death.fillna(0).sum()))

        text = st.text('Saving data...')
        file_name = 'model/model_df_'+ model_name +'.p'
        df.to_pickle(file_name)
       
        outfile = 'model/model_'+ model_name +'.json'
        model.save_model(outfile)

        text.markdown(
            'Saving data...Done! Data saved to '+ file_name + '. Model saved to '+ outfile
        )

        st.dataframe(df.head())

st.subheader('Model Understanding and Evaluation')

model_files = glob.glob('model/model_*.json')
model_df_files = glob.glob('model/model_df*.p')
model_names = [file.replace('model/model_','').replace('.json','') for file in model_files]
model_names = sorted(model_names)
model_selected = st.selectbox(
    'Select a model:',
    tuple(model_names)
)
if not model_selected:
    model_selected = 'baseline_age_gender'
model_file = 'model/model_' + model_selected + '.json'
model_df_file = 'model/model_df_' + model_selected + '.p'

if st.button('C-Index'):
    df = pd.read_pickle(model_df_file)
    features = [col.replace('_shap','') for col in df.columns if "_shap" in col]
    calc_c_index(df, features=features)

if st.checkbox('Risk Score Distribution'):
    df = pd.read_pickle(model_df_file)
    pir_low, pir_high, eth_low, eth_high = age_race_filters('risk_score_dist')
    df = df[(df['pir']>=pir_low) & (df['pir']<=pir_high)]
    df = df[(df['race']>=eth_low) & (df['race']<=eth_high)]

    nplot = 1
    if st.checkbox('Add second model'):
        model_selected_2 = st.selectbox(
            'Select a second model:',
            tuple(model_names)
        )
        model_df_file_2 = 'model/model_df_' + model_selected_2 + '.p'
        df2 = pd.read_pickle(model_df_file_2)
        df2 = df2[(df2['pir']>=pir_low) & (df2['pir']<=pir_high)]
        df2 = df2[(df2['race']>=eth_low) & (df2['race']<=eth_high)]
        nplot=2

    cols = df.columns
    if st.checkbox('Apply filter to population'):
        filter_col, filt_low, filt_high = create_filter_widgets(cols, key='risk_score_dist')
        df = df[(df[filter_col]>=filt_low) & (df[filter_col]<=filt_high)]
        if nplot==2:
            df2 = df2[(df2[filter_col]>=filt_low) & (df2[filter_col]<=filt_high)]

    def plot_risk_score(df_list, figsize=(10,6),nplot=1):
        fig = plt.figure(figsize=figsize)
        median = round(df_list[0]["risk_score"].median(),3)
        ax = sns.kdeplot(data=df_list[0], x="risk_score", linestyle='-')
        plt.axvline(x=median, c='grey',linestyle='-')
        if nplot>1:
            median2 = round(df_list[1]["risk_score"].median(),3)
            ax = sns.kdeplot(data=df_list[1], x="risk_score", linestyle='--')
            plt.axvline(x=median2, c='grey', linestyle='--')
        
        ax.set_xlim(-0.2,1.2)
        st.pyplot(fig)
        st.text("Median - Model 1: "+ str(median))
        if nplot>1: 
            st.text("Median - Model 2: "+ str(median2))
            st.text("Delta: "+str(round(median2-median,3)))
    
    if nplot == 1:
        plot_risk_score([df], nplot=1)
    else:
        plot_risk_score([df, df2], nplot=2)

if st.checkbox('Actual To Expected by Risk Score Group'):
    df = pd.read_pickle(model_df_file)
    df['All cohorts'] = 1
    cols = df.columns

    model_dfs = glob.glob('model/model_df_*.p')
    model_names = [model_df.replace('model/model_df_','').replace('.p','') for model_df in model_dfs]
    model_names = sorted(model_names)
    nplot = 1
    if st.checkbox('Compare to another model'):
        model_selected_2 = st.selectbox(
            'Select the second model:',
            tuple(model_names)
        )
        model_df_file_2 = 'model/model_df_' + model_selected_2 + '.p'
        df2 = pd.read_pickle(model_df_file_2)
        nplot=2
    
    if nplot==1:
        hue = st.selectbox(
            'Select variable used to color bars',
            tuple(['All cohorts','train_test', 'gender','age_band_5','age_band_10','cohort_5','cohort_10','pir_bin','race' ])
        )
    else:
        hue = 'model_name'
        df['model_name'] = model_selected
        df2['model_name']= model_selected_2
        df = pd.concat([df,df2], axis=0)

    if st.checkbox('Apply filter to actual to expected chart'):
        filter_col, filt_low, filt_high = create_filter_widgets(cols, key='ae')
        df = df[(df[filter_col]>=filt_low) & (df[filter_col]<=filt_high)]

    def calculate_ae(df,hue):
        ae = df.groupby([hue,'risk_group']).agg(
            life_years = ('exposure_months','sum'),
            expected_deaths = ('expected_deaths','sum'), 
            actual_deaths = ('death','sum')
        )
        ae['ae'] = ae['actual_deaths']/ae['expected_deaths']*100
        ae['life_years'] = ae['life_years']/12
        ae['actual_deaths_cum'] = ae.groupby([hue])['actual_deaths'].cumsum()
        ae['expected_deaths_cum'] = ae.groupby([hue])['expected_deaths'].cumsum()
        ae['cumulative_ae'] = ae['actual_deaths_cum']/ae['expected_deaths_cum']*100
        ae = ae.reset_index()
        ae['life_years_sum']=ae[hue].map(ae.groupby(hue)["life_years"].sum())
        ae['Frequency'] = ae['life_years']/ae['life_years_sum']
        
        return ae

    ae = calculate_ae(df, hue)

    def plot_ae(ae,figsize=(10,3)):
        fig = plt.figure(figsize=figsize)
        sns.barplot(data=ae, x='risk_group', y='ae', hue=hue)
        st.pyplot(fig)

        fig = plt.figure(figsize=figsize)
        sns.lineplot(data=ae, x='risk_group', y='cumulative_ae', hue=hue)
        plt.legend([],[], frameon=False)
        st.pyplot(fig)
    
        fig = plt.figure(figsize=(figsize[0], figsize[1]/2))

        sns.barplot(data=ae, x='risk_group', y='Frequency', hue=hue)
        plt.legend([],[], frameon=False)
        st.pyplot(fig)

    plot_ae(ae)

    st.dataframe(ae)

if st.checkbox('SHAP Importance'):
    df = pd.read_pickle(model_df_file)

    def create_df_shap_imp(df):
        cols = [col for col in df.columns if "_shap" in col]
        cols = ['age' if 'age' in col else col for col in cols]
        df_shap = pd.DataFrame()
        for col in cols:
            df_tmp = pd.DataFrame()
            df_tmp["Importance"] = df[col].abs()
            df_tmp["Feature"] = col.replace('_shap','')
            df_shap = pd.concat([df_shap, df_tmp], axis=0)
        df_shap = df_shap.groupby(['Feature']).agg(Importance = ('Importance','sum')).reset_index()
        df_shap = df_shap.sort_values(by=['Importance'], ascending=False)
        return df_shap

    # Get no filter x limits
    df_shap = create_df_shap_imp(df)
    xlim_age_gender = df_shap[ (df_shap['Feature']=='age') | (df_shap['Feature']=='gender')]["Importance"].max()
    xlim_not_age_gender = df_shap[ (df_shap['Feature']!='age') & (df_shap['Feature']!='gender')]["Importance"].max()
    order = list(df_shap['Feature'])
    order = [col for col in order if col not in ['age', 'gender']]

    def age_gender_filter(df,key):
        col1, col2 = st.columns(2)
        with col1:
            age_low = st.slider("Age Min:",0,100, value=0, step=5, key=key+'0')
        with col2:
            age_high = st.slider("Age Max: ",0,100, value=100, step=5,key=key+'1')

        col1, col2 = st.columns(2)
        with col1:
            gender_low = st.slider("Gender Min:", 0, 1, value=0, step=1, key=key+'2')
        with col2:
            gender_high = st.slider("Gender Max: ", 0, 1, value=1, step=1, key=key+'3')

        df = df[(df['age']>=age_low) & (df['age']<=age_high)]
        df = df[(df['gender']>=gender_low) & (df['gender']<=gender_high)]

        return df

    df = age_gender_filter(df, 'shap_imp_age_gender')

    if st.checkbox('Apply filter to feature importance graphs'):
        filter_col, filt_low, filt_high = create_filter_widgets(df.columns, key='shap_imp')
        df = df[(df[filter_col]>=filt_low) & (df[filter_col]<=filt_high)]

    df_shap = create_df_shap_imp(df)

    df_shap = df_shap.groupby(['Feature']).agg(Importance = ('Importance','sum')).reset_index()
    df_shap = df_shap.sort_values(by=['Importance'], ascending=False)

    def plot_shap_imp(df_shap, xlim_age_gender, order):
        #fig = plt.figure(figsize=(10,3))
        #ax = sns.barplot(data=df_shap, x='Feature', y='Importance', palette='Blues_r', order=["age","gender"])
        #ax.set_ylim(0,xlim_age_gender*1.05)
        #st.pyplot(fig)

        if order:
            fig = plt.figure(figsize=(10,len(order)/2))
            ax = sns.barplot(data=df_shap, x='Importance', y='Feature', palette='Blues_r', order=order)
            ax.set_xlim(0,xlim_not_age_gender*1.005)
            st.pyplot(fig)
    
    plot_shap_imp(df_shap, xlim_age_gender, order)

if st.checkbox('SHAP Dependency'):
    df = pd.read_pickle(model_df_file)

    cols = [col.replace('_shap','') for col in df.columns if '_shap' in col]

    col1, col2 = st.columns(2)
    with col1:
        col = st.selectbox(
            'Select column for SHAP: ',
            tuple(cols),
            index=0
        )
        outliers = st.select_slider('Eliminate outliers:', options=['No','Yes'])

        x_max = df[col].max()
        x_min = df[col].min()
        x_lim_min = st.slider(col+" min", min_value=x_min , max_value=x_max, value = x_min)

    if outliers == "Yes":
        df = df[(df[col].rank(pct=True)>=0.01) & (df[col].rank(pct=True)<=0.99)]

    with col2:
        col_2 = st.selectbox(
            'Select column for color: ',
            tuple(sorted(df.columns)),
            index=sorted(df.columns).index('gender')
        )
        s = st.slider('Select dot size:', min_value=1, max_value=70, value=20, step=1)

        x_lim_max = st.slider(col+" max", min_value=x_min , max_value=x_max, value = x_max)

    h = (x_max - x_min) / 100
    def plot_dependence(df, col, col_2):
            fig = plt.figure(figsize=(10,6))
            col_tmp = col
            if 'age' in col_tmp: col_tmp='age'
            scr = sns.scatterplot(
                data=df,
                x=col_tmp, 
                y=col+'_shap',
                hue=col_2,
                s=s
            )
            scr.set(xlim=(x_lim_min-h, x_lim_max*1.02))
            st.pyplot(fig)

    if st.checkbox('Filter population'):
        filter_col, filt_low, filt_high = create_filter_widgets(cols, key='shap_dep')
        df = df[(df[filter_col]>=filt_low) & (df[filter_col]<=filt_high)]
        
    plot_dependence(df, col, col_2)

if st.checkbox('Compare Risk Scores - 3D Histogram'):

    model_dfs = glob.glob('model/model_df_*.p')
    model_names = [model_df.replace('model/model_df_','').replace('.p','') for model_df in model_dfs]
    model_names = sorted(model_names)

    col1, col2 = st.columns(2)

    with col1:
        model_name_1 = st.selectbox('Model 1: ', tuple(model_names), index=0)
        
    with col2:
        model_name_2 = st.selectbox('Model 2: ', tuple(model_names), index=1)

    model_df_1 = 'model/model_df_' + model_name_1 + '.p'
    model_df_2 = 'model/model_df_' + model_name_2 + '.p'

    bins = st.slider('Bin size: ',1,20,10)
    col1, col2 = st.columns(2)
    with col1:
        tilt = st.slider('Tilt figure: ', -30, 30, 0, 5)
    with col2:
        spin = st.slider('Spin the figure', -180, 180, 0, 5)

    df1 = pd.read_pickle(model_df_1)
    df2 = pd.read_pickle(model_df_2)

    xAmplitudes = df1['risk_score'] #your data here
    yAmplitudes = df2['risk_score']#your other data here

    x = np.array(xAmplitudes)   #turn x,y data into numpy arrays
    y = np.array(yAmplitudes)

    fig = plt.figure(figsize=(10,10))          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(x, y, bins=(bins, bins))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    ax.view_init(30+tilt,-110+spin*-1)
    plt.title("3D Histogram of Model Risk Scores \n Bivariate Distribution")
    plt.xlabel(model_name_1)
    plt.ylabel(model_name_2)
    #plt.savefig("Your_title_goes_here")
    st.pyplot(fig)

st.subheader('Calibrate Model Systems')

# Population filters and simulation/sampling

# Create Underwriting Rules (creates config file)

# Model 1, Model 2, Model 1 Class, Model 2 Class, Combined Class
# Goal is create a dataframe and output to csv a config file like the one above
# that represents the full recursive mapping between models

# Select from a grid of 6 models
# Select from a grid of 6 model class numbers
# 
# Select from a generated grid of percentile bin boundaries for each class (variable number of classes)

def create_custom_risk_groups(df, bins=[0, 0.25, 0.75, 1.0], cut_col='risk_score'):
    labels = []
    labels_numeric = []
    i = 1
    for bin, next_bin in zip(bins, bins[1:]):
        bin = int(round(bin*100,0))
        next_bin = int(round(next_bin*100,0))
        bin = str(bin).zfill(2)
        next_bin = str(next_bin).zfill(2)
        labels.append(bin+'-'+next_bin)
        labels_numeric.append(i)
        i = i+1
    df['risk_group_cstm'] = pd.cut(df[cut_col], bins, labels=labels, include_lowest=True).astype(str)
    df['risk_group_cstm_n'] = pd.cut(df[cut_col], bins, labels=labels_numeric, include_lowest=True)
    return df

def calculate_ae_multi(df, cols=['risk_group','risk_group']):
    ae = df.groupby(cols).agg(
        life_years = ('exposure_months','sum'),
        expected_deaths = ('expected_deaths','sum'), 
        actual_deaths = ('death','sum')
    )
    ae['ae'] =round(ae['actual_deaths']/ae['expected_deaths']*100,1)
    ae['life_years'] = ae['life_years']/12
    ae = ae.dropna(subset=['life_years', 'expected_deaths','ae'])
    ae = ae.reset_index()
    ae['Frequency'] = round(ae['life_years']/ae['life_years'].sum()*100,1)
    
    return ae

def plot_heatmaps(ae, figsize=(10,4)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1,2)
    
    ax = fig.add_subplot(gs[0,0])
    ax.set_title('AE 2015 VBT Unismoke')
    tmp = ae.pivot(index='class_model_1', columns='class_model_2', values='ae')
    ax = sns.heatmap(tmp, annot=True, fmt='.0f')

    ax = fig.add_subplot(gs[0,1])
    ax.set_title('Exposure Percent')
    tmp = ae.pivot(index='class_model_1', columns='class_model_2', values='Frequency')
    ax = sns.heatmap(tmp, annot=True, fmt='.0f')
    st.pyplot(fig)

def map_two_models(key='mtm', df_in=None):

    col1, col2 = st.columns(2)
    with col1:
        if df_in is not None:
            model_1 = st.selectbox('Combined score from mapping: ', tuple(['class_combined']),key='5'+str(key))
        else:
            model_1 = st.selectbox('Model 1: ', tuple(model_names), index=model_names.index('bmi_waist_pulse_armd_legp'), key='1'+str(key))
        score_cuts_1 = st.text_input('Score cuts', value='0.2, 0.8', key='2'+str(key)).split(",")
        score_cuts_1 = [0] + [float(score.strip()) for score in score_cuts_1] + [1]
    with col2:
        if df_in is not None:
            model_2_text = 'Additional model:'
        else:
            model_2_text = 'Model 2: '
        model_2 = st.selectbox(model_2_text, tuple(model_names), index=model_names.index('labs'), key='3'+str(key))
        score_cuts_2 = st.text_input('Score cuts', value='0.2, 0.8', key='4'+str(key)).split(",")
        score_cuts_2 = [0] + [float(score.strip()) for score in score_cuts_2] + [1]

    if df_in is not None:
        df1 = df_in
        df1 = create_custom_risk_groups(df1, bins=score_cuts_1, cut_col='class_combined')
        col1 = df1.class_combined

    else:
        df1 = pd.read_pickle('model/model_df_' + model_1 + '.p')
        df1 = create_custom_risk_groups(df1, bins=score_cuts_1)
        col1 = df1.risk_group_cstm_n
    
    df2 = pd.read_pickle('model/model_df_' + model_2 + '.p')
    df2 = create_custom_risk_groups(df2, bins=score_cuts_2)


    df = pd.DataFrame(zip(col1, df2.risk_group_cstm_n), columns=['class_model_1','class_model_2'])
    df = pd.concat([df, df1[['exposure_months', 'expected_deaths', 'death']]], axis=1)

    ae = calculate_ae_multi(df, cols =['class_model_1','class_model_2'])

    ae["model_1"] = model_1
    ae["model_2"] = model_2
    ae["rule"] = key

    return ae, df

def attach_class_mapping(ae, key='acm'):
    cells = ae.shape[0]
    value = [1] * cells
    map_1 = st.text_input('Class mapping (top left to bottom right)', value=value, key=str(key)).split(",")
    map_1 = [int(str(x).replace('[','').replace(']','')) for x in map_1]

    tmp = pd.DataFrame(map_1, columns=['class_combined'])
    ae = pd.concat([ae, tmp], axis=1)
    ae[['class_model_1', 'class_model_2', 'class_combined']] = ae[['class_model_1', 'class_model_2', 'class_combined']].astype(int)
    return ae

if st.checkbox('Create scoring system'):
    model_names = get_model_names()

    key=1
    ae, df = map_two_models(key=key) 
    plot_heatmaps(ae)
    ae = attach_class_mapping(ae, key=key)
    oncols = ['class_model_1', 'class_model_2']
    cols = ['class_combined']
    df = pd.merge(df, ae[oncols+cols], on=oncols, how='left')
    ae_cum = ae

    key=2
    ae, df = map_two_models(key=key, df_in = df) 
    plot_heatmaps(ae)
    ae = attach_class_mapping(ae, key=key)
    oncols = ['class_model_1', 'class_model_2']
    cols = ['class_combined']
    df = pd.merge(df, ae[oncols+cols], on=oncols, how='left')
    ae_cum = pd.concat([ae_cum, ae], axis=0)

    key=3
    ae, df = map_two_models(key=key, df_in = df) 
    plot_heatmaps(ae)
    ae = attach_class_mapping(ae, key=key)
    oncols = ['class_model_1', 'class_model_2']
    cols = ['class_combined']
    df = pd.merge(df, ae[oncols+cols], on=oncols, how='left')
    ae_cum = pd.concat([ae_cum, ae], axis=0)

    key=4
    ae, df = map_two_models(key=key, df_in = df) 
    plot_heatmaps(ae)
    ae = attach_class_mapping(ae, key=key)
    oncols = ['class_model_1', 'class_model_2']
    cols = ['class_combined']
    df = pd.merge(df, ae[oncols+cols], on=oncols, how='left')
    ae_cum = pd.concat([ae_cum, ae], axis=0)

    key=5
    ae, _ = map_two_models(key=key, df_in = df) 
    plot_heatmaps(ae)
    ae = attach_class_mapping(ae, key=key)
    ae_cum = pd.concat([ae_cum, ae], axis=0)

    system_name = st.text_input('Enter a name: ', value='')
    save = st.button('Save')
    if save:
        ae_cum.to_pickle('model/system_'+system_name+'.p')
        ae_cum.to_csv('model/system_'+system_name+'.csv', index=False)

if st.checkbox('Scoring system results'):

    system_files = glob.glob('model/system_*.p')
    system_names = [system_file.replace('model/system_','').replace('.p','') for system_file in system_files]
    system_names = sorted(system_names)

    system_name = st.selectbox('Select scoring system', tuple(system_names))

    file_name = 'model/system_' + system_name + '.p'

    df = pd.read_pickle(file_name)
    df = df.rename(columns={'actual_deaths':'death',  'life_years':'exposure_months'})
    df['exposure_months'] = df['exposure_months'] * 12

    ae = calculate_ae_multi(df, cols=['rule','class_combined'])
    ae['actual_deaths_cum'] = ae.groupby(['rule'])['actual_deaths'].cumsum()
    ae['expected_deaths_cum'] = ae.groupby(['rule'])['expected_deaths'].cumsum()
    ae['cumulative_ae'] = ae['actual_deaths_cum']/ae['expected_deaths_cum']*100
    ae = ae.reset_index()
    ae['life_years_sum']=ae['rule'].map(ae.groupby('rule')["life_years"].sum())
    ae['Frequency'] = ae['life_years']/ae['life_years_sum']

    genre = st.radio(
     "What would you like to plot",
     ('Progressive results', 'Final results'))

    if genre == 'Progressive results':
        x = 'rule'
        hue = 'class_combined'
    else:
        ae = ae[ae.rule==ae.rule.max()]
        x = 'class_combined'
        hue = None

    figsize = (10, 4)
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(data=ae, x=x, y='ae', hue=hue, palette='Blues')
    st.pyplot(fig)
    
    figsize = (10, 2)
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=ae, x=x, y='cumulative_ae', hue=hue, palette='Blues')
    plt.legend([],[], frameon=False)
    st.pyplot(fig)

    figsize = (10, 2)
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=ae, x=x, y='Frequency', hue=hue, palette='Blues')
    plt.legend([],[], frameon=False)
    st.pyplot(fig)

    ae





