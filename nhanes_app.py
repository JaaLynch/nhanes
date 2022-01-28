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

def age_race_filters(df):
        col1, col2 = st.columns(2)
        with col1:
            pir_low = st.slider("Poverty to Income Ratio Min:", 0, 5, value=0, step=1)
            eth_low = st.slider("Race/Ethnicity Min:", 0, 5, value=0, step=1)   
        with col2:
            pir_high = st.slider("Poverty to Income Ratio Max: ", 0, 5, value=5, step=1)
            eth_high = st.slider("Race/Ethnicity Max: ", 0, 5, value=5, step=1)
        
        df = df[(df['pir']>=pir_low) & (df['pir']<=pir_high)]
        df = df[(df['race']>=eth_low) & (df['race']<=eth_high)]
        return df



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
    
    df['ex_vis_gl'] = df[['ex_vis_gl_n', 'ex_vis_gl_f']].max(axis=1)
    df['ex_vis_ac'] = df[['ex_vis_ac_r','ex_vis_ac_l']].max(axis=1)
    df['ex_vis_ker_cyl'] = df[['ex_vis_ker_cyl_r','ex_vis_ker_cyl_l']].max(axis=1)
    df['ex_vis_ker_rad'] = df[['ex_vis_ker_rad_r','ex_vis_ker_rad_l']].max(axis=1)
    df['ex_vis_ker_axs'] = df[['ex_vis_ker_axs_r','ex_vis_ker_axs_l']].max(axis=1)
    df['ex_vis_ker_pow'] = df[['ex_vis_ker_pow_r','ex_vis_ker_pow_l']].max(axis=1)
    df['ex_vis_axs'] = df[['ex_vis_axs_r','ex_vis_axs_l']].max(axis=1)
    df['ex_vis_cyl'] = df[['ex_vis_cyl_r','ex_vis_cyl_l']].max(axis=1)
    df['ex_vis_sph'] = df[['ex_vis_sph_r','ex_vis_sph_l']].max(axis=1)

    df['bmi_calc'] = df['ex_ht'] / (df['ex_wt'] ** 2)

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
    if st.checkbox('Check to show parameters'):
        with open('params.json', 'r') as f:
            params = json.load(f)
        st.json(params)
    st.text('Edit params.json to customize model parameters')

    features = st.multiselect(
        'Select model features',
        sorted(df.columns),
        default=['age','gender']
    )

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input(
                'Enter a name for your model: ' 
        )
        train_button = st.button('Train Model')
    with col2:
        if not model_name:
            model_name = 'blank'
        num_boost_round = st.slider('Number of training iterations:', min_value=0, max_value=5000, step=100)
        

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
model_selected = st.selectbox(
    'Select a model:',
    tuple(model_names)
)
if not model_selected:
    model_selected = 'baseline_age_gender'
model_file = 'model/model_' + model_selected + '.json'
model_df_file = 'model/model_df_' + model_selected + '.p'

c_index = c_index_on_df(model_df_file)
st.text("C-Index: "+str(round(1-c_index,4)))

if st.checkbox('Risk Score Distribution'):
    df = pd.read_pickle(model_df_file)
    df = age_race_filters(df)

    cols = df.columns
    if st.checkbox('Apply filter to population'):
        filter_col, filt_low, filt_high = create_filter_widgets(cols, key='risk_score_dist')
        df = df[(df[filter_col]>=filt_low) & (df[filter_col]<=filt_high)]

    def plot_risk_score(df, figsize=(10,6)):
        fig = plt.figure(figsize=figsize)
        ax = sns.kdeplot(data=df, x="risk_score")
        mean = round(df["risk_score"].mean(),3)
        median = round(df["risk_score"].median(),3)
        #plt.axvline(x=mean, c='black')
        plt.axvline(x=median, c='grey')
        ax.set_xlim(-0.2,1.2)
        st.pyplot(fig)
        st.text("Mean Risk Score: "+ str(mean))
        st.text("Median Risk Score (grey): "+ str(median))
    
    plot_risk_score(df)

if st.checkbox('Actual To Expected by Risk Score Group'):
    df = pd.read_pickle(model_df_file)
    df['All cohorts'] = 1
    cols = df.columns
    
    hue = st.selectbox(
        'Select variable used to color bars',
        tuple(['All cohorts','gender','age_band_5','age_band_10','cohort_5','cohort_10','pir_bin','race' ])
    )
    
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
        sns.barplot(data=ae, x='risk_group', y='cumulative_ae', hue=hue)
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
        fig = plt.figure(figsize=(10,3))
        ax = sns.barplot(data=df_shap, x='Feature', y='Importance', palette='Blues_r', order=["age","gender"])
        ax.set_ylim(0,xlim_age_gender*1.05)
        st.pyplot(fig)

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
            scr = sns.scatterplot(
                data=df,
                x=col, 
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