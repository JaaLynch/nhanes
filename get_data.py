import wget
import os

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

# Loop through the provided a list of paths
def batch_download_data(files):
    for file in files:
        download_data(file)

# Run the program
files = [
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DS1TOT_F.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DS2TOT_F.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BPX_F.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BMX_F.XPT'
]

batch_download_data(files)

files_demo = [
    'https://wwwn.cdc.gov/Nchs/Nhanes/1999-2000/DEMO.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2001-2002/DEMO_B.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DEMO_C.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/DEMO_D.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2007-2008/DEMO_E.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT',
    'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.XPT'
]



