import pandas as pd
import argparse
import datetime as dt
import time
import pickle
import numpy as np


def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--loop_index",
        type=int,
        help="which longitudian files are transformed - 24 files in total",
        required=True
    )
    args = parser.parse_args()
    return args

def main():
    args = cli_parser()
    n = args.loop_index   
    long = 0

    # LOAD AND PREPROCESS

    df = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc2.txt.'+str(n))
    df.loc[df['EVENT_AGE']<0,'EVENT_AGE']=0 # for some SES and GEO codes event age is negative, here those cases are corrected to 0
    labels = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/label.csv')
    sex = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features.csv',usecols=['FINREGISTRYID','sex'])
    demo = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/demographic_features2.csv')
    emigrated = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/emigrated.csv') #XXX


    ######## for recording participants stats

    if n == 1:
        all_Ids = sex.iloc[:300000,:]
    elif n == 24:
        all_Ids = sex.iloc[300000*23:,:]
    else:
        all_Ids = sex.iloc[300000*(n-1):300000*n,:]

        
    all_Ids = all_Ids.merge(labels, on='FINREGISTRYID', how='left')
    all_Ids = all_Ids[all_Ids['LABEL']!=2]
    all_no_dead = all_Ids.shape[0]
    all_Ids = all_Ids[all_Ids['sex'].notna()]
    all_no_sex = all_Ids.shape[0]
    all_Ids = all_Ids.merge(emigrated, on='FINREGISTRYID', how='left')
    all_Ids=all_Ids[all_Ids['live_abroad']!=1]
    all_no_emigrated = all_Ids.shape[0]

    #############################################################

    initial_n = df['FINREGISTRYID'].nunique() #######
    df = df[df['CODE'].notna()]
    df['year'] = df['PVM'].apply(lambda x : x[:4])
    df['year'] = df['year'].astype(int)
    df['month'] = df['PVM'].apply(lambda x : x[5:7])
    df['month'] = df['month'].astype(int)
    df = df[df['year']<=2017] # remove data before 2018
    df = df.drop(df[(df['year']==2017)&(df['month']>9)].index) # remove data from tge last three months of 2017 (time buffer)
    n_before2016_10 = df['FINREGISTRYID'].nunique() #######
    df.drop(['year','month'], axis = 1,inplace=True)


    df = df.merge(labels, on='FINREGISTRYID', how='left')
    df = df.merge(demo, on='FINREGISTRYID', how='left')
    df = df.merge(sex, on='FINREGISTRYID', how='left')

    df = df[df['LABEL']!=2]
    n_nodead_bef2018 = df['FINREGISTRYID'].nunique() #######
    df = df[df['sex'].notna()]
    n_no_sex = df['FINREGISTRYID'].nunique() #######

    df = df.merge(emigrated, on='FINREGISTRYID', how='left')
    df = df[df['live_abroad']!=1]
    emigr = df['FINREGISTRYID'].nunique()
    del df['live_abroad']
    del df['sex']

    df['AGE_YEAR']=df['EVENT_AGE'].astype(int)
    df['CODE'] = df['CODE'].astype(str)

    #######################
    # TOKENIZE CODES
    types_df = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/code_dict_no_rare_additional_omits.csv')
    types = dict(zip(types_df.Code, types_df.Token))
    code_list = df['CODE'].values.tolist()
    new_code_list = []
    for code in code_list:
        if code in types: new_code_list.append(types[code])
        else: print(code, 'CODE NOT IN THE LIST!!!!')
    df['CODE'] = new_code_list
    df['CODE'] = df['CODE'].astype(str)

    #######################


    # MAIN LOOP TO CONSTRUCT DF AND LISTS FOR RNN
    df = df.groupby('FINREGISTRYID')
    n_codes = 100
    lines = []
    #fset=[]

    wide_needed = False
    if wide_needed:
        times = 100
        lines_wide = [] #wide
        line_wide = [] #wide
        a = 1 #wide
        
    for g, f in df:

        # dataframe
        codes = []
        ages = []
        ID = g
        #sources = []
        labell = f['LABEL'].iloc[0]
        dem = f.iloc[0,6]
        unique_visits = f.AGE_YEAR.unique()
        for visit in unique_visits:
            visit_codes = list(set(f[f['AGE_YEAR']==visit]['CODE'].values.tolist())) # keep not all codes but only unique ones
            if len(visit_codes)>n_codes:
                long = long+1
                visit_codes=visit_codes[:n_codes] # if number of unique codes per visit exeeds 100, keep only 100 first codes.
            codes.append(';'.join(visit_codes))
            ages.append(str(visit))
            #sources.append(';'.join(f[f['AGE_YEAR']==visit]['SOURCE'].values.tolist()))
        line = [ID,labell,' '.join(ages),' '.join(codes),len(ages),dem] #' '.join(sources),
        lines.append(line)

        if wide_needed:
            line_wide.extend(line) #wide        
            if a%times==0:#wide
                lines_wide.append(line_wide) #wide 
                line_wide = [] #wide
            a+=1 #wide
        
        # nested list for pytroch RNN
    #    n_seq=[]
    #    for v in range(len(codes)):
    #        nv=[]
    #        nv.append([int(ages[v])])
    #        visit_codes = list(map(int, codes[v].split(";")))
    #        if len(visit_codes)>n_codes: visit_codes=visit_codes[:n_codes] # if number of codes per visit exeeds 100, keep only 100 first codes.
    #        nv.append(visit_codes)                      
    #        n_seq.append(nv)
    #    n_pt= [g,int(labell),n_seq,dem]
    #    fset.append(n_pt)
        
    # SAVE RESULTS
    grouped = pd.DataFrame(lines,columns = ["FINREGISTRYID","LABEL","AGE_YEARS", "CODES","N_visits",'DEMO'])
    grouped.to_csv('/data/projects/project_avabalas/RNN/preprocessing_new/grouped_DF_all_codes_100codes.csv.'+str(n),index=False)
    #pickle.dump(fset, open('/data/projects/project_avabalas/RNN/preprocessing_new/grouped_list_all_codes.pickle.'+str(n), 'wb'), -1)

    results = np.concatenate([[[n]],[[all_no_dead]],[[all_no_sex]],[[all_no_emigrated]],[[initial_n]],[[n_before2016_10]],[[n_nodead_bef2018]],[[n_no_sex]],[[emigr]],[[long]]], axis = 1)


    with open("/data/projects/project_avabalas/RNN/final_data_result3.csv", "a") as myfile:
        np.savetxt(myfile, results.astype(int), fmt='%i', delimiter=',', newline='\n')


if __name__ == "__main__":
    main()


# Bash to run this file in parallel

#set -x
#for num in $(seq -w 01 24); do
#	INDEX=$(echo $num | sed 's/^0*//')
#	python3 /data/project_avabalas/RNN/preprocessing/final_sets2.py --loop_index $INDEX &
#	if (( $INDEX % 5 == 0 )); then sleep 80m; fi
#done

