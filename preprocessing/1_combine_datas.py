import argparse
import pandas as pd
import datetime as dt

def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--loop_index",
        type=int,
        help="which detailed longitudianl and endpoint longitudinal files are colmined - 24 of each in total",
        required=True
    )
    args = parser.parse_args()
    return args

def main():    
    args = cli_parser()
    n = args.loop_index
    dob = pd.read_csv('/data/processed_data/dvv/Finregistry_IDs_and_full_DOB.txt')

    path_endp = '/data/projects/project_avabalas/RNN/preprocessing/endpoint_longitudinal.txt.'+str(n)
    path_detai = '/data/processed_data/endpointer/supporting_files/main/longitudinal.txt.'+str(n)
    path_new = '/data/projects/project_avabalas/RNN/preprocessing_new/all_long_features.csv.'+str(n)
    endp = pd.read_csv(path_endp)
    detai = pd.read_csv(path_detai, usecols=['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1','CATEGORY'])
    new = pd.read_csv(path_new)

    
    # Endpoints
    endp = endp.merge(dob, on='FINREGISTRYID', how='left') 
    endp['EVENT_AGE_D']=endp['EVENT_AGE']*365.24
    endp['EVENT_AGE_D'] = endp['EVENT_AGE_D'].astype(int)
    endp = endp.rename(columns={'DOB(YYYY-MM-DD)': 'DOB'})
    endp['DOB'] = pd.to_datetime(endp['DOB'])
    endp['PVM'] = endp.apply(lambda x: x.DOB + dt.timedelta(days=x.EVENT_AGE_D), axis=1)
    endp['PVM'] = endp['PVM'].dt.strftime('%Y-%m-%d')
    endp = endp.rename(columns={'EVENT_TYPE': 'SOURCE', 'ENDPOINT': 'CODE'})
    endp = endp.drop(['EVENT_YEAR','ICDVER','DOB','EVENT_AGE_D'], axis=1)
    endp = endp[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE']]
    endp['SOURCE']="E"

    # Drug purhcases
    purch = detai[detai['SOURCE']=='PURCH'].copy()
    purch = purch[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1']]
    purch['SOURCE']="D"
    purch = purch.rename(columns={'CODE1': 'CODE'})
    purch = purch[purch['CODE'].notna()]
    purch['CODE']=purch['CODE'].apply(lambda x: x[:5])

    # Nomesco
    nom = detai[(detai['SOURCE'].str.contains("OPER_"))&(detai['CATEGORY'].str.contains("NOM"))].copy()
    nom = nom[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1']]
    nom['SOURCE']="N"
    nom = nom.rename(columns={'CODE1': 'CODE'})
    nom['len']=nom['CODE'].apply(lambda x: len(x))
    nom = nom[nom['len']==5]
    nom['CODE']=nom['CODE'].apply(lambda x:x[:3])
    nom = nom.drop(['len'], axis=1)
    nom['CODE']=nom['CODE'].apply(lambda x: "N_"+x)
    
    # ICPC2
    icp = detai[(detai['SOURCE']=='PRIM_OUT')&(detai['CATEGORY'].str.contains("ICP"))].copy()
    icp = icp[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1']]
    icp['SOURCE']="C"
    icp = icp.rename(columns={'CODE1': 'CODE'})
    icp['len']=icp['CODE'].apply(lambda x: len(x))
    icp = icp[icp['len']==3].copy()
    icp = icp.drop(['len'], axis=1)
    icp['CODE']=icp['CODE'].apply(lambda x: "C_"+x)

    # SPAT
    spat = detai[(detai['SOURCE']=='PRIM_OUT')&(detai['CATEGORY'].str.contains("OP"))&(detai['CATEGORY'].str.contains("MOP")==False)].copy()
    spat = spat[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1']]
    spat['SOURCE']="S"
    spat = spat.rename(columns={'CODE1': 'CODE'})
    spat['len']=spat['CODE'].apply(lambda x: len(x))
    spat = spat[spat['len']==8]
    spat = spat.drop(['len'], axis=1)    
    
    # ICD Avohilmo
    icd = detai[(detai['SOURCE']=='PRIM_OUT')&(detai['CATEGORY'].str.contains("ICD"))].copy()
    icd = icd[['FINREGISTRYID', 'SOURCE', 'EVENT_AGE', 'PVM','CODE1']]
    icd['SOURCE']="I"
    icd = icd.rename(columns={'CODE1': 'CODE'})
    icd['CODE']=icd['CODE'].apply(lambda x: x[:3])
    icd['len']=icd['CODE'].apply(lambda x: len(x))
    icd = icd[icd['len']==3]
    icd = icd.drop(['len'], axis=1)
    icd['CODE']=icd['CODE'].apply(lambda x: "I_"+x)
        
    comb = pd.concat([endp,purch,nom,icp,spat,icd,new])
    comb = comb.sort_values(["FINREGISTRYID", "EVENT_AGE"], ascending = (True, True))

    w_path = '/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc.txt.'+str(n)
    comb.to_csv(w_path,index=False)

if __name__ == "__main__":
    main()


# Shell to run this file in parallel

#set -x
#for num in $(seq -w 01 24); do
#	INDEX=$(echo $num | sed 's/^0*//')
#	python3 /data/project_avabalas/RNN/preprocessing/newcodes/combine_datas.py --loop_index $INDEX &
#	if (( $INDEX % 5 == 0 )); then sleep 110m; fi
#done
