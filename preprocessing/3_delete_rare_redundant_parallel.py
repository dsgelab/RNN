import argparse
import pandas as pd
import datetime as dt

def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--loop_index",
        type=int,
        help="which  files are processed - 24 of each in total",
        required=True
    )
    args = parser.parse_args()
    return args

def main():    
    args = cli_parser()
    n = args.loop_index
    
    endp = pd.read_excel('/home/avabalas/scripts/FINNGEN_ENDPOINTS_DF8_Final_2021-09-02_OMITS.xlsx')
    omits = endp[((endp['OMIT']==2) | (endp['OMIT']==1)) & (endp['Modification_reason']!='EXMORE/EXALLC priorities') & (endp['NAME']!='DEATH')]['NAME'].unique()
    meds = endp[(endp['KELA_ATC'].notna()) & (endp['HD_ICD_10'].isna())]['NAME'].unique()
    composite = endp[(endp['COD_ICD_10'].isna()) & (endp['HD_ICD_10'].isna()) & (endp['HD_ICD_10'].isna()) & (endp['CANC_TOPO'].isna())
                 & (endp['KELA_ATC'].isna()) & (endp['KELA_REIMB'].isna()) & (endp['OPER_NOM'].isna()) & ~(endp['NAME'].str.contains('#_This_follow'))]['NAME'].unique()
    additional_omits = list(meds)+list(composite)
    new_omits = [item for item in additional_omits if item not in list(omits)] # will be faster than ussing additional_omits which contain already omited endpoints
    dicty = pd.read_csv('/data/projects/project_avabalas/RNN/preprocessing_new/code_dict.csv')
    rare = dicty[dicty['Count_ID']<=70]['Code'].unique().tolist()
    new_omits = new_omits+rare
    
    path_in = '/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc.txt.'+str(n)
    in_file = open(path_in)
    path_out = '/data/projects/project_avabalas/RNN/preprocessing_new/combined_endp_atc2.txt.'+str(n)
    out_file = open(path_out, "x")
    OUT_HEADER = in_file.readline()
    OUT_HEADER = ",".join(OUT_HEADER.rstrip("\n").split(','))
    print(OUT_HEADER, file=out_file)
    # write only not omited endpoints from a new list to an output file
    zzz = 0
    for row in in_file:
        records = row.rstrip("\n").split(',')
        if records[4] in new_omits:
            zzz = zzz+1
            continue            
        else:
            print(
                ",".join(records),
                file=out_file
            )
    in_file.close()
    out_file.close() 
    


if __name__ == "__main__":
    main()


# Shell to run this file in parallel

#set -x
#for num in $(seq -w 01 24); do
#	INDEX=$(echo $num | sed 's/^0*//')
#	python3 /data/projects/project_avabalas/RNN/preprocessing_new/3_delete_rare_redundant_paralel.py --loop_index $INDEX &
#done
