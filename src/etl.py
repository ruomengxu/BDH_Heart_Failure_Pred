import utils
import pandas as pd
import numpy as np
import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    events_mortality=pd.merge(events,mortality,on='patient_id',how='left')
    indx_date_alive=events_mortality[events_mortality['label']!=1].groupby('patient_id')['timestamp_x'].agg({'indx_date':'max'})
    indx_date_a=pd.DataFrame(indx_date_alive['indx_date'])
    indx_date_a['patient_id']=indx_date_alive.index
    mortality['indx_date']=pd.to_datetime(mortality['timestamp']).dt.date-pd.Timedelta(days=30)
    indx_date_d=mortality[['patient_id','indx_date']]
    indx_date=pd.concat([indx_date_d,indx_date_a])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv',columns=['patient_id', 'indx_date'],index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    events_indx=pd.merge(events,indx_date,on='patient_id',how='inner')
    index1=pd.to_datetime(events_indx['timestamp'])<=pd.to_datetime(events_indx['indx_date'])
    index2=pd.to_datetime(events_indx['timestamp'])>=pd.to_datetime(events_indx['indx_date'])-pd.Timedelta(days=2000)
    filtered_events=events_indx[index1.values & index2.values]
    filtered_events=filtered_events[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    # merge=pd.merge(filtered_events_df,feature_map,on='event_id',how='inner')
    # merge=merge.dropna(subset=['value'])
    merge=pd.merge(filtered_events_df,feature_map_df,on='event_id',how='inner')
    merge=merge.dropna(subset=['value'])
    df_lab=merge[merge['event_id'].str.contains('LAB')]
    lab_count=df_lab.groupby(['patient_id','idx'])['idx'].agg({'feature_value_':'count'})
    df_not_lab=merge[~merge['event_id'].str.contains('LAB')]
    not_lab_sum=df_not_lab.groupby(['patient_id','idx'])['value'].agg({'feature_value_':'sum'})
    df1=pd.DataFrame(lab_count)
    df1.reset_index(inplace=True)
    df2=pd.DataFrame(not_lab_sum)
    df2.reset_index(inplace=True)
    df=pd.concat([df1,df2])
    df['max']=df.groupby('idx')['feature_value_'].transform('max')
    df['feature_value']=df['feature_value_']/df['max']
    df.rename(columns={'idx':'feature_id'},inplace=True)
    aggregated_events=df[['patient_id', 'feature_id', 'feature_value']]
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features=aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
    events_mortality=pd.merge(events,mortality,on='patient_id',how='left')
    events_mortality=events_mortality.fillna(0)
    mortality=events_mortality.set_index('patient_id')['label'].to_dict()
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    for key in sorted(patient_features.keys()):
        deliverable1.write('%d '%mortality[key])
        deliverable2.write('%d %d '%(key,mortality[key]))
        for value in sorted(patient_features[key],key=lambda x:x[0]):
            deliverable1.write('%d:%.6f '%(int(value[0]),value[1]))
            deliverable2.write('%d:%.6f '%(int(value[0]),value[1]))
        deliverable1.write('\n')
        deliverable2.write('\n')

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()