import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events[['patient_id','event_id','timestamp']], mortality[['patient_id','label']]

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    count=events['patient_id'].value_counts()
    events_count=pd.DataFrame({'patient_id':count.index,'count':count.values})
    events=pd.merge(events_count,mortality,on='patient_id',how='left')
    avg_dead_event_count = round(events[events['label']==1]['count'].mean(),1)
    max_dead_event_count = events[events['label']==1]['count'].max()
    min_dead_event_count = events[events['label']==1]['count'].min()
    avg_alive_event_count = round(events[events['label']!=1]['count'].mean(),1)
    max_alive_event_count = events[events['label']!=1]['count'].max()
    min_alive_event_count = events[events['label']!=1]['count'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    count_unique=events.groupby('patient_id').timestamp.nunique()
    encounter_count=pd.DataFrame({'patient_id':count_unique.index,'count':count_unique.values})
    events=pd.merge(encounter_count,mortality,on='patient_id',how='left')
    avg_dead_encounter_count = round(events[events['label']==1]['count'].mean(),1)
    max_dead_encounter_count = events[events['label']==1]['count'].max()
    min_dead_encounter_count = events[events['label']==1]['count'].min()
    avg_alive_encounter_count = round(events[events['label']!=1]['count'].mean(),1)
    max_alive_encounter_count = events[events['label']!=1]['count'].max()
    min_alive_encounter_count = events[events['label']!=1]['count'].min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    count_date=events.groupby('patient_id')['timestamp'].agg(['max','min'])
    count_date['max']=pd.to_datetime(count_date['max'])
    count_date['min']=pd.to_datetime(count_date['min'])
    count_date['diff']=count_date['max']-count_date['min']
    count_date=pd.DataFrame(count_date['diff'])
    count_date.reset_index(inplace=True)
    events=pd.merge(count_date,mortality,on='patient_id',how='left')
    avg_dead_rec_len = round(events[events['label']==1]['diff'].mean().total_seconds() / (24 * 60 * 60),1)
    max_dead_rec_len = round(events[events['label']==1]['diff'].max().total_seconds() / (24 * 60 * 60),1)
    min_dead_rec_len = round(events[events['label']==1]['diff'].min().total_seconds() / (24 * 60 * 60),1)
    avg_alive_rec_len = round(events[events['label']!=1]['diff'].mean().total_seconds() / (24 * 60 * 60),1)
    max_alive_rec_len = round(events[events['label']!=1]['diff'].max().total_seconds() / (24 * 60 * 60),1)
    min_alive_rec_len = round(events[events['label']!=1]['diff'].min().total_seconds() / (24 * 60 * 60),1)

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
