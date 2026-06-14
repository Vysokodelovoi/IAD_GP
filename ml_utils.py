import pandas as pd
import time
import json
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

RESULTS_FILE = 'experiment_results.parquet'

def load_results():
    try:
        return pd.read_parquet(RESULTS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'method', 'params', 'mean_score', 'duration_sec'])

def save_results(df):
    df.to_parquet(RESULTS_FILE, index=False)

def run_experiment(pipeline, method_name, param_dict=None, cv=5):
    """
    Запускает эксперимент и логирует результат.
    """
    if param_dict is None:
        param_dict = {}
    
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    df = pd.read_parquet('main_df_edited.parquet')
    X = df.drop(columns=['reservation_status', 'reservation_status_date', 'is_canceled']).copy()
    y = df['is_canceled']
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    
    duration = time.time() - start_time
    mean_score = scores.mean()
    
    
    result_row = {
        'timestamp': timestamp,
        'method': method_name,
        'params': json.dumps(param_dict),
        'mean_score': mean_score,
        'duration_sec': duration,
    }
    
    results_df = load_results()
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    save_results(results_df)
    
    print(f"{method_name} | score: {mean_score:.5f} time: {duration:.1f}s")
    return mean_score, scores, results_df

