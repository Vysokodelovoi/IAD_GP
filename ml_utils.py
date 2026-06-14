import pandas as pd
import time
import json
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
RESULTS_FILE = 'experiment_results.parquet'

def load_results():
    try:
        return pd.read_parquet(RESULTS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'method', 'params', 'score', 'duration_sec'])

def save_results(df):
    df.to_parquet(RESULTS_FILE, index=False)

def run_experiment(pipeline, method_name, param_dict=None):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time
    score = f1_score(y_test, pipeline.predict(X_test))
    
    
    result_row = {
        'timestamp': timestamp,
        'method': method_name,
        'params': json.dumps(param_dict),
        'score': score,
        'duration_sec': duration,
    }
    
    results_df = load_results()
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    save_results(results_df)
    
    print(f"{method_name} | score: {score:.5f} time: {duration:.1f}s")
    return result_row

