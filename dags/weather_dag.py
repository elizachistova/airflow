from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
import requests 
import json
from datetime import datetime
import os
import pandas as pd
from training import compute_model_score, prepare_data, train_and_save_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from airflow.utils.task_group import TaskGroup


def fetch_data():
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        raise ValueError("API Key not found -> set API_KEY environment variable.")
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    cities = Variable.get(key="cities",deserialize_json=True) 
    parent_folder = '/app/raw_files'

    all_data = [] 

    for city in cities:
        params = {
        "q": city,
        "appid": API_KEY 
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if response.status_code == 200:
            all_data.append(data)  # Add the city's data to the list
            print(f"Data for {city} added successfully.")
        else:
            print(f"Error for {city}: {response.status_code}, {response.reason}")

    if all_data:
        timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')
        filename = os.path.join(parent_folder, f'{timestamp}.json')

        with open(filename, 'w') as file:
            json.dump(all_data, file, indent=4) 
        print(f"Data saved to {filename}")
    else:
        print("No valid data to save.")

def transform_data_into_csv(n_files=None, filename=None):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append(
                {
                    'temperature': data_city['main']['temp'],
                    'city': data_city['name'],
                    'pression': data_city['main']['pressure'],
                    'date': f.split('.')[0]
                }
            )

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    df.to_csv(os.path.join('/app/clean_data', filename), index=False)

def my_task_2():
    transform_data_into_csv(n_files=20, filename='data.csv')

def my_task_3():
    transform_data_into_csv(n_files=None, filename='fulldata.csv')

def linear_regression(**kwargs):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_lr = compute_model_score(LinearRegression(), X, y)
    kwargs['ti'].xcom_push(key='score_lr', value=score_lr)
    return score_lr

def decision_tree(**kwargs):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
    kwargs['ti'].xcom_push(key='score_dt', value=score_dt)
    return score_dt

def random_forest(**kwargs):
    X, y = prepare_data('/app/clean_data/fulldata.csv')
    score_rf = compute_model_score(RandomForestRegressor(), X, y)
    kwargs['ti'].xcom_push(key='score_rf', value=score_rf)
    return score_rf

def compare_models(**kwargs):
    ti = kwargs['ti']
    score_rf = ti.xcom_pull(key='score_rf')
    score_dt = ti.xcom_pull(key='score_dt')
    score_lr = ti.xcom_pull(key='score_lr')
    X, y = prepare_data('/app/clean_data/fulldata.csv')

    if (score_lr < score_dt and score_lr < score_rf):
        train_and_save_model(
            LinearRegression(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
    elif (score_dt < score_lr and score_dt < score_rf):
        train_and_save_model(
            DecisionTreeRegressor(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )
    else:
        train_and_save_model(
            RandomForestRegressor(),
            X,
            y,
            '/app/clean_data/best_model.pickle'
        )


with DAG(
    dag_id='weather_api_dag',
    description='weather dashboard',
    tags=['weather_api'],
    schedule_interval='* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False,
) as my_dag:

    task1 = PythonOperator(
            task_id="fetch_data",
            python_callable=fetch_data,
        )
    task2 = PythonOperator(
            task_id="transform_data_2",
            python_callable=my_task_2,
        )
    task3 = PythonOperator(
            task_id="transform_data_3",
            python_callable=my_task_3,
        )
    with TaskGroup('models') as models_group:
        task4a = PythonOperator(
            task_id="linear_regression",
            python_callable=linear_regression,
        )
        task4b = PythonOperator(
                task_id="decision_tree",
                python_callable=decision_tree,
            )
        task4c = PythonOperator(
                task_id="random_forest",
                python_callable=random_forest,
            )
    task5 = PythonOperator(
            task_id="comparemodels",
            python_callable=compare_models,
        )
    

    task1 >> [task2,task3]
    task3 >> models_group
    models_group >> task5