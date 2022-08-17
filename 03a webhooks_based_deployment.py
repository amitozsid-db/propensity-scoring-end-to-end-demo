# Databricks notebook source
dbutils.widgets.text("experiment_name","03_Model Training-Experiment-3b073c72")

# COMMAND ----------

# MAGIC %run "./00_Overview & Configuration"

# COMMAND ----------

# MAGIC %run ./config/mlflow_rest_lib

# COMMAND ----------

experiment_name = dbutils.widgets.get('experiment_name')  #ENTER YOUR EXPERIMENT NAME HERE
model_name = experiment_name#'alternate_workflow_registry'

# COMMAND ----------

import mlflow
import databricks.automl
import pyspark.sql.functions as f
from databricks.feature_store import FeatureStoreClient, FeatureLookup

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

experiment_id = [a.experiment_id for a in client.list_experiments() if experiment_name in a.name ][0]

run_id = client.search_runs(experiment_id, order_by=["metrics.test_accuracy_score"], max_results=1)[0].info.run_id

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"

client = get_client()
# client.set_tag(run_id, key='db_table', value='e2e_mldemo.churn_features')
# client.set_tag(run_id, key='demographic_vars', value='seniorCitizen,gender_Female')

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

dbutils.notebook.run("03b_webhooks_setup", 60, {"model_name": model_name})

# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_name,
                   'version': model_details.version,
                   'stage': 'Staging',
                   'archive_existing_versions': 'true'}

mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
