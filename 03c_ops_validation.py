# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Tests
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step5.png?raw=true" width=75%>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Model in Transition

# COMMAND ----------

# MAGIC %run ./config/mlflow_rest_lib

# COMMAND ----------

import mlflow, json
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
client = get_client()

# After receiving payload from webhooks, use MLflow client to retrieve model details and lineage
try:
  registry_event = json.loads(dbutils.widgets.get('event_message'))
  model_name = registry_event['model_name']
  version = registry_event['version']
  if 'to_stage' in registry_event and registry_event['to_stage'] != 'Staging':
    dbutils.notebook.exit()
except Exception:
  model_name = 'alternate_workflow_registry'
  version = "1"

print(model_name, version)

# Use webhook payload to load model details and run info
model_details = client.get_model_version(model_name, version)
run_info = client.get_run(run_id=model_details.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Validate prediction

# COMMAND ----------

# print(model_details)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC 
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC 
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

model_uri = f'models:/{model_name}/{version}'
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_name, version=version, key="has_signature", value=1)
else:
  client.set_model_version_tag(name=model_name, version=version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Demographic accuracy
# MAGIC 
# MAGIC How does the model perform across various slices of the customer base?

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Documentation 
# MAGIC Is the model documented visually and in plain english? 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description check
# MAGIC 
# MAGIC Has the data scientist provided a description of the model being submitted?

# COMMAND ----------

# If there's no description or an insufficient number of charaters, tag accordingly
if not model_details.description:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=0)
  print("Did you forget to add a description?")
elif not len(model_details.description) > 20:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=0)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=model_name, version=version, key="has_description", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Has the data scientist logged supplemental artifacts along with the original model?

# COMMAND ----------

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = client.download_artifacts(run_info.info.run_id, "", local_dir)

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=model_name, version=version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=model_name, version=version, key = "has_artifacts", value = 1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

results = client.get_model_version(model_name, version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC Notify the Slack channel with the same webhook used to alert on transition change in MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move to Staging or Archived
# MAGIC 
# MAGIC The next phase of this models' lifecycle will be to `Staging` or `Archived`, depending on how it fared in testing.

# COMMAND ----------

import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()


# COMMAND ----------

# If any checks failed, reject and move to Archived
if '0' in results or 'fail' in results: 
  reject_request_body = {'name': model_details.name, 
                         'version': model_details.version, 
                         'stage': 'Staging', 
                         'comment': 'Tests failed - check the tags or the job run to see what happened.'}
  
  mlflow_call_endpoint('transition-requests/reject', 'POST', json.dumps(reject_request_body))
  
else: 
  approve_request_body = {'name': model_details.name,
                          'version': model_details.version,
                          'stage': 'Staging',
                          'archive_existing_versions': 'true',
                          'comment': 'All tests passed!  Moving to staging.'}
  
  mlflow_call_endpoint('transition-requests/approve', 'POST', json.dumps(approve_request_body))
