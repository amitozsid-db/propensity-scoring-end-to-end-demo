# Databricks notebook source
# MAGIC %run "./00_Overview & Configuration"

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

import bamboolib as bam

# COMMAND ----------

config['dbfs_mount']

# COMMAND ----------

# DBTITLE 1,Transformations
bam

# COMMAND ----------

import pandas as pd; import numpy as np
df = pd.read_csv(r'/dbfs/tmp/amitoz_sidhu/propensity/bronze/transaction_data.csv', sep=',', decimal='.', nrows=100000)
# Step: Create new column 'amount_list' from formula 'SALES_VALUE-RETAIL_DISC-COUPON_DISC-COUPON_MATCH_DISC'
df['amount_list'] = df['SALES_VALUE']-df['RETAIL_DISC']-df['COUPON_DISC']-df['COUPON_MATCH_DISC']

# Step: Replace missing values
df[['amount_list']] = df[['amount_list']].fillna(0.0)

# Step: Replace missing values
df[['COUPON_DISC', 'COUPON_MATCH_DISC', 'RETAIL_DISC', 'SALES_VALUE']] = df[['COUPON_DISC', 'COUPON_MATCH_DISC', 'RETAIL_DISC', 'SALES_VALUE']].fillna(0.0)

# Step: Create new column 'campaign_coupon_discount' from formula 'COUPON_DISC*(-1)'
df['campaign_coupon_discount'] = df['COUPON_DISC']*(-1)

# Step: Set values of campaign_coupon_discount to value of column 'campaign_coupon_discount' where COUPON_MATCH_DISC == 0.0 and otherwise to 0.0
tmp_condition = df['COUPON_MATCH_DISC'] == 0.0
df.loc[tmp_condition, 'campaign_coupon_discount'] = df['campaign_coupon_discount']
df.loc[~tmp_condition, 'campaign_coupon_discount'] = 0.0



# COMMAND ----------

bam

# COMMAND ----------

import pandas as pd; import numpy as np
df = pd.read_csv(r'/dbfs/tmp/amitoz_sidhu/propensity/bronze/product.csv', sep=',', decimal='.', nrows=100000)
# Step: Inner Join with df where PRODUCT_ID=PRODUCT_ID
joined_df = pd.merge(df, df, how='inner', on=['PRODUCT_ID'])


