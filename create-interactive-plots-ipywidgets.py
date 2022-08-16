# Databricks notebook source
# MAGIC %md
# MAGIC # Create Interactive Fast and beautiful dashboards using ipywidgets. 
# MAGIC 
# MAGIC ## We will look into how we can interactively visualize data frames using ipywidgets. 
# MAGIC I have divided this kernel into three parts first two parts are simple introductions to concepts and last part is application of those concepts.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizations are best way to communicate results with stake holders but sometimes looking at tabular data makes more sense. But there is always limitations in python visualization eco-system, not any more stick with kernel and you will be able to interactively visualize pandas Data Frame with amazing colors and interactions. 
# MAGIC ## Contents:
# MAGIC 1.  [Ipywidgets interact]().
# MAGIC     1. [Implicit Interact]()
# MAGIC         1. [Integer widget]()
# MAGIC         2. [Boolean widget]()
# MAGIC         3. [String widget]()
# MAGIC         4. [Float widget]()    
# MAGIC     2. [Explicit Interact]()
# MAGIC         1. [IntSlider]()
# MAGIC         2. [FloatRangeSlider]()
# MAGIC         3. [ToggleButton]()
# MAGIC         4. [RadioButton]()
# MAGIC         5. [DropDown]()
# MAGIC         6. [Visualize All Widgets]()
# MAGIC         7. [Using Layout Templates]()
# MAGIC 2. [Data Frame Style]()
# MAGIC     1. [Using background Gradient]()
# MAGIC     2. [Highlight Minimum and Maximum Value]() 
# MAGIC     3. [Using Apply Map]()
# MAGIC     4. [Using Apply]()
# MAGIC     5. [Using DataFrame bar]()
# MAGIC     6. [Using Text Gradient]()
# MAGIC 3. [Project Application]()
# MAGIC     1. [Handling Null Values]()
# MAGIC     2. [Outlier Removal]()
# MAGIC     3. [Visualize Relation Between 2 Quantitative Variables]()
# MAGIC     4. [Visualize Relation Between 2 Quantitative Variables with Hue]()
# MAGIC     5. [Univariate visualization of Quantitative Variable]()
# MAGIC     6. [Univariate visualization of Quantitative Variable with hue]()
# MAGIC     7. [Pivoting]()
# MAGIC     8. [Crosstab]()
# MAGIC     9. [Dashboard]()
# MAGIC   
# MAGIC   
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1'></a>
# MAGIC ## I. ipywidgets interact
# MAGIC <a id='section_1.1'></a>
# MAGIC ### a. implicit interact
# MAGIC 1. At most basic level interact auto-generates UI controls for function arguments, and then calls the function with those arguments when you manipulate the controls interactively. 
# MAGIC 2. Or speaking simply interact takes the function and arguments of the function to create a interactive plot, this plot can be manipulated by changing the value of function arguments. So let's take look at example. 

# COMMAND ----------

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# COMMAND ----------

# MAGIC %md
# MAGIC #### interact intro implicit widgets. 
# MAGIC When we pass particular data structures to interact those displays different widgets based on those inputs. 

# COMMAND ----------

def simple_function(x):
    return x 

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.1.1'></a>
# MAGIC #### passing a integer we get a integer slider 

# COMMAND ----------


_ = interact(simple_function,x = (10,20))

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.1.2'></a>
# MAGIC #### passing a boolean we get boolean radio buttons 

# COMMAND ----------


_ = interact(simple_function,x = True)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.1.3'></a>
# MAGIC #### passing a string we get text box

# COMMAND ----------

_ = interact(simple_function, x = 'hello')

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.1.4'></a>
# MAGIC #### passing a float we get float slider 

# COMMAND ----------

_ = interact(simple_function, x = (0.0,10.0))

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.2'></a>
# MAGIC ## Interact explicit widgets

# COMMAND ----------

# consider this sample form a normal distribution
values = np.random.standard_normal(size = 1000)
cat_values = np.random.choice(a = ['one','two','three'],size = 1000)
cat_values2 = np.random.choice(a = ['alpha','beta','gamma'],size = 1000)
sample_df = pd.DataFrame({'values': values,
                          'categories': cat_values,
                          'other categories': cat_values2
                         })
sample_df.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC consider this histogram, lets convert this into a function which takes arguments of histogram and plots a histogram

# COMMAND ----------

# consider this histogram, lets convert this into a function which takes arguments 
# of histogram and plots a histogram
plt.figure(dpi = 120)
sns.histplot(data = sample_df, x = 'values',palette='BuPu')

# COMMAND ----------

def plot_histogram(bins = 10, hue = 'categories', kde = False, palette = 'Blues', x_range_1 = (-3,3)): 
    """plots histogram
    params:
    =======
    bins: int
        histogram bins
    hue: str
        categorical columns to color 
    kde: bool 
        wether to show kde plot 
    palette: str
        palette of histogram
    x_range_1: tuple(int,int)
        x range of the plot 
    returns:
        histogram
    """
    plt.figure(dpi = 120)
    sns.histplot(data = sample_df, 
                        x = 'values',
                        palette=palette, 
                        bins = bins, 
                        hue = hue, 
                        kde = kde,
                 
                        
                       )
    plt.xlim(x_range_1)
    

# COMMAND ----------

plot_histogram()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.2.1'></a>
# MAGIC #### IntSlider
# MAGIC 1. The slider is displayed with a specified, initial value. Lower and upper bounds are defined by min and max, and the value can be incremented according to the step parameter.
# MAGIC 2. The first argument of plot_histogram is bins which can be changed with the int slider. And other values will be set as default values as described above.

# COMMAND ----------

_ = interact(plot_histogram,
         bins = widgets.IntSlider(
             value = 10,
             min = 5,
             max = 200,
             step = 10
         )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.2.2'></a>
# MAGIC #### FloatRangeSlider
# MAGIC 1. Say we want to form minimum value to maximum value, in this case setting the x limit 
# MAGIC    we can leverage the power of float slider. 

# COMMAND ----------

_ = interact(plot_histogram,
        x_range_1 = widgets.FloatRangeSlider(
            value = [-3,3], 
            min = -5,
            max = 5,
            step = 0.2,
            readout_format='.1f',
        )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id='section_1.2.3'></a>
# MAGIC #### ToggleButton 
# MAGIC 1. Suppose we want to visualize some categorical features and we want button like widget, toggle button widget can be leveraged. 
# MAGIC 2. In this example we want button for hue parameter. 

# COMMAND ----------

_ = interact(plot_histogram,
        hue = widgets.ToggleButtons(
            options = ['categories','other categories'],
            tooltip = ['categories','other categories'],
            disabled=False,
            button_style = 'success') )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_1.2.4'></a>
# MAGIC #### Radio Buttons 
# MAGIC 1. Lets create radio buttons to visualize kde plot of the hostogra,  

# COMMAND ----------

_ = interact(plot_histogram,
             kde = widgets.RadioButtons(
                 options = [True,False],
                 disabled = False)
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_1.2.5'></a>
# MAGIC #### Dropdown
# MAGIC 1. Dropdown widget can be used to get a dropdown list. In our case we will vary paletts of out plot to visualize various color schemes.

# COMMAND ----------

_ = interact(plot_histogram,
             palette = widgets.Dropdown(
                 options = ['pastel','husl','Set2','flare','crest','magma','icefire']
             )
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_1.2.6'></a>
# MAGIC ### Visualizing all widgets

# COMMAND ----------

_ = interact(plot_histogram,
             palette = widgets.Dropdown(
                 options = ['pastel','husl','Set2','flare','crest','magma','icefire']
                 ),
             kde = widgets.RadioButtons(
                     options = [True,False],
                     disabled = False),
            hue = widgets.ToggleButtons(
                options = ['categories','other categories'],
                tooltip = ['categories','other categories'],
                disabled=False,
                button_style = 'success'),
             bins = widgets.IntSlider(
                 value = 10, # intilal value 
                 min = 5,
                 max = 200,
                 step = 10
         ),
        x_range_1 = widgets.FloatRangeSlider(
            value = [-3,3], 
            min = -5,
            max = 5,
            step = 0.2,
            readout_format='.1f',
        ),
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_1.2.7'></a>
# MAGIC #### Say we want to arrange our widgets in horizontally or vertically we can do this using HBox and VBox 

# COMMAND ----------

b1 = widgets.Button(description='button 1')
b2 = widgets.Button(description='button 2')
b3 = widgets.Button(description='button 3')
b4 = widgets.Button(description='button 4')
b5 = widgets.Button(description='button 5')
b6 = widgets.Button(description='button 6')
# arrange (b1 b2) (b3 b4) (b5 b5) horizontally and 
# all this groups vertically 

hbox1 = widgets.HBox([b1,b2])
hbox2 = widgets.HBox([b3,b4])
hbox3 = widgets.HBox([b5,b6]) 
hbox1

# COMMAND ----------

widgets.VBox([hbox1,hbox2, hbox3])

# COMMAND ----------

# MAGIC %md
# MAGIC ## II. Now lets take a look at pandas table styling with stock data.

# COMMAND ----------

companies = pd.read_csv('https://raw.githubusercontent.com/kshirsagarsiddharth/ipywidgets_data/main/stocks.csv', index_col='Date', parse_dates=True)
# resampling daily data to annual data in this case BY means business start year 
companies = companies.resample('BY').mean()
companies.head()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.1'></a>
# MAGIC #### 1. Using Background Gradient function to visualize heatmap of the dataframe
# MAGIC ```
# MAGIC background_gradient(
# MAGIC     cmap='PuBu',
# MAGIC     low: 'float' = 0,
# MAGIC     high: 'float' = 0,
# MAGIC     axis: 'Axis | None' = 0,
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     text_color_threshold: 'float' = 0.408,
# MAGIC     vmin: 'float | None' = None,
# MAGIC     vmax: 'float | None' = None,
# MAGIC     gmap: 'Sequence | None' = None,
# MAGIC ) -> 'Styler'
# MAGIC Docstring:
# MAGIC Color the background in a gradient style.
# MAGIC 
# MAGIC The background color is determined according
# MAGIC to the data in each column, row or frame, or by a given
# MAGIC gradient map. Requires matplotlib.
# MAGIC ```

# COMMAND ----------

companies.style.background_gradient()

# COMMAND ----------

# MAGIC %md
# MAGIC using color palettes from seaborne
# MAGIC https://seaborn.pydata.org/tutorial/color_palettes.html

# COMMAND ----------

color_palette = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
companies.style.background_gradient(color_palette)

# COMMAND ----------

color_palette = sns.color_palette("vlag", as_cmap=True)
companies.style.background_gradient(color_palette)

# COMMAND ----------

color_palette = sns.color_palette("coolwarm", as_cmap=True)
companies.style.background_gradient(color_palette)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.2'></a>
# MAGIC #### 2. Highlighting minimum and maximum values
# MAGIC ```
# MAGIC highlight_min(
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     color: 'str' = 'yellow',
# MAGIC     axis: 'Axis | None' = 0,
# MAGIC     props: 'str | None' = None,
# MAGIC ) -> 'Styler'
# MAGIC Docstring:
# MAGIC Highlight the minimum with a style.
# MAGIC ```

# COMMAND ----------

# highliting minimum
# this understandable given 2002 tech bubble
companies.style.highlight_min(color = '#ff8a8a')

# COMMAND ----------

# highliting maximum 
companies.style.highlight_max(color = '#4fc277')

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.3'></a>
# MAGIC #### 3. Using Apply Map 
# MAGIC 1. This function will take first argument as function and second argument as label,array-like
# MAGIC ```
# MAGIC applymap(
# MAGIC     func: 'Callable',
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     **kwargs,
# MAGIC ) -> 'Styler'
# MAGIC Docstring:
# MAGIC Apply a CSS-styling function elementwise.
# MAGIC 
# MAGIC Updates the HTML representation with the result.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #### lets consider the below data frame which calculates returns of a stock
# MAGIC 1. Returns(https://www.investopedia.com/terms/r/return.asp)
# MAGIC     1. A return is the change in price of an asset, investment, or project over time, which may be represented in terms of price change or percentage change.
# MAGIC     2. A positive return represents a profit while a negative return marks a loss
# MAGIC 
# MAGIC ![image.png](attachment:19220509-74af-409c-a847-813681c1817b.png)

# COMMAND ----------

companies_returns = companies.pct_change().dropna()
companies_returns.head()

# COMMAND ----------

## Highliting losses in the dataframe 
def color_negative_values(value): 
    """
    This function takes in values of dataframe 
    if particular value is negative it is colored as redwhich implies loss
    if value is greater than one it implies higher profit
    """
    if value < 0:
        color = '#ff8a8a'
    elif value > 1:
        color = '#4fc277'
    else:
        color = 'black'
    return f"color: {color}"
companies_returns.style.applymap(color_negative_values)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Suppose we want to increase the size of text and add some background color

# COMMAND ----------

companies_returns.style.applymap(lambda x: 'font-size:23.2px; background-color: #d9e0ff')

# COMMAND ----------

# MAGIC %md
# MAGIC #### using profit loss and font and background function together

# COMMAND ----------

companies_returns.style.applymap(color_negative_values).applymap(lambda x: 'font-size:23.2px; background-color: #d9e0ff')

# COMMAND ----------

# using the subset argument
companies.style.applymap(lambda x: 'font-size:23.2px; background-color: #ccd8ff', subset=['IBM','WMT'])

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.4'></a>
# MAGIC ### 4. Apply: Suppose we want to find maximum and minimum on a row i.e which asset performed best and worst on a given day, we can do this using apply function this method takes the function and axis on which we want to apply the function 
# MAGIC ```
# MAGIC apply(
# MAGIC     func: 'Callable[..., Styler]',
# MAGIC     axis: 'Axis | None' = 0,
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     **kwargs,
# MAGIC ) -> 'Styler'
# MAGIC Docstring:
# MAGIC Apply a CSS-styling function column-wise, row-wise, or table-wise.
# MAGIC 
# MAGIC Updates the HTML representation with the result.
# MAGIC ```

# COMMAND ----------

def find_max(values):
    return np.where(values == np.max(values),'color: red; font-size:20.2px','font-size:20.2px')

def find_min(values):
    return np.where(values == np.min(values),'color: green; font-size:20.2px','font-size:20.2px')

# the function of apply should take a series and should return an object of same length
companies_returns.style.apply(find_max, axis = 1).apply(find_min, axis = 1)

# COMMAND ----------

# give border if the value in columns lie betwen 0.5 and 1 
companies_returns.style.apply(lambda x : ['color:#66de70; border-style: inset;font-size:20.2px' if (val > 0.5 and val < 1) else None for val in x], axis = 1)

# COMMAND ----------

# give border color if the value is greater than 1 
companies_returns.style.apply(lambda x : ['color:#87a9ff; border-style: ridge; border-width:7px;font-size:20.2px' if (val > 1) else 'opacity:0.2;' for val in x], axis = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.5'></a>
# MAGIC #### 5. bar: Lets say we are interested in only two assets and we want to compare both asset performance for each year side by side. We can leverage bar method of pandas style.  
# MAGIC ```
# MAGIC bar(
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     axis: 'Axis | None' = 0,
# MAGIC     color='#d65f5f',
# MAGIC     width: 'float' = 100,
# MAGIC     align: 'str' = 'left',
# MAGIC     vmin: 'float | None' = None,
# MAGIC     vmax: 'float | None' = None,
# MAGIC ) -> 'Styler'
# MAGIC ```

# COMMAND ----------

companies_returns.style.bar(subset = ['AMZN','IBM'], color = ['#d65f5f', '#5fba7d'], align = 'mid')

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_2.6'></a>
# MAGIC #### 6. Using text gradient to highlight text 
# MAGIC ```
# MAGIC text_gradient(
# MAGIC     cmap='PuBu',
# MAGIC     low: 'float' = 0,
# MAGIC     high: 'float' = 0,
# MAGIC     axis: 'Axis | None' = 0,
# MAGIC     subset: 'Subset | None' = None,
# MAGIC     vmin: 'float | None' = None,
# MAGIC     vmax: 'float | None' = None,
# MAGIC     gmap: 'Sequence | None' = None,
# MAGIC ) -> 'Styler'
# MAGIC Docstring:
# MAGIC Color the text in a gradient style.
# MAGIC 
# MAGIC The text color is determined according
# MAGIC to the data in each column, row or frame, or by a given
# MAGIC gradient map. Requires matplotlib.
# MAGIC ```

# COMMAND ----------

companies_returns.style.text_gradient(subset = ['AAPL','AMZN'], cmap = 'icefire').applymap(lambda x : 'font-size:22.2px; font-weight:bold')

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3'></a>
# MAGIC ## III. Applying ipywidgets and pandas styler to perform loan default analysis analysis.

# COMMAND ----------

df = pd.read_csv('https://raw.githubusercontent.com/kshirsagarsiddharth/ipywidgets_data/main/loan_data_raw.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.1'></a>
# MAGIC #### 1. Handling Null Values

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC less than 10 percent of data is null hence we can replace this with median. This is not recommended but for sake of simplicity we can do this. 

# COMMAND ----------

df['loan_int_rate'].isnull().sum() * 100 / df.shape[0]

# COMMAND ----------

df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# COMMAND ----------

# MAGIC %md
# MAGIC lets fill person employment length data frame with median value 

# COMMAND ----------

df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

# COMMAND ----------

# MAGIC %md
# MAGIC #### now we are rid of null values. 

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.2'></a>
# MAGIC ### Outlier detection and removal 
# MAGIC Generally outliers lie with quantitative data hence we want to perform interactive visualization of all quantitative variables. 

# COMMAND ----------

df.select_dtypes(include = 'number').columns

# COMMAND ----------

numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate','loan_percent_income',
       'cb_person_cred_hist_length']
numeric_cols

# COMMAND ----------

def scatter_plot_int(x = 'person_age',y = 'person_income'):
    plt.figure(dpi = 120)
    sns.set_style('whitegrid')
    return sns.scatterplot(data = df, x = x,y = y, alpha = 0.6)

# COMMAND ----------

scatter_plot_int()

# COMMAND ----------

_ = interact(scatter_plot_int,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             y = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 )
            )

# COMMAND ----------

# MAGIC %md
# MAGIC #### After using above interactive plot I found these outliers 
# MAGIC 1. person age > 80 outliers
# MAGIC 2. person income > 5 outliers
# MAGIC 3. person employment length > 60 outliers 

# COMMAND ----------

df = df[~(df['person_age'] > 80)]
#df = df[~(df['person_income'] > 5)]
#df = df[~(df['person_emp_length'] > 70)]



# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.3'></a>
# MAGIC #### Visualize Relation Between 2 Quantitative Variables

# COMMAND ----------

def scatter_plot_int(x = 'person_age',y = 'person_income'):
    plt.figure(dpi = 120)
    sns.set_style('whitegrid')
    return sns.scatterplot(data = df, x = x,y = y, alpha = 0.6, ).set_title('Visualize Relation Between 2 Quantative Variables')
A = interact(scatter_plot_int,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             y = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 )
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.4'></a>
# MAGIC #### 'Visualize Relation Between 2 Quantitative Variables with Hue'

# COMMAND ----------

def scatter_plot_int_with_hue(x = 'person_age',y = 'person_income', hue = 'loan_grade'):
    plt.figure(dpi = 120)
    sns.set_style('whitegrid')
    return sns.scatterplot(data = df, x = x,y = y, alpha = 0.6, hue = hue, cmap = 'Set2').set_title('Visualize Relation Between 2 Quantative Variables with Hue')

B = interact(scatter_plot_int_with_hue,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             y = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
              hue = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ) 
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.5'></a>
# MAGIC #### Univariate visualization of Quantitative Variable

# COMMAND ----------

def univariate_hist(x = 'person_age'):
    plt.figure(dpi = 130)
    sns.set_style('whitegrid')
    return sns.kdeplot(x = x, data = df, fill=True, palette = 'crest').set_title('Univariate visualization of Quantative Variable')


C = interact(univariate_hist,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 )
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.6'></a>
# MAGIC #### Univariate visualization of Quantitative Variable with hue 

# COMMAND ----------

def univariate_hist_with_hue(x = 'person_age', hue = 'person_income'):
    plt.figure(dpi = 130)
    sns.set_style('whitegrid')
    return sns.kdeplot(x = x, data = df, fill=True, palette = 'crest', hue = hue).set_title('Univariate visualization of Quantative Variable with hue')


D = interact(univariate_hist_with_hue,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             hue = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ) 
             
            )

# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.7'></a>
# MAGIC #### Pivoting 
# MAGIC Lets say you want to perform some pivoting with categorical data being in index and columns and, values being numerical data, this can be done using pivot tables.
# MAGIC in simple words we want to perform some pivoting.

# COMMAND ----------

def visualize_pivot_tables(index = 'person_home_ownership', column = 'loan_grade', values = 'loan_amnt', axis = 0):
    color_palette = sns.color_palette("vlag_r", as_cmap=True)
    t = pd.pivot_table(data = df, index = index, columns = column, values = values)
    return t.style.background_gradient(color_palette,axis = axis).applymap(lambda x : 'font-size:17.2px; font-weight:bold; opacity:0.9')
    

# COMMAND ----------

E = interact(visualize_pivot_tables,
             
             index = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ),
             column = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status'],
                 index = 1
                 ),
             values = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length'],
                 index = 3
                 ),
             axis = widgets.RadioButtons(options = [('Row',0),('Columns',1)])
                 
             
            )


# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.8'></a>
# MAGIC #### Analyze data between two categorical variables.

# COMMAND ----------

def interactive_crosstab(index = 'loan_grade', column = 'loan_intent'):
    crosstab = pd.crosstab(df[index], df[column]).style.text_gradient(cmap = 'icefire').applymap(lambda x : 'font-size:22.2px; font-weight:bold')
    return (crosstab.set_table_styles([
                    {
                        "selector":"thead",
                        "props": [("background-color", "#d0d0df"), 
                                  ("color", "black"),
                                  ("font-size", "20px"), ("font-style", "bold")]
                    },
                    {
                        "selector":"th.row_heading",
                        "props": [("background-color", "#7ebecc"), 
                                  ("color", "black"), 
                                  ("font-size", "2rem"), 
                                  ("font-style", "bold")
                                 ]
                    },
                        ]))
        
        

# COMMAND ----------

F = interact(interactive_crosstab,
             
             index = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ),
             column = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status'],
                 index = 1
                 )
            )


# COMMAND ----------

# MAGIC %md
# MAGIC <a id = 'section_3.9'></a>
# MAGIC ## Create Dashboard
# MAGIC from data acquired from above plots we create a dashboard using interactive class

# COMMAND ----------

A = interactive(scatter_plot_int,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             y = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 )
            )

B = interactive(scatter_plot_int_with_hue,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             y = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
              hue = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ) 
            )

C = interactive(univariate_hist,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 )
            )


D = interactive(univariate_hist_with_hue,
             x = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
                 ),
             hue = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ) 
             
            )


# COMMAND ----------

hbox1 = widgets.HBox([widgets.Label('One'), A,B])
hbox2 = widgets.HBox([widgets.Label('Two'),C,D])
widgets.VBox([hbox1, hbox2])

# COMMAND ----------

F = interact(interactive_crosstab,
             
             index = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ),
             column = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status'],
                 index = 1
                 )
            )

# COMMAND ----------


E = interact(visualize_pivot_tables,
             
             index = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status']
                 ),
             column = widgets.Dropdown(
                 options = ['person_home_ownership', 'loan_intent', 'loan_grade','cb_person_default_on_file','loan_status'],
                 index = 1
                 ),
             values = widgets.Dropdown(
                 options = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length'],
                 index = 3
                 ),
             axis = widgets.RadioButtons(options = [('Row',0),('Columns',1)])            
            )
