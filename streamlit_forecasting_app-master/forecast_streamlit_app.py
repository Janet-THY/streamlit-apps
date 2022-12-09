import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

def load_csv():
    df_input = pd.DataFrame()  
    df_input=pd.read_csv(input,sep=None ,engine='python', encoding='utf-8',
                            parse_dates=True,
                            infer_datetime_format=True)
    return df_input

def prep_data(df):
    df_input = df.rename({date_col:"ds",metric_col:"y"},errors='raise',axis=1)
    st.markdown("The selected date column is now labeled as **ds** and the values columns as **y**")
    df_input = df_input[['ds','y']]
    df_input =  df_input.sort_values(by='ds',ascending=True)
    return df_input


st.title('e-4Cast Time Series Forecasting 📈')

"""
This interactive web app allows you to generate future forecast in a few minutes!
The forecasting library used is Facebook's open-source Prophet library [Prophet](https://facebook.github.io/prophet/). 
You'll be able to import your data from a correctly-labelled CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast. 

**In beta mode**
"""

"""
### Step 1: Import Data
"""
with st.expander("Data format"): 
    st.write("Import a time series csv file which contains columns labeled as `ds` (dates) and `y`(target you wish to forecast). The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.")
    st.write("For example:")
    st.image("./images/time_series_data_preprocessing_ecommerce.jpeg")

st.sidebar.image("./images/prophet.png")
st.sidebar.header("About")
st.sidebar.markdown("Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
st.sidebar.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
st.sidebar.write("")
st.sidebar.write("Created on 08/12/2022")
    
df = st.file_uploader('', type='csv')

st.info(
f"""
                👆 Upload a .csv file first. Sample to try: [daily_purchase_order.csv](https://git.generalassemb.ly/janet-thy/capstone/blob/main/datasets/osc_pur_daily_for_test.csv)
                """
        )

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce') 
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)
        
"""
### Step 2: Select Forecast Horizon

Please note that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
min_value = 1, max_value = 365)

if df is not None:
    m = Prophet()
    m.fit(data)

"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.write(fcst_filtered)
    
    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig1 = m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = m.plot_components(forecast)
    st.write(fig2)


"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
    
"""
### End of page

To learn more about forecasting model or my inspiration.
"""
with st.expander("Explanation"):
    st.image("./images/prophet.png")
    st.header("About")
    st.markdown("Official documentation of **[Facebook Prophet](https://facebook.github.io/prophet/)**")
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    st.write("")
    st.write("Inspired by [Zach Renwick](https://twitter.com/zachrenwick)")
    st.write("Created on 08/12/2022")