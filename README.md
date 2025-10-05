# Treasury Bills Predictive Model

## 1. Project Overview
This project develops a predictive model for ***Treasury Bill(T-Bills) rates***, using historical financial and economic indicators. The goal  of this project is to forecast short-term interest rate movements for the 91-day T-Bills inorder to support investment decisons and policy analysis.
## 2. Datascience Workflow
This project follows a CRISP-DM (Cross-Industry Standard Process for Data Mining) workflow. It provides a structured and iterative approach consisting of six phases. They include:
     - Business Understanding
     - Data Understanding
     - Data Preparation
     - Modelling
     - Evaluation
     - Deployment
### 2.1 Business Understanding
Treasury Bills (T-Bills) serve as crucial short-term government financing tools and are widely regarded as safe investment options for both individuals and institutions. Therefore, the ability to anticipate T-Bill rates offers several advantages to various stakeholders. They include;
   - ***Investors***: the prediction helps investors make rational investment decisions.
   - ***Policakers***: helps understand the government cost of borrowing and macroeconomic variables regarding T-Bill rates.
   - ***Financial institutions***: T-Bills rate predicts the value of government borrowing and helps manage treasury and liquidity.
   - ***Researchers and analysts***: predicts the relationship between macroeconomic variables and short-term interest rates.
   
The main objective is to improve the accuracy of forecasting T-Bills by the use of statistical and machine learning techniques leading to better decision making by the stakeholders.

### 2.2 Data Understanding
### 2.2.1 Sources
Our data was comprehensively collected from various publicly available sources, thus ensuring our model reprocubility. The sources include:
   1. ***Central Bank of Kenya(CBK)***
This was the primary source of the target variable and monetary policy data. Some of the datasets include;
   - ***Central Bank Rate(CBR)***
   - ***Treasury Bills Average Rates***: for 91-day treasury bill weighted average rate(Target variable)
   - ***Auction Metrics Issues of Treasury Bills***: for amount offered,amount allotted, amount redeemed
Data was scrapped from the ***CBK*** published statistical bulletin.
   2. ***Kenya National Bureau of Statistics(KNBS)***
Mostly this is the source of Kenyan macroeconomic and socioeconomic data. Datasets scrapped from the official KNBS website annual economic surveys include:
   - ***Inflation Rates***: for Consumer Price Index(CPI)
   - ***Annual GDP***: For Annual growth figures
   - ***Public Debt***: For domestic and external debt
   3. ***Federal Reserve Economic Data(FRED)***
FRED is a global body that provide essential data on international benchmarks and economic conditions. Keys dataset scrapped from the FRED API include:
   - ***Trade Weighted Average Indicative Rates***: for US yields e.g the Dollar index

 ### 2.2.2 Variables Used
   1. Treasury bills weighted average rates(Target variable)
   2. Central Bank Rate(CBR)
   3. Public debt(domestic and external debts)
   4. Exchange rates
   5. Inflation rate

### 2.3 Data Preparation
Building the predictive model required the integration and extensive cleaning of multiple public economic datasets. This step established uniformity in the data, ensured the right data types were used, and coordinated different time series variables.
Key steps included:
   - Handling missing values and removing duplicates.
   - Correcting date formats for consistency.
   - Merging multiple datasets based on ***Year*** and ***Month***.
   - Creating time-series sequences(lags and rolling averages) to capture temporal dependencies
   - Standardizing numerical variables using ***StandardScaler*** to prepare data for machine learning.
   - Splitting the data into training and testing sets(80/20) to evaluate predictive performance.

### 2.4 Modelling
Several models were tested with the intent to accurately come up with the best model that forecast treasury bill rates. Both Statistical and machine learning models were used. They include:
   1.***Statistical Models***: 
      - Linear Regression(Baseline Model) for establishing linear relationships between t-bill and macroecomic indicators.
      - ARIMA a time-series forecasting for capturing seasonality and important trends.
      - Garch for volatility and variance in t-bill rates. 
   2.***Machine Learning Models***:
      - Random Forest Regressor for capturing nonlinear relationships
      - LightGBM(LGBM) for optimizing predictive performance through boosting.
### 2.5 Evaluation
All the models were evaluated using forecasting and standard regression metrics:
   - ***MAE(Mean Absolute Error)***- measures average prediction error.
   - ***RMSE(Root Mean Squared Error)***-penalizes larger errors more strongly.
   - ***r2(Coeffecient of Determination)***-this expalains how well a model explains the variance in t-bill rates.
Evaluation also included visualizations of predicted vs actual t-bill rates to visually show how the models were performing.
The best performing model had an ***RMSE of 0.3140***
### 2.6 Deployment
The best performing model(***LightGBM***) was deployed using ***Streamlit*** to provide an interactive and user-friendly interface for Treasury Bills forecasting.
The Streamlit dashboard allows users to:
   - View historical vs predicted t-bill rates
   - Input different macroeconomic variables(e.g., inflation rate,CBR,external and domestic debt) to simulate different scenarios.
-***Deployment option***: For local deployment, use run ***Streamlit run app.py*** to launch it on a dashboard locally.

## 3. How to run the notebook
 1. Clone the repository:

git clone https://github.com/boniface2025/GROUP-6-CAPSTONE-PROJECT-PHASE-5.git

2. Install dependencies:

pip install -r requirements.txt

3. Launch Jupyter Notebook:

jupyter notebook

Then open notebooks/EDA.ipynb to explore and train models.

4. Run the Streamlit app:

streamlit run app.py

## 4. Expected Output
   - Cleaned and merged datasets with Treasury Bill rates and macroeconomic indicators.
   - Visualizations showing historical vs predicted T-Bill rates.
   - Forecasting metrics(MAE,RMSE,R2)
   - Interactive Streamlit dashboard for scenario-based forecasting.
## 5. Tech Stack
   -***Python***:pandas,numpy,scikit-learn,statsmodels
   -***Machine Learning***:XGBoost,Random Forest Regressor,LightGBM
   -***Visualization***:matplotlib,seaborn,plotly
   -***Deployment***:Streamlit
   
## 6. Future Work
   - Develop a hybrid model that combines statistical and machine learning models to capture temporal dependancies and nonlinear patterns.
   - To enhance more accuracy, we'll incorporate global market indicators and macroeconomic sentiment data.
   - Integrate our model to a financial analytic dashboard.
## 7. Presentation Slides
Access the project presentation slides here:

## 7. Contributors
  1. Hilda Jerotich
  2. Erick Kibugi
  3. Barnice Wandeto
  4. Boniface Njeri
  5. Alice Muia
  6. David Muriithi

