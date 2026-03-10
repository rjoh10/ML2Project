# DS 4420: Machine Learning and Data Mining 2  
Authors: Iba Baig and Rhea Johnson  

## Project Title  
**Forecasting Electricity Demand During Snowstorm Events**

## Project Overview  
Severe winter weather can place major stress on the electric grid through both rising electricity demand and weather-related outages. During snowstorms and extreme cold events, electricity consumption may increase significantly due to heating needs, while outages and infrastructure disruptions can reduce grid reliability.  
This project investigates whether short-term electricity demand can be forecasted during extreme winter weather events and how uncertainty in those forecasts can be quantified. Our goal is to better understand how snowstorm-related weather conditions affect electric consumption and grid stress.


## Proposed Methods  

- Autoregressive / Time Series Modeling: for short-term electricity demand forecasting  
- Bayesian Modeling :to quantify uncertainty in forecasts and provide probabilistic predictions  

## Data Sources  
We are exploring the following data sources for electricity demand and winter weather variables:

### Electricity Demand Data
- U.S. Energy Information Administration (EIA) Grid Monitor**  
  Real-time and historical electricity grid data  
  https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48  

### Climate and Weather Data
- NOAA Climate Data Online (CDO) 
  Temperature, wind, snowfall, and other weather variables  
  https://www.ncdc.noaa.gov/cdo-web/  


## Literature Review Summary  
Severe winter weather conditions can threaten the reliability of the power grid in several ways, including rising electricity demand during extreme cold and weather-driven outages that interrupt service. Recent research has increasingly used machine learning to study grid vulnerability and outage-related problems. This makes snowstorm-related electricity demand an important extension of existing work, especially in regions where winter weather can sharply increase grid stress.

Much of the existing literature focuses more on outage prediction than on demand forecasting during snowstorm conditions. For example, Cerrai et al. (2020) developed outage prediction models for snow and ice storms in the Northeastern United States using several machine learning methods, including Random Forest, Bayesian Additive Regression Trees, and a Generalized Linear Model. Their results showed that performance varied by storm severity, with machine learning models performing better for lower-impact events while the GLM remained competitive for extreme events. This work is especially useful because it suggests that winter-weather grid impacts are influenced by a broad combination of physical and environmental variables, rather than temperature alone.

Our project builds on this prior work by shifting the focus from outage prediction toward **electricity demand forecasting during snowstorm events**, while still recognizing the close relationship between demand spikes, outage risk, and overall grid stress.

## Sources Referenced  
- Cerrai, D. et al. (2020). *Using machine learning and big data approaches to predict power outages caused by extreme events: a case study of Hurricane Sandy.*  
  https://www.sciencedirect.com/science/article/abs/pii/S2352467719302668

- Argonne National Laboratory. *Outage prediction and grid vulnerability identification using machine learning on utility outage data.*  
  https://www.anl.gov/esia/outage-prediction-and-grid-vulnerability-identification-using-machine-learning-on-utility-outage

- [Springer Article on Winter Storm Outage Prediction]  
  https://link.springer.com/article/10.1007/s43762-025-00222-9

## Repository Contents  
This repository will contain:
- Literature review materials for Phase I
- Data collection
- Exploratory analysis 
- Proof-of-concept implementation for the model
- Final project Results

## Phase I Goal  
For Phase I, this repository will provide:
- A short literature review
- A proof-of-concept first model

