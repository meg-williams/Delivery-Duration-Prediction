# Delivery Duration Prediction


This project was created from a StrataScratch problem given by DoorDash. One of the most import parts of DoorDash's customer service is providing an accurate estimate to the customer for when their order will arrive. In this project multiple regression models along with hyperparameter searches are used to determine which might give the best results. 
***
<br>

## Dataset


| Category                       | Value   |
|--------------------------------|---------|
| Size of Dataset                | `197,428` |
| Number of Features             | `16  `    |
| Number of Numerical Features   | `13`      |
| Number of Categorical Features | `1`       |
| Number of DateTime             | `2`       |

<br>

This dataset, `historical_data.csv`, contains data from deliveries over the course of the first two months in 2015. The file has the following features: 

- **market_id:** A city/region in which DoorDash operates. 

- **created_at:** Timestamp in UTC when the order was submitted by the consumer to DoorDash. 

- **actual_delivery_time:** Timestamp in UTC when the order was delivered to the consumer.

- **store_id:** an id representing the restaurant the order was submitted for. 

- **store_primary_category:** cuisine category of the restaurant. 

- **order_protocol:** a store can receive orders from DoorDash through many modes. This field represents an id denoting the protocol.

- **total_items:** total number of items in the order.

- **subtotal:** total value of the order submitted (in cents).

- **num_distinct_items:** number of distinct items included in the order. 

- **min_item_price:** price of the item with the least cost in the order (in cents). 

- **max_item_price:** price of the item with the highest cost in the order (in cents).

- **total_onshift_dashers:** Number of available dashers who are within 10 miles of the store at the time of order creation. 

- **total_busy_dashers:** Subset of above total_onshift_dashers who are currently working on an order.

- **total_outstanding_orders:** Number of orders within 10 miles of this order that are currently being processed.

- **estimated_order_place_duration:** Estimated time for the restaurant to receive the order from DoorDash (in seconds). 

- **estimated_store_to_consumer_driving_duration:** Estimated travel time between store and consumer (in seconds)

<br>

## Data Processing

In order to prepare the data to go through the models the following was done: 

- Nulls were imputed using mode and KNN imputation 
- Convert timestamp columns from object to datetime 
- actual_delivery_time was subtracted from created_at to create the target variable
- Boxplots, histograms, and KDE plots were created to check for outliers and skewness
- Day and hour were extracted from created_at to created day and hour columns
- Dasher_ratio was created using total_onshift_dashers and total_busy_dashers
- Net_dashers_orders was created using total_onshift_dashers and total_outstanding_orders 
- A correltion matrix and heatmap were used to remove colunms that were colinear 

## models 

Train test split was used to created a train set and a test set. Models were evaluated using r squared and RMSE: 

- Linear Regression
- Polynomial Regression 
    - run using pipeline with PolynomialFeatures, StandardScaler, Ridge
    - gird search was used to find the best hyperparameters
- Random Forest Regression 
    - Randomized Search was used to find the best hyperparameters
