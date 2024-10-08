# A-Novel-Multi-Task-Learning-Based-Approach-to-Multi-Energy-System-Load-Forecasting
This repository contains the implementation of a novel multi-task learning-based approach for load forecasting in multi-energy systems (MES). The proposed deep learning model, D-TCNet, integrates Multi-Layer Perceptrons (MLP) and Temporal Convolutional Networks (TCN) to forecast multiple energy loads (heating, cooling, electricity) simultaneously. By leveraging load correlations and advanced network architecture, the model improves the accuracy of load forecasting across all energy types and seasons. The dataset used for this research is from the University of Austin Tempe Campus, spanning the years 2016 to 2019.
## Data Description
The dataset used in this study is from the University of Austin Tempe Campus, focusing on heating, cooling, and electricity consumption data for a multi-energy system. The data was collected from 2016 to 2019 at an hourly interval.

### Key Features:

<div align="center">

| Parameter        | Value                              |
|------------------|------------------------------------|
| Data Source      | [Campus Metabolism](http://cm.asu.edu/) |
| Data Start Date  | January 2nd, 2016                  |
| Data End Date    | March 31st, 2020                   |
| Data Interval    | 1 hour                             |
| Total Data Points| 37,221                             |
| Features         | 8                                  |

</div>
The data was divided into seasons (summer, fall, winter, spring), and distance correlation analysis was used to identify the coupling between energy variables across seasons. Temporal features (past consumption values) and meteorological features were included in the input variables.

## Network Architecture
The proposed model, D-TCNet, is a deep learning network that combines Multi-Layer Perceptrons (MLP) and Temporal Convolutional Networks (TCN). The architecture leverages multi-task learning (MTL) to forecast cooling, heating, and electricity loads simultaneously by sharing information across these tasks.

### Key Components:
Multi-Layer Perceptron (MLP): Encodes spatial information and reduces data dimensionality.

Temporal Convolutional Network (TCN): Captures temporal dependencies in time series data using causal and dilated convolutions with residual blocks.

Multi-Task Learning (MTL): Allows simultaneous learning of different energy variables by sharing a part of the network across tasks.

The network also includes residual connections for efficient learning of long-range temporal dependencies, reducing the number of layers required for a large receptive field.

## Results
The proposed D-TCNet model showed improved performance across all seasons for multi-energy load forecasting. It was compared against traditional machine learning models (SVM, Random Forest) and state-of-the-art deep learning models (LSTM, Bi-LSTM).

### Performance Metrics:
Mean Absolute Percentage Error (MAPE): Used to evaluate forecasting accuracy.

Root Mean Squared Error (RMSE): Used to measure prediction error.

### Key Findings:
The D-TCNet model significantly outperformed single-task learning approaches and other state-of-the-art methods, particularly in accuracy (lower MAPE and RMSE).
The model achieved better results in forecasting energy demand across different seasons by leveraging multi-task learning and distance correlation analysis.
