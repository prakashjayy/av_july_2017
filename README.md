# av_july_2017
Work on AV July 2017 Fractal Hiring Hackathon.

This is a two day Hackathon conducted by Analytics Vidya and Fractal Analytics to hire Data-Scientists.

- Public Leaderboard - 24th
- Private Leaderboard (Final Ranking) - 9th

- There were in total 2489 participants out of which 211 participants made it to the Leaderboard.


## To Reproduce the results, run the following files in the specified order
- inital_data_clean.py
- secondary_data_clean.py
- model_4.py

or

run ```make``` in your command line. make sure you have **make** installed on your system.

other files:
- utils.py : Contains moving average function I used in this model.

## Dataset Overview:
Welcome to Antallagma - a digital exchange for trading goods. Antallagma started its operations 5 years back and has supported more than a million transactions till date. The Antallagma platform enables working of a traditional exchange on an online portal.  More about dataset here https://datahack.analyticsvidhya.com/contest/fractal-analytics-hiring-hackathon/

Description:
  - There were in-total of 1529 unique stocks in the train data and 1447 stocks in test data.
  - The train data is from jan-2014 to june-2016 and the test data is from july-2016 to dec-2016.
  - Category_1, Category_2, Category_3 are Binary masked feature, Ordered Masked feature, Unordered Masked feature respectively.
  - Price (Median Price at Sale on that day), Number_Of_Sales (Total Item Sold on that day) are two target variables


## Evaluation Criteria:
Overall Error = Lambda1 x RMSE error of volumes + Lambda2 x RMSE error of prices Where Lambda1 and Lambda2 are normalising parameters

## Initial cleaning:
- Removed all the stocks which are not present in the test data and added missing dates for each stock (if der are any) from the start of it listing. (might occur if there are no buyers or sellers for the stock on a given day) (inital_data_clean.py) - Total data_points: 808786
- Since stocks which don't have buyers or sellers on a given day were not present in the data. I took each stock ID, added time_frame from 2014-jan to 2016-june and imputed 0 value to both price and Number_Of_Sales if the stock is not traded on that day. Appended all the datasets (secondary_data_clean.py) - Total data_points: 1319664.
- I have created features as week, month, year and removed category_1, category_2, category_3 from the data as they are unique for each StockID. my final model used moving averages by using extensive grid search using solid validation framework.

## My Final Approach:
---------------------
- Used exponential moving average on daily data using the formula
```
F(t+1) = alpha * F(t)  + alpha * (1 - aplha) * F(t-1) + alpha * (1 - alpha)^2 * F(t-2) + .....N
```
- alpha and N are different for each stock and they were choosen using grid search and validation rmse.
- Final Validation RMSE:
  - Price: 5.6502
  - Number_Of_Sales: 2307.63
- It took 40 mins to run on my system which has 32GB RAM.



### Things that worked:
- Aggregating each stock **weekly** and using 27 week moving average (Public Leaderboard: 0.72)
- Agrregrating each stock **monthly** and then using moving averages by tuning params using validation framework
(Public Leaderboard: 0.65)
- Agrregrating each stock **weekly** and then using moving averages by tuning params using validation framework (Public Leaderboard: 0.54)
- Using **daily** moving averages by tuning params using validation framework (Public Leaderboard: 0.41) (**Final Solution**). Each of these models took (40min - 3 hours) depending on the level of grid search you have done.

### Things I tried and failed:
- Made a submission in the intial using median (Score Pubilc = 0.97)
- Building XGBoost, LightGBM Model for each and every stock indvidually using 3 day , 7 day, 21 day, 60 day, 120 day moving averages. This didn't give good results and the predicitions were mainly straight lines. I actually messed with the train_data and I guess this would have worked well. This Process took 1 hour with simple grid search I applied.
- Building Linear models didn't help on individual stocks.
- I tried to build one model for all stocks using category features, different time features and moving averages but it is not supporting my RAM and I didn't have any faith in working of this model. I need to rethink on this aspect.
- I tried LSTMs and RNNs, Neural Networks, there are two aspects here, modelling individual stocks and all the stocks at a time, I faced memory issues and simple models led to underfitting.

### Things I should have tried:
- Using **exponential moving averages**, Trend and other variables as features and training linear models on each stocks individually. (I have seen trend in few stocks while plotting)
- **Pair Trading** I have this concept in my mind during the entire competition but didn't give it a try as I was busy with usually work mentioned above. If we can check if there is any correlation between two stocks, this would be a great boost.
- **Auto-Correlations (ACF) and using AR(N)** as features in linear models. I have previously not worked on daily data and making R-forecast package work on this has created some problems to me. Will check this in my free time. But building models on this will actually help.
- The overall negitive correlation between price and Number of Sales is -0.43 using Spearman. It might be much more if i look at each stock individually. I could have used these while modelling using mv.
- Stacking of different models that worked and given positive lift in leaderboard.
- In the end I ran one more search for finding the best model out of 10+ models I have built for each and every stock in price and sales individually. I got this idea before 10 mins of competition ending. So ended up leaving the code in between. Possibly blending with different weights and validating it with validation data would have helped also. 
- Building some sort of UI to visualize the data with ease. R shiny will help here. I lost a bit of touch on it in recent times and didn't thought i can handle it in two days.

### Key takeways
- Don't underestimate simple models. Jumping directly to **complex Algorthims** like LightGBM and XGBoost is the costliest mistake i have done in recent time.
- When you are doing extensive grid search and running computations for long time, check upfront if everything is working as desirable. I ended up wasting last 4 hours in this competition because of a bug in my code. I didn't have time for re-running it.
- Visualize the data as much as you can. Spend 80% of your time in creating compelling plots to have a clear picture of what you model tells.
- Build a strong validation framework. In timeseries mainly try to plot the predictions and check how the model is fitting the original data. Trust it more than your Public leaderboard
- Don't give up. Initially checking each stock individually is a nightmare (1447 stocks) and there needs to be a lot of cleaning need to be done for automated.
- Though I ended up building models using Python writing my own functions for moving averages and other stuff, R might have been a better choice for this kind of dataset with its vast libraries like forecast, zoo, lubridate, data.table, ggplot2 etc.


In the end, thank you Fractal Analytics and Analytics Vidya for organising the competition and making us work on Weekends. It was fun.

PS: My code is very messy and verbose in its present form. I can reduce it by writing compelling functions but replicating the results would take hours and I don't want to mess with it now. Apologies for it.
