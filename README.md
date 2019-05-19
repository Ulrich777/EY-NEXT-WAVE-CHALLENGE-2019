# EY-NEXT-WAVE-CHALLENGE-2019
* The objective was to gain insight on citizens move along with the smart city project. 
With data on trips recorded by the mean of devices, we were challenged to predict if a t
rip ends into the city center given data on time,  the entry and exit locations 2D coordinates, 
and speed measurements.

* Overall we were ranked 23/373  and 4/30 in France where we made it all the way to national finals.

* To understand our approach
  * you can refer to the pdf file `Next_Wave_Presentation.pdf`
  * All the helper functions can be found in the file helper. `makedata.py` and `getfeatures.py` 
    implement both data preprocessing and features engineering. The models are coded in the file
    `predictors.py`.
  * To reproduce the result: (1) split first the data into train, eval and test with `split_data.py`,
    and (2) run the python script `run_classification.py`. Yet if you are interested in our regression model
    (i) just estimate the modes of the arrival coordinates with `likelihood_modes.py` and (ii) run the regression
    with `run_regression.py`. 
