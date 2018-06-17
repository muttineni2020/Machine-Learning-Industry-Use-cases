# Getting-and-Cleaning-Data---Cleansing
Collect data , merge and clean data set

One of the most exciting areas in all of data science right now is wearable computing - see for example this article . 
Companies like Fitbit, Nike, and Jawbone Up are racing to develop the most advanced algorithms to attract new users. 
The data linked to from the course website represent data collected from the accelerometers from the Samsung Galaxy S smartphone. 
A full description is available at the site where the data was obtained

This repository contains the following files:

- `README.md`, this file, which provides an overview of the data set and how it was created.
- `tidyData.txt`, which contains the output of the data set.
- `CodeBook.md`, the code book, which describes the contents of the data set (data, variables and transformations).
- `run_analysis.R`, the R script that was used to create the data set

Training and test data were first merged together to create one data set, then the measurements on the mean and standard deviation were extracted for each measurement (79 variables extracted from the original 561), and then the measurements were averaged for each subject and activity, resulting in the final data set.

## Creating the data set <a name="creating-data-set"></a>

The R script `run_analysis.R` can be used to create the data set. It retrieves the source data set and transforms it to produce the final data set by implementing the following steps (see the Code book for details, as well as the comments in the script itself):

- Download and unzip source data if it doesn't exist.
- Read data.
- Merge the training and the test sets to create one data set.
- Extract only the measurements on the mean and standard deviation for each measurement.
- Use descriptive activity names to name the activities in the data set.
- Appropriately label the data set with descriptive variable names.
- Create a second, independent tidy set with the average of each variable for each activity and each subject.
- Write the data set to the `tidyData.txt` file.

The `tidyData.txt` in this repository was created by running the `run_analysis.R` script using R version 3.4.3 (2017-11-30) on Windows.

This script requires the `dplyr` package.
