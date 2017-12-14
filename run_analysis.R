##############################################################################
#
# FILE
#   run_analysis.R
#
# OVERVIEW
#   Using data collected from the accelerometers from the Samsung Galaxy S 
#   smartphone, to make a clean data set, outputting the
#   resulting tidy data to a file named "tidy_data.txt".
#   See README.md for details.
#

library(dplyr)

## Getting data and Reading data from files in zip folder


Url <- "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"
zipName <- "UCI HAR Dataset.zip"

if (!file.exists(zipName)) {
  download.file(Url, zipName, mode = "wb") ## download zip file containing data
}

# unzipping file if it is not exists in data directory
fileName <- "UCI HAR Dataset"
if (!file.exists(fileName)) {
  unzip(zipName)
}

# read training data
trainingSub <- read.table(file.path(fileName, "train", "subject_train.txt"))
trainingVal <- read.table(file.path(fileName, "train", "X_train.txt"))
trainingAct <- read.table(file.path(fileName, "train", "y_train.txt"))


# read test data
testSub <- read.table(file.path(fileName, "test", "subject_test.txt"))
testVal <- read.table(file.path(fileName, "test", "X_test.txt"))
testAct <- read.table(file.path(fileName, "test", "y_test.txt"))

# read features, don't convert text labels to factors
features <- read.table(file.path(fileName, "features.txt"), as.is = TRUE)

# read activity labels
activities <- read.table(file.path(fileName, "activity_labels.txt"))
colnames(activities) <- c("activityId", "activityLabel")


#########################################################################
# 1 - Merges the training and the test sets to create one data set.
#########################################################################

# concatenate individual data tables to make single data table
resultTT <- rbind(
  cbind(trainingSub, trainingVal, trainingAct),
  cbind(testSub, testVal, testAct)
)
###print(resultTT)

# Optional - remove individual data tables. It would be good save memory if we remove unused data tables.
rm(trainingSub, trainingVal, trainingAct, testSub, testVal, testAct)

# assign column names
colnames(resultTT) <- c("subject", features[, 2], "activity")


##############################################################################
# 2 - Extracts only the measurements on the mean 
#     and standard deviation for each measurement.
##############################################################################

# determine columns of data set to keep based on column name.
columnsToKeep <- grepl("subject|activity|mean|std", colnames(resultTT))

# and keep data in these columns only
resultTT <- resultTT[, columnsToKeep]


##############################################################################
# 3 - Uses descriptive activity names to name the activities in the data set
##############################################################################

# replace activity values with named factor levels
resultTT$activity <- factor(resultTT$activity, 
                                 levels = activities[, 1], labels = activities[, 2])


##############################################################################
# 4 - Appropriately labels the data set with descriptive variable names.
##############################################################################

# get column names
resultTTCols <- colnames(resultTT)

# remove special characters
resultTTCols <- gsub("[\\(\\)-]", "", resultTTCols)

# expand abbreviations and clean up names
resultTTCols <- gsub("^f", "frequencyDomain", resultTTCols)
resultTTCols <- gsub("^t", "timeDomain", resultTTCols)
resultTTCols <- gsub("Acc", "Accelerometer", resultTTCols)
resultTTCols <- gsub("Gyro", "Gyroscope", resultTTCols)
resultTTCols <- gsub("Mag", "Magnitude", resultTTCols)
resultTTCols <- gsub("Freq", "Frequency", resultTTCols)
resultTTCols <- gsub("mean", "Mean", resultTTCols)
resultTTCols <- gsub("std", "StandardDeviation", resultTTCols)

resultTTCols <- gsub("BodyBody", "Body", resultTTCols)

# use new labels as column names
colnames(resultTT) <- resultTTCols


##############################################################################
# 5 - From the data set in step 4, creates a second, independent 
#          tidy data set with the average of each variable 
#          for each activity and each subject.
##############################################################################

# group by subject and activity and summarise using mean
resultTTMeans <- resultTT %>% 
  group_by(subject, activity) %>%
  summarise_all(funs(mean))

# output to file "tidy_data.txt"
write.table(resultTTMeans, "tidyData.txt", row.names = FALSE, 
            quote = FALSE)