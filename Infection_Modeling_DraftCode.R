#Connection string to connect to SQL Server named instance
connStr <- paste("Driver=SQL Server; Server=", "ALTERYX_DEVMWS3",
                 ";Database=", "NOVID_DB", ";Trusted_Connection=true;", sep = "");

#Get the data from SQL Server Table
SQL_infectiondata <- RxSqlServerData(table = "dbo.infection_tracking",
                                  connectionString = connStr, returnDataFrame = TRUE);

#Import the data into a data frame
infectiondata <- rxImport(SQL_infectiondata);

#Let's see the structure of the data and the top rows
# Infection data, given the number of infections on a given day
head(infectiondata);
str(infectiondata);

#Changing the four factor columns to factor types
#This helps when building the model because we are explicitly saying that these values are categorical
infectiondata$Day <- factor(infectiondata$Day);
infectiondata$InfectedCount <- factor(infectiondata$InfectedCount);
infectiondata$ExposedCount <- factor(infectiondata$ExposedCount);
infectiondata$RecoveredCount <- factor(infectiondata$RecoveredCount);


#Visualize the dataset after the change
str(infectiondata);

