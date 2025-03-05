using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SmartphoneData
{
    public string DeviceName { get; set; }
    // Charging time will be in minutes
    public float ChargingTime { get; set; }
    public string OperatingSystem { get; set; }
}

public class SmartphonePrediction
{
    // This is where the name of the device that gets predicted gets storedd
    [ColumnName("PredictedLabel")]
    public string PredictedDeviceName;
}

// This is a program class
class Program
{
    static void Main(string[] args)
    {
        // Introduces the user to the program
        Console.WriteLine("Welcome to the Smartphone Recommendation System, hopefully we can recommend a phone you will like!\n");

        // This path leads to where the smartphone database is on the PC
        string filePath = @"C:\Users\Patrick\OneDrive - Anglia Ruskin University\Final Project\Smartphone Specifications.csv";
        List<SmartphoneData> smartphoneList = LoadSmartphoneDataFromExcel(filePath);

        // Starts the machine learning environment with ML.NET then loads 
        // the data from the smartphone list, which allows for it to be handled
        // correctly
        var mlContext = new MLContext();
        IDataView dataView = mlContext.Data.LoadFromEnumerable(smartphoneList);

        // Gives the device name of each smartphone a key with a number 
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(SmartphoneData.DeviceName))
            // Allows for the machine learning algorithm to process the operating
            // system for each phone it is not a number
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("OperatingSystemEncoded", "OperatingSystem"))
            // Combines Features, ChargingTime and OperatingSystemEncoded into a single vector
            .Append(mlContext.Transforms.Concatenate("Features", "ChargingTime", "OperatingSystemEncoded"))
            .Append(mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"))
            // Turns the numbered keys for the predicted label back into text
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // This is used to train the model so that it can then predict the best smartphone
        // for the user using the prediction engine 
        var model = pipeline.Fit(dataView);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<SmartphoneData, SmartphonePrediction>(model);

        // Allows for the program to keep track of which devices have and have
        // not been recommended, which prevents the user being recommended the same
        // device more than once
        var history = new Dictionary<string, HashSet<string>>(); 

        // This is a while loop that runs while true
        // another comment
        while (true)
        {
            // This will validate the user's input for the operating system
            string os = ValidateOperatingSystemInput();

            // This will validate the user's input for the charging time input
            float maxChargingTime = ValidateChargingTimeInput();

            // Combines what the user prefers for the operating system and the max charging time
            // into a single string
            string inputKey = $"{os}_{maxChargingTime}";
            var mostSuitablePhone = new SmartphoneData
            {
                OperatingSystem = os,
                ChargingTime = maxChargingTime
            };

            // Recommends a phone to the user 
            var prediction = predictionEngine.Predict(mostSuitablePhone);

            // Checks to see if the same input has been made before and adjusts the recommendation
            if (history.ContainsKey(inputKey) && history[inputKey].Contains(prediction.PredictedDeviceName))
            {
                Console.WriteLine("You've already received this recommendation before with the same preferences. Let's find another option...\n");
                continue;
            }

            // Records the recommendation for this input
            if (!history.ContainsKey(inputKey))
                history[inputKey] = new HashSet<string>();

            // Ensures that the same smartphone isn't recommended in the case that the user enters the same preference
            // they entered when they said they didn't like the recommended device
            history[inputKey].Add(prediction.PredictedDeviceName);

            
            Console.WriteLine($"\nThe phone we recommend that will suit you the most is: {prediction.PredictedDeviceName}");
            Console.WriteLine("Do you like this recommendation? (yes/no)");

            if (Console.ReadLine().Trim().ToLower() == "yes")
            {
                Console.WriteLine("\nThank you for using the Smartphone Recommendation System. We're glad you liked our recommendation!");
                Console.WriteLine("Please press Enter to close the program...");
                Console.ReadLine();

                // Exits the while loop 
                break;
            }

            Console.WriteLine("\nLet's try adjusting your preferences to find a better match.");
        }
    }

    static string ValidateOperatingSystemInput()
    {
        // While loop will be used to handle errors and prevent the user from inputting the wrong data
        while (true)
        {
            Console.WriteLine("Please enter which operating system you would prefer (ANDROID or IOS):");

            // Gives the user some leniency with the upper and lowercase spelling of 'ANDROID' and 'IOS'
            var input = Console.ReadLine().ToUpper();
            Console.WriteLine("");

            // Makes sure that at the very minimum the user inputs either 'ANDROID' OR 'IOS'
            // otherwise the while loop will continue to repeat itself until the user enters
            // a valid input
            if (input == "ANDROID" || input == "IOS")
                return input;

            Console.WriteLine("This is an invalid input. Please enter either 'ANDROID' or 'IOS'.");
        }
        
    }

    static float ValidateChargingTimeInput()
    {
        // While loop will be used to handle errors and prevent the user from inputting the wrong data
        while (true)
        {
            Console.WriteLine("What's the highest charging speed time you can tolerate? (in minutes)");
            
            // Checks to make sure that what the user inputted was a valid number
            // and not a letter. If it can confirm that the user inputted a valid
            // number it returns the float.TryParse as true so that the charging time
            // returns and the code can continue iterating, otherwise the while loop
            // will repeat itself until the input is a valid number
            if (float.TryParse(Console.ReadLine(), out float chargingTime))
                return chargingTime;

            Console.WriteLine("\nInvalid input. Please enter a valid number.");
        }
    }

    static List<SmartphoneData> LoadSmartphoneDataFromExcel(string filePath)
    {
        var smartphones = new List<SmartphoneData>();
        var lines = File.ReadAllLines(filePath);

        // Since the first line contains the header, it will be skipped
        for (int i = 1; i < lines.Length; i++)
        {
            var values = lines[i].Split(',');
            if (values.Length >= 3)
            {
                smartphones.Add(new SmartphoneData
                {
                    DeviceName = values[0].Trim(),
                    ChargingTime = ConvertChargingTimeToMinutes(values[1].Trim()),
                    OperatingSystem = values[2].Trim()
                });
            }
        }

        return smartphones;
    }

    static float ConvertChargingTimeToMinutes(string chargingTime)
    {
        // Since the excel sheet stored the charging time in an hrs and mins format, this
        // will be converting that into just minutes, which makes it easier for the machine
        // learning algorithm to work with
        int totalMinutes = 0;
        var parts = chargingTime.Split(new[] { 'h', ' ' }, StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length > 0)
            totalMinutes += int.Parse(parts[0]) * 60;

        if (parts.Length > 1)
            totalMinutes += int.Parse(parts[1].Replace("min", ""));

        return totalMinutes;
    }
}
