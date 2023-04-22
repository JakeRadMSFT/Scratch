// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using ObjectDetective;

MLContext mlContext;

mlContext = new MLContext();

Console.WriteLine("Hello, World!");

mlContext.Log += MlContext_Log;

void MlContext_Log(object? sender, LoggingEventArgs e)
{
    if(!e.RawMessage.StartsWith("Channel"))
    {
        Console.WriteLine(e.RawMessage);
    }
}

ObjectDetectionTSV.Train(@"C:\dev\Scratch\ObjectDetective\ObjectDetective\ObjectDetectionTSV.mlnet", mlContext);

var modelInputTSV = new ObjectDetectionTSV.ModelInput()
{
    ImagePath = @"C:\dev\datasets\object-detection\fruit103.png",
};


var predictionTSV = ObjectDetectionTSV.Predict(modelInputTSV);

Console.WriteLine(predictionTSV.ToString());


mlContext = new MLContext();
mlContext.Log += MlContext_Log;

ObjectDetectionVOTT.Train(@"C:\dev\Scratch\ObjectDetective\ObjectDetective\ObjectDetectionVOTT.mlnet", mlContext);

var modelInputVOTT = new ObjectDetectionVOTT.ModelInput()
{
    ImagePath = @"C:\dev\datasets\OD-cats\OD-cats\vott-json-export\IMG_5098.jpg",
};


var predictionVOTT = ObjectDetectionVOTT.Predict(modelInputVOTT);

Console.WriteLine(predictionVOTT.ToString());



Console.WriteLine("Goodbye");
