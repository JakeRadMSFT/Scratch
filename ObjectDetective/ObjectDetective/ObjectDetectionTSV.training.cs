﻿// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.AutoFormerV2;
// This file was auto-generated by ML.NET Model Builder.


namespace ObjectDetective
{
    public partial class ObjectDetectionTSV
    {

        public const string RetrainFilePath = @"C:\dev\datasets\object-detection\fruit-detection-ten.tsv";
        public const string RetrainImagePath = @"C:\dev\datasets\object-detection";
        public const char RetrainSeparatorChar = '\t';
        public const bool RetrainHasHeader = false;


        /// <summary>
        /// Train a new model with the provided dataset.
        /// </summary>
        /// <param name="outputModelPath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet"</param>
        /// <param name="inputDataFilePath">Path to the data file for training.</param>
        /// <param name="separatorChar">Separator character for delimited training file.</param>
        /// <param name="hasHeader">Boolean if training file has a header.</param>
        public static void Train(string outputModelPath, MLContext? mlContext = null, string inputDataFilePath = RetrainFilePath, char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader)
        {
            mlContext ??= new MLContext();

            var data = LoadIDataViewFromTSVFile(mlContext, inputDataFilePath, separatorChar, hasHeader);
            var model = RetrainModel(mlContext, data);
            SaveModel(mlContext, model, data, outputModelPath);
        }

        /// <summary>
        /// Load an IDataView from a file path.
        /// </summary>
        /// <param name="mlContext">The common context for all ML.NET operations.</param>
        /// <param name="inputDataFilePath">Path to the data file for training.</param>
        /// <param name="separatorChar">Separator character for delimited training file.</param>
        /// <param name="hasHeader">Boolean if training file has a header.</param>
        /// <returns>IDataView with loaded training data.</returns>
        public static IDataView LoadIDataViewFromTSVFile(MLContext mlContext, string inputDataFilePath, char separatorChar, bool hasHeader)
        {

            var data = mlContext.Data.LoadFromTextFile<ModelInput>(inputDataFilePath, separatorChar, hasHeader);


            var reshapePipline = mlContext.Transforms.Text.TokenizeIntoWords("Labels", separators: new char[] { ',' })
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Box", separators: new char[] { ',' }))
                .Append(mlContext.Transforms.Conversion.ConvertType("Box", "Box", DataKind.Single));

            var reshape = reshapePipline.Fit(data);

            return reshape.Transform(data);
        }


        /// <summary>
        /// Save a model at the specified path.
        /// </summary>
        /// <param name="mlContext">The common context for all ML.NET operations.</param>
        /// <param name="model">Model to save.</param>
        /// <param name="data">IDataView used to train the model.</param>
        /// <param name="modelSavePath">File path for saving the model. Should be similar to "C:\YourPath\ModelName.mlnet.</param>
        public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
        {
            // Pull the data schema from the IDataView used for training the model
            DataViewSchema dataViewSchema = data.Schema;

            using (var fs = File.Create(modelSavePath))
            {
                mlContext.Model.Save(model, dataViewSchema, fs);
            }
        }


        /// <summary>
        /// Retrains model using the pipeline generated as part of the training process.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData, string imageFolder = RetrainImagePath)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }


        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext, string imageFolder = RetrainImagePath)
        {
            var options = new ObjectDetectionTrainer.Options()
            {
                LabelColumnName = "Labels",
                ImageColumnName = "Image",
                BoundingBoxColumnName = "Box",
                PredictedLabelColumnName = "PredictedLabel",
                ScoreColumnName = "Score",
                ScoreThreshold = 0.2,
                MaxEpoch = 1,
            };

            var chain = new EstimatorChain<ITransformer>();

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Labels")
                .Append(mlContext.Transforms.LoadImages("Image", imageFolder, "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("Image", 400, 300))
                .Append(mlContext.MulticlassClassification.Trainers.ObjectDetection(options))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            return pipeline;
        }
    }
}
