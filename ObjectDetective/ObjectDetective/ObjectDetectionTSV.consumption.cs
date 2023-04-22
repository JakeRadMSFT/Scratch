// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ObjectDetective
{
    public partial class ObjectDetectionTSV
    {

        /// <summary>
        /// model input class for ObjectDetection.
        /// </summary>
        #region model input class
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName(@"ImagePath")]
            public string ImagePath { get; set; }

            [LoadColumn(1)]
            [ColumnName(@"Labels")]
            public string Labels { get; set; }

            [LoadColumn(2)]
            [ColumnName(@"Box")]
            public string Box { get; set; }
        }
        #endregion

        /// <summary>
        /// model output class for ObjectDetection.
        /// </summary>
        #region model output class

        public class ModelOutput
        {
            [ColumnName("Box")]
            public float[] Boxes { get; set; } = new float[0];

            [ColumnName("PredictedLabel")]
            public string[] Labels { get; set; } = new string[0];

            [ColumnName("Score")]
            public float[] Scores { get; set; } = new float[0];

            public BoundingBox[] BoundingBoxes
            {
                get
                {
                    var boundingBoxes = Enumerable.Range(0, this.Labels.Length)
                              .Select((index) =>
                              {
                                  var boxes = this.Boxes;
                                  var scores = this.Scores;
                                  var labels = this.Labels;

                                  return new BoundingBox()
                                  {
                                      Left = boxes[index * 4],
                                      Top = boxes[(index * 4) + 1],
                                      Right = boxes[(index * 4) + 2],
                                      Bottom = boxes[(index * 4) + 3],
                                      Score = scores[index],
                                      Label = labels[index],
                                  };
                              }).ToArray();
                    return boundingBoxes;
                }
            }


            public override string ToString()
            {
                return string.Join("\n", this.BoundingBoxes.Select(x => x.ToString()));
            }

            public class BoundingBox
            {
                public float Top;

                public float Left;

                public float Right;

                public float Bottom;

                public string Label;

                public float Score;

                public override string ToString()
                {
                    return $"Top: {this.Top}, Left: {this.Left}, Right: {this.Right}, Bottom: {this.Bottom}, Label: {this.Label}, Score: {this.Score}";
                }
            }
        }
        #endregion


        private static string MLNetModelPath = @"C:\dev\Scratch\ObjectDetective\ObjectDetective\ObjectDetectionTSV.mlnet";

        public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);


        private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
        {
            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
        }

        /// <summary>
        /// Use this method to predict on <see cref="ModelInput"/>.
        /// </summary>
        /// <param name="input">model input.</param>
        /// <returns><seealso cref=" ModelOutput"/></returns>
        public static ModelOutput Predict(ModelInput input)
        {
            var predEngine = PredictEngine.Value;
            return predEngine.Predict(input);
        }
    }
}
