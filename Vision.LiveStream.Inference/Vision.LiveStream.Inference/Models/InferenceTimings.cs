namespace Vision.LiveStream.Inference.Models
{
    public readonly struct InferenceTimings
    {
        public double PreprocessMs { get; init; }
        public double InferenceMs { get; init; }
        public double PostprocessMs { get; init; }
    }
}
