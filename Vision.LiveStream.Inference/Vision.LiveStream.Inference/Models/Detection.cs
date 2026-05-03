namespace Vision.LiveStream.Inference.Models
{
    /// <summary>
    /// YOLOv8 추론 결과 한 건. 좌표계는 원본 이미지 픽셀 기준 (letterbox 역변환 후).
    /// </summary>
    public sealed class Detection
    {
        public float X { get; init; }

        public float Y { get; init; }

        public float Width { get; init; }

        public float Height { get; init; }

        public int ClassId { get; init; }

        public string ClassName { get; init; } = string.Empty;

        public float Confidence { get; init; }

        public override string ToString()
        {
            return $"{ClassName}({ClassId}) {Confidence:P1} @ ({X:F0},{Y:F0} {Width:F0}x{Height:F0})";
        }
    }
}
