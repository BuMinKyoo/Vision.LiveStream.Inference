using Microsoft.ML.OnnxRuntime.Tensors;

namespace Vision.LiveStream.Inference.Services.Yolo
{
    /// <summary>
    /// 전처리 결과. 추론 후 letterbox 좌표를 원본으로 되돌리는 데
    /// Scale/PadX/PadY/Original* 가 사용됨.
    /// </summary>
    public sealed record LetterboxResult(
        DenseTensor<float> Tensor,
        float Scale,
        int PadX,
        int PadY,
        int OriginalWidth,
        int OriginalHeight);
}
