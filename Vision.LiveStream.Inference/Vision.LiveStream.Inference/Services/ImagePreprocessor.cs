using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Vision.LiveStream.Inference.Services
{
    /// <summary>
    /// 전처리 결과. 추론 후 letterbox 좌표를 원본으로 되돌리는 데 Scale/PadX/PadY/Original* 가 사용됨.
    /// </summary>
    public sealed record LetterboxResult(
        DenseTensor<float> Tensor,
        float Scale,
        int PadX,
        int PadY,
        int OriginalWidth,
        int OriginalHeight);

    /// <summary>
    /// JPG/PNG 이미지를 YOLOv8 입력 텐서 [1,3,640,640] 로 변환.
    /// 4단계: (1) letterbox 리사이즈 (2) RGB 정렬 (3) 0~1 정규화 (4) HWC→CHW 차원 재배열.
    /// </summary>
    public static class ImagePreprocessor
    {
        public const int InputSize = 640;

        // YOLOv8 letterbox 표준 패딩 색 (회색)
        private const float PadValueNormalized = 114f / 255f;

        public static LetterboxResult Preprocess(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);

            int origW = image.Width;
            int origH = image.Height;

            float scale = System.Math.Min(
                (float)InputSize / origW,
                (float)InputSize / origH);

            int newW = (int)System.Math.Round(origW * scale);
            int newH = (int)System.Math.Round(origH * scale);
            int padX = (InputSize - newW) / 2;
            int padY = (InputSize - newH) / 2;

            image.Mutate(x => x.Resize(newW, newH));

            var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            // 회색 패딩으로 채우기 (letterbox 빈 영역)
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < InputSize; y++)
                {
                    for (int x = 0; x < InputSize; x++)
                    {
                        tensor[0, c, y, x] = PadValueNormalized;
                    }
                }
            }

            // 리사이즈된 이미지 픽셀을 패딩된 영역에 복사 (HWC → CHW + 정규화 동시)
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < newH; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < newW; x++)
                    {
                        Rgb24 px = row[x];
                        tensor[0, 0, y + padY, x + padX] = px.R / 255f;
                        tensor[0, 1, y + padY, x + padX] = px.G / 255f;
                        tensor[0, 2, y + padY, x + padX] = px.B / 255f;
                    }
                }
            });

            return new LetterboxResult(tensor, scale, padX, padY, origW, origH);
        }
    }
}
