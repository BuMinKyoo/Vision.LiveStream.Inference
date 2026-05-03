using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Vision.LiveStream.Inference.Services.Yolo
{
    /// <summary>
    /// YOLOv8 입력 텐서 [1,3,640,640] 만들기. 책임 1개:
    /// "어떤 형태의 픽셀이 들어오든 letterbox + 정규화 + CHW 텐서로 변환".
    /// 진입점은 입력 형태별로 둘 (파일 경로 / 메모리 BGR byte[]).
    /// 4단계: (1) letterbox 리사이즈 (2) RGB 정렬 (3) 0~1 정규화 (4) HWC→CHW 차원 재배열.
    /// </summary>
    public static class YoloPreprocessor
    {
        public const int InputSize = 640;

        // YOLOv8 letterbox 표준 패딩 색 (회색)
        private const float PadValueNormalized = 114f / 255f;

        /// <summary>
        /// 디스크의 JPG/PNG 파일을 읽어 전처리. (Snapshot 도메인용)
        /// </summary>
        public static LetterboxResult Preprocess(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            return PreprocessCore(image);
        }

        /// <summary>
        /// 메모리상 BGR 픽셀(OpenCV Mat 기본 포맷)을 받아 전처리. (RTSP 프레임 도메인용)
        /// bgrPixels 길이 = width * height * 3, 채널 순서 B-G-R, row-major.
        /// </summary>
        public static LetterboxResult Preprocess(byte[] bgrPixels, int width, int height)
        {
            using var image = BgrBytesToImage(bgrPixels, width, height);
            return PreprocessCore(image);
        }

        private static Image<Rgb24> BgrBytesToImage(byte[] bgr, int width, int height)
        {
            var image = new Image<Rgb24>(width, height);
            image.ProcessPixelRows(accessor =>
            {
                int idx = 0;
                for (int y = 0; y < height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        // OpenCV BGR → ImageSharp Rgb24(RGB) 채널 스왑
                        row[x] = new Rgb24(bgr[idx + 2], bgr[idx + 1], bgr[idx]);
                        idx += 3;
                    }
                }
            });
            return image;
        }

        private static LetterboxResult PreprocessCore(Image<Rgb24> image)
        {
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
