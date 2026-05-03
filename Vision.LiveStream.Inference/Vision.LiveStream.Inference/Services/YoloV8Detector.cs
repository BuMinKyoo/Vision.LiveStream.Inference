using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vision.LiveStream.Inference.Models;

namespace Vision.LiveStream.Inference.Services
{
    /// <summary>
    /// YOLOv8 ONNX 모델을 사용한 객체 검출기.
    /// 출력 텐서: [1, 84, 8400] = (cx, cy, w, h, class0..class79) × 8400 후보.
    /// </summary>
    public sealed class YoloV8Detector : IObjectDetector
    {
        private const float ConfidenceThreshold = 0.25f;
        private const float IouThreshold = 0.45f;

        private readonly InferenceSession _session;
        private readonly string _inputName;

        public YoloV8Detector(string modelPath)
        {
            _session = new InferenceSession(modelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }

        public Task<IReadOnlyList<Detection>> DetectAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            return Task.Run<IReadOnlyList<Detection>>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                LetterboxResult lb = ImagePreprocessor.Preprocess(imagePath);

                cancellationToken.ThrowIfCancellationRequested();

                var input = NamedOnnxValue.CreateFromTensor(_inputName, lb.Tensor);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(new[] { input });

                Tensor<float> output = results.First().AsTensor<float>();

                List<Detection> candidates = ParseOutput(output, lb);

                cancellationToken.ThrowIfCancellationRequested();

                return ApplyNms(candidates, IouThreshold);
            }, cancellationToken);
        }

        private static List<Detection> ParseOutput(Tensor<float> output, LetterboxResult lb)
        {
            // output dims: [1, 84, 8400]
            int numChannels = output.Dimensions[1];
            int numAnchors = output.Dimensions[2];
            int numClasses = numChannels - 4;

            var candidates = new List<Detection>();

            for (int i = 0; i < numAnchors; i++)
            {
                int bestClass = -1;
                float bestScore = ConfidenceThreshold;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, 4 + c, i];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                if (bestClass < 0)
                {
                    continue;
                }

                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                // letterbox 좌표 → 원본 이미지 좌표 (패딩 빼고 스케일 역연산)
                float x = (cx - w / 2f - lb.PadX) / lb.Scale;
                float y = (cy - h / 2f - lb.PadY) / lb.Scale;
                float bw = w / lb.Scale;
                float bh = h / lb.Scale;

                // 이미지 영역으로 클램프
                x = Math.Clamp(x, 0f, lb.OriginalWidth);
                y = Math.Clamp(y, 0f, lb.OriginalHeight);
                bw = Math.Min(bw, lb.OriginalWidth - x);
                bh = Math.Min(bh, lb.OriginalHeight - y);

                if (bw <= 0 || bh <= 0)
                {
                    continue;
                }

                candidates.Add(new Detection
                {
                    X = x,
                    Y = y,
                    Width = bw,
                    Height = bh,
                    ClassId = bestClass,
                    ClassName = CocoLabels.Get(bestClass),
                    Confidence = bestScore
                });
            }

            return candidates;
        }

        private static List<Detection> ApplyNms(List<Detection> input, float iouThreshold)
        {
            var sorted = input.OrderByDescending(d => d.Confidence).ToList();
            var result = new List<Detection>();
            var suppressed = new bool[sorted.Count];

            for (int i = 0; i < sorted.Count; i++)
            {
                if (suppressed[i])
                {
                    continue;
                }

                result.Add(sorted[i]);

                for (int j = i + 1; j < sorted.Count; j++)
                {
                    if (suppressed[j])
                    {
                        continue;
                    }
                    if (sorted[i].ClassId != sorted[j].ClassId)
                    {
                        continue;
                    }
                    if (Iou(sorted[i], sorted[j]) > iouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }

            return result;
        }

        private static float Iou(Detection a, Detection b)
        {
            float x1 = Math.Max(a.X, b.X);
            float y1 = Math.Max(a.Y, b.Y);
            float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

            if (x2 <= x1 || y2 <= y1)
            {
                return 0f;
            }

            float intersection = (x2 - x1) * (y2 - y1);
            float union = a.Width * a.Height + b.Width * b.Height - intersection;
            return intersection / union;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
