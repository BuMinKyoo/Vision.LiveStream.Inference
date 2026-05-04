using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.LiveStream.Inference.Models;
using Vision.LiveStream.Inference.Services.Yolo;

namespace Vision.LiveStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP 프레임(메모리상 BGR 픽셀)에서 객체를 검출하는 도메인 어댑터.
    /// 외부에서 주입된 YoloInferenceEngine 을 공유.
    /// </summary>
    public sealed class RtspFrameDetector : IRtspFrameDetector
    {
        private readonly YoloInferenceEngine _engine;

        public RtspFrameDetector(YoloInferenceEngine engine)
        {
            _engine = engine;
        }

        public Task<IReadOnlyList<Detection>> DetectAsync(byte[] bgrPixels, int width, int height, CancellationToken cancellationToken = default)
        {
            // Task.Run: 전처리 + 추론은 CPU 집약적이므로 ThreadPool 스레드에서 실행
            return Task.Run<IReadOnlyList<Detection>>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                // 전처리: BGR byte[] → letterbox 리사이즈 → 정규화 → CHW 텐서 [1,3,640,640]
                LetterboxResult lb = YoloPreprocessor.Preprocess(bgrPixels, width, height);

                // 추론: ONNX 세션 실행 → 후처리(NMS) → 원본 좌표계 Detection 리스트
                return _engine.Detect(lb, cancellationToken);
            }, cancellationToken);
        }
    }
}
