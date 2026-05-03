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
            return Task.Run<IReadOnlyList<Detection>>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                LetterboxResult lb = YoloPreprocessor.Preprocess(bgrPixels, width, height);
                return _engine.Detect(lb, cancellationToken);
            }, cancellationToken);
        }
    }
}
