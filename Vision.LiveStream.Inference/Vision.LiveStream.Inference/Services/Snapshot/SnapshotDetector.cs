using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.LiveStream.Inference.Models;
using Vision.LiveStream.Inference.Services.Yolo;

namespace Vision.LiveStream.Inference.Services.Snapshot
{
    /// <summary>
    /// 정적 이미지(스냅샷) 도메인 어댑터.
    /// 외부에서 주입된 YoloInferenceEngine 을 공유하며, 자기 자신은 세션을 소유하지 않음
    /// (RTSP 도메인과 같은 엔진을 공유할 수 있도록).
    /// </summary>
    public sealed class SnapshotDetector : ISnapshotDetector
    {
        private readonly YoloInferenceEngine _engine;

        public SnapshotDetector(YoloInferenceEngine engine)
        {
            _engine = engine;
        }

        public Task<IReadOnlyList<Detection>> DetectAsync(string imagePath, CancellationToken cancellationToken = default)
        {
            return Task.Run<IReadOnlyList<Detection>>(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                LetterboxResult lb = YoloPreprocessor.Preprocess(imagePath);
                return _engine.Detect(lb, cancellationToken);
            }, cancellationToken);
        }
    }
}
