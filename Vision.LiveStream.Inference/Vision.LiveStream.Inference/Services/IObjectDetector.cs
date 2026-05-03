using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Vision.LiveStream.Inference.Models;

namespace Vision.LiveStream.Inference.Services
{
    /// <summary>
    /// 이미지에서 객체를 검출하는 추상화. YOLOv8 외에 다른 모델로 교체 가능하도록 분리.
    /// </summary>
    public interface IObjectDetector : IDisposable
    {
        /// <summary>
        /// 입력 이미지 경로의 객체를 검출. 결과 좌표는 원본 이미지 픽셀 기준.
        /// </summary>
        Task<IReadOnlyList<Detection>> DetectAsync(string imagePath, CancellationToken cancellationToken = default);
    }
}
