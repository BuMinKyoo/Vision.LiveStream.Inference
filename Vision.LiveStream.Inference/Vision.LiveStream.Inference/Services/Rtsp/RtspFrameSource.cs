using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using OpenCvSharp;

namespace Vision.LiveStream.Inference.Services.Rtsp
{
    /// <summary>
    /// RTSP URL 에서 프레임을 읽어 1슬롯 latest-only 큐로 노출하는 서비스.
    /// 책임 1개: "수신 → 큐 채우기".
    /// 추론/렌더링 책임은 호출자(ViewModel) 가 가짐.
    ///
    /// 스레드 모델:
    ///   - 자기 전용 백그라운드 스레드 1개에서 VideoCapture.Read() 블로킹 루프
    ///   - Channel 의 FullMode=DropOldest 로 쓰기 비차단 → 소비자가 늦어도 송신 안 막힘
    ///   - 결과: 소비자는 항상 "가장 최신 프레임" 만 보게 됨 (영상 끊김 방지의 핵심)
    /// </summary>
    public sealed class RtspFrameSource : IDisposable
    {
        private readonly string _url;
        private readonly Channel<RtspFrame> _channel;

        private CancellationTokenSource? _cts;
        private Thread? _captureThread;
        private volatile bool _isRunning;

        public RtspFrameSource(string rtspUrl)
        {
            _url = rtspUrl;
            _channel = Channel.CreateBounded<RtspFrame>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true
            });
        }

        /// <summary>
        /// 소비자(ViewModel) 는 이 Reader 에서 ReadAsync / WaitToReadAsync 로 프레임 받음.
        /// </summary>
        public ChannelReader<RtspFrame> Reader => _channel.Reader;

        public bool IsRunning => _isRunning;

        /// <summary>
        /// 연결/오류/종료 상태 변화 알림 (UI 에 출력하기 좋게).
        /// </summary>
        public event EventHandler<string>? StatusChanged;

        /// <summary>
        /// 프레임이 도착할 때마다 캡처 스레드에서 발생.
        /// 영상 표시(WriteableBitmap)용 — 모든 프레임을 받음. 핸들러에서 무거운 작업 금지.
        /// 추론은 별도로 Reader(채널)에서 latest-only 로 받을 것.
        /// </summary>
        public event EventHandler<RtspFrame>? FrameCaptured;

        public void Start()
        {
            if (_isRunning)
            {
                return;
            }

            _cts = new CancellationTokenSource();
            _captureThread = new Thread(() => CaptureLoop(_cts.Token))
            {
                IsBackground = true,
                Name = $"RtspCapture[{_url}]"
            };
            _isRunning = true;
            _captureThread.Start();
        }

        public void Stop()
        {
            if (!_isRunning)
            {
                return;
            }

            _isRunning = false;
            _cts?.Cancel();
            _captureThread?.Join(TimeSpan.FromSeconds(2));
            _captureThread = null;
            _cts?.Dispose();
            _cts = null;
        }

        private void CaptureLoop(CancellationToken ct)
        {
            VideoCapture? capture = null;
            Mat? mat = null;
            try
            {
                capture = new VideoCapture(_url);
                if (!capture.IsOpened())
                {
                    RaiseStatus($"열기 실패: {_url}");
                    return;
                }

                RaiseStatus($"연결됨: {_url} ({capture.FrameWidth}x{capture.FrameHeight} @ {capture.Fps:F1} fps)");

                mat = new Mat();
                int consecutiveFailures = 0;

                while (!ct.IsCancellationRequested)
                {
                    if (!capture.Read(mat) || mat.Empty())
                    {
                        consecutiveFailures++;
                        if (consecutiveFailures >= 30)
                        {
                            // 1초간 못 읽으면(33ms × 30) 끊긴 걸로 판단
                            RaiseStatus("프레임 수신 끊김");
                            break;
                        }
                        Thread.Sleep(33);
                        continue;
                    }

                    consecutiveFailures = 0;

                    int width = mat.Width;
                    int height = mat.Height;
                    int byteCount = width * height * 3;

                    var bgr = new byte[byteCount];
                    Marshal.Copy(mat.Data, bgr, 0, byteCount);

                    var frame = new RtspFrame(bgr, width, height, DateTime.UtcNow);

                    // 1) 영상 표시용 — 모든 프레임 즉시 알림 (구독자가 Dispatcher 로 던질 책임)
                    FrameCaptured?.Invoke(this, frame);

                    // 2) 추론용 — DropOldest 정책이라 항상 즉시 성공. 옛 프레임은 폐기됨.
                    _channel.Writer.TryWrite(frame);
                }
            }
            catch (Exception ex)
            {
                RaiseStatus($"캡처 오류: {ex.Message}");
            }
            finally
            {
                mat?.Dispose();
                capture?.Dispose();
                _channel.Writer.TryComplete();
                RaiseStatus("정지");
            }
        }

        private void RaiseStatus(string message)
        {
            StatusChanged?.Invoke(this, message);
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
