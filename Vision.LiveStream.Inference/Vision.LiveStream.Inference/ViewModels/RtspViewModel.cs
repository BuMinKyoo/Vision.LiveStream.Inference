using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Vision.LiveStream.Inference.Common;
using Vision.LiveStream.Inference.Models;
using Vision.LiveStream.Inference.Services.Rtsp;

namespace Vision.LiveStream.Inference.ViewModels
{
    /// <summary>
    /// RTSP 스트림 객체 검출 ViewModel.
    /// 책임: RtspFrameSource(수신) + IRtspFrameDetector(추론) 의 결과를 UI 바인딩으로 노출.
    /// 스레드 분리:
    ///   - 영상 표시: RtspFrameSource.FrameCaptured (캡처 스레드) → Dispatcher.BeginInvoke → WriteableBitmap
    ///   - 추론     : RtspFrameSource.Reader (latest-only) → Task → Dispatcher.BeginInvoke → Detections
    /// </summary>
    public class RtspViewModel : BaseViewModel, IDisposable
    {
        private readonly IRtspFrameDetector _detector;
        private readonly Dispatcher _dispatcher;

        private string _rtspUrl = "rtsp://localhost:8554/cam1";
        private bool _isStreaming;
        private string _statusMessage = "RTSP URL 입력 후 [연결] 버튼을 누르세요.";
        private WriteableBitmap? _imageSource;
        private int _imageWidth;
        private int _imageHeight;
        private double _displayFps;
        private double _inferenceFps;

        private RtspFrameSource? _source;
        private CancellationTokenSource? _cts;
        private Task? _inferenceTask;

        private readonly FpsCounter _displayFpsCounter = new();
        private readonly FpsCounter _inferenceFpsCounter = new();

        public RtspViewModel(IRtspFrameDetector detector)
        {
            _detector = detector;
            _dispatcher = Application.Current.Dispatcher;
            ConnectCommand = new RelayCommand(Connect, () => !IsStreaming && !string.IsNullOrWhiteSpace(RtspUrl));
            DisconnectCommand = new RelayCommand(Disconnect, () => IsStreaming);
        }

        public ObservableCollection<Detection> Detections { get; } = new();

        public RelayCommand ConnectCommand { get; }
        public RelayCommand DisconnectCommand { get; }

        public string RtspUrl
        {
            get => _rtspUrl;
            set
            {
                if (_rtspUrl == value)
                {
                    return;
                }
                _rtspUrl = value;
                OnPropertyChanged();
                ConnectCommand.RaiseCanExecuteChanged();
            }
        }

        public bool IsStreaming
        {
            get => _isStreaming;
            private set
            {
                if (_isStreaming == value)
                {
                    return;
                }
                _isStreaming = value;
                OnPropertyChanged();
                ConnectCommand.RaiseCanExecuteChanged();
                DisconnectCommand.RaiseCanExecuteChanged();
            }
        }

        public string StatusMessage
        {
            get => _statusMessage;
            private set
            {
                if (_statusMessage == value)
                {
                    return;
                }
                _statusMessage = value;
                OnPropertyChanged();
            }
        }

        public WriteableBitmap? ImageSource
        {
            get => _imageSource;
            private set
            {
                if (_imageSource == value)
                {
                    return;
                }
                _imageSource = value;
                OnPropertyChanged();
            }
        }

        public int ImageWidth
        {
            get => _imageWidth;
            private set
            {
                if (_imageWidth == value)
                {
                    return;
                }
                _imageWidth = value;
                OnPropertyChanged();
            }
        }

        public int ImageHeight
        {
            get => _imageHeight;
            private set
            {
                if (_imageHeight == value)
                {
                    return;
                }
                _imageHeight = value;
                OnPropertyChanged();
            }
        }

        public double DisplayFps
        {
            get => _displayFps;
            private set
            {
                if (Math.Abs(_displayFps - value) < 0.01)
                {
                    return;
                }
                _displayFps = value;
                OnPropertyChanged();
            }
        }

        public double InferenceFps
        {
            get => _inferenceFps;
            private set
            {
                if (Math.Abs(_inferenceFps - value) < 0.01)
                {
                    return;
                }
                _inferenceFps = value;
                OnPropertyChanged();
            }
        }

        private void Connect()
        {
            if (IsStreaming)
            {
                return;
            }

            try
            {
                _source = new RtspFrameSource(RtspUrl);
                _source.StatusChanged += OnSourceStatusChanged;
                _source.FrameCaptured += OnFrameCapturedForDisplay;
                _source.Start();

                _cts = new CancellationTokenSource();
                _inferenceTask = Task.Run(() => InferenceLoopAsync(_cts.Token));

                IsStreaming = true;
                StatusMessage = "연결 중...";
            }
            catch (Exception ex)
            {
                StatusMessage = $"연결 실패: {ex.Message}";
                Disconnect();
            }
        }

        private void Disconnect()
        {
            if (!IsStreaming && _source == null)
            {
                return;
            }

            try
            {
                _cts?.Cancel();

                if (_source != null)
                {
                    _source.FrameCaptured -= OnFrameCapturedForDisplay;
                    _source.StatusChanged -= OnSourceStatusChanged;
                    _source.Stop();
                    _source.Dispose();
                    _source = null;
                }

                _inferenceTask = null;
                _cts?.Dispose();
                _cts = null;

                IsStreaming = false;
                Detections.Clear();
                DisplayFps = 0;
                InferenceFps = 0;
            }
            catch (Exception ex)
            {
                StatusMessage = $"중지 중 오류: {ex.Message}";
            }
        }

        private void OnSourceStatusChanged(object? sender, string message)
        {
            _dispatcher.BeginInvoke(() => StatusMessage = message);
        }

        /// <summary>
        /// 캡처 스레드에서 직접 호출됨. UI 작업은 Dispatcher 로 마샬링.
        /// 이 핸들러는 추론을 기다리지 않고 즉시 화면을 갱신함 → 영상 끊김 방지.
        /// </summary>
        private void OnFrameCapturedForDisplay(object? sender, RtspFrame frame)
        {
            // 캡처 스레드에서 측정
            _displayFpsCounter.Tick(out double fps);

            _dispatcher.BeginInvoke(() =>
            {
                RenderFrameToBitmap(frame);
                DisplayFps = fps;
            }, DispatcherPriority.Render);
        }

        /// <summary>
        /// UI 스레드에서만 호출되는 비트맵 갱신.
        /// </summary>
        private void RenderFrameToBitmap(RtspFrame frame)
        {
            if (_imageSource == null
                || _imageSource.PixelWidth != frame.Width
                || _imageSource.PixelHeight != frame.Height)
            {
                // 첫 프레임 또는 해상도 변경 시 비트맵 재생성
                ImageSource = new WriteableBitmap(
                    frame.Width, frame.Height, 96, 96, PixelFormats.Bgr24, null);
                ImageWidth = frame.Width;
                ImageHeight = frame.Height;
            }

            int stride = frame.Width * 3;
            _imageSource!.WritePixels(
                new Int32Rect(0, 0, frame.Width, frame.Height),
                frame.BgrPixels,
                stride,
                0);
        }

        /// <summary>
        /// 별도 Task 로 돌아가는 추론 루프.
        /// Reader 가 latest-only 라 추론이 느려도 옛 프레임 누적되지 않음.
        /// </summary>
        private async Task InferenceLoopAsync(CancellationToken ct)
        {
            if (_source == null)
            {
                return;
            }

            try
            {
                await foreach (RtspFrame frame in _source.Reader.ReadAllAsync(ct))
                {
                    if (ct.IsCancellationRequested)
                    {
                        break;
                    }

                    var sw = Stopwatch.StartNew();
                    IReadOnlyList<Detection> detections = await _detector
                        .DetectAsync(frame.BgrPixels, frame.Width, frame.Height, ct)
                        .ConfigureAwait(false);
                    sw.Stop();

                    _inferenceFpsCounter.Tick(out double fps);

                    // fire-and-forget: UI 업데이트가 끝나길 기다릴 필요 없음
                    _ = _dispatcher.BeginInvoke(() =>
                    {
                        Detections.Clear();
                        foreach (Detection d in detections)
                        {
                            Detections.Add(d);
                        }
                        InferenceFps = fps;
                    }, DispatcherPriority.Background);
                }
            }
            catch (OperationCanceledException)
            {
                // 정상 취소
            }
            catch (Exception ex)
            {
                _ = _dispatcher.BeginInvoke(() => StatusMessage = $"추론 오류: {ex.Message}");
            }
        }

        public void Dispose()
        {
            Disconnect();
        }

        /// <summary>
        /// 1초 슬라이딩 윈도우 FPS 카운터. 호출하는 스레드가 일관되어야 정확함.
        /// </summary>
        private sealed class FpsCounter
        {
            private readonly Stopwatch _sw = Stopwatch.StartNew();
            private int _count;
            private long _lastReportMs;
            private double _lastFps;

            public void Tick(out double fps)
            {
                _count++;
                long elapsed = _sw.ElapsedMilliseconds - _lastReportMs;
                if (elapsed >= 1000)
                {
                    _lastFps = _count * 1000.0 / elapsed;
                    _count = 0;
                    _lastReportMs = _sw.ElapsedMilliseconds;
                }
                fps = _lastFps;
            }
        }
    }
}
