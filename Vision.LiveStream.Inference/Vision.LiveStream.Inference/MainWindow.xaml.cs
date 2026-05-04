using System;
using System.IO;
using System.Windows;
using Vision.LiveStream.Inference.Services.Rtsp;
using Vision.LiveStream.Inference.Services.Snapshot;
using Vision.LiveStream.Inference.Services.Yolo;
using Vision.LiveStream.Inference.ViewModels;

namespace Vision.LiveStream.Inference
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly YoloInferenceEngine? _engine;
        private SnapshotViewModel? _snapshotVm;
        private RtspViewModel? _rtspVm;

        public MainWindow()
        {
            InitializeComponent();

            string modelPath = Path.Combine(
                AppContext.BaseDirectory, "Assets", "Models", "yolov8n.onnx");

            if (!File.Exists(modelPath))
            {
                MessageBox.Show(
                    "ONNX 모델 파일을 찾을 수 없습니다.\n\n경로: " + modelPath +
                    "\n\nyolov8n.onnx 를 해당 경로에 배치하거나 빌드 후 다시 실행하세요.",
                    "모델 누락",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
                return;
            }

            try
            {
                // 추론 엔진 1개를 두 도메인 어댑터가 공유 → ONNX 모델 메모리 1회만 로드.
                _engine = new YoloInferenceEngine(modelPath);

                var snapshotDetector = new SnapshotDetector(_engine);
                var rtspDetector = new RtspFrameDetector(_engine);

                _snapshotVm = new SnapshotViewModel(snapshotDetector);
                _rtspVm = new RtspViewModel(rtspDetector);

                DataContext = new ShellViewModel(_snapshotVm, _rtspVm);

                Closed += (_, _) =>
                {
                    _snapshotVm?.Dispose();
                    _rtspVm?.Dispose();
                    _engine?.Dispose();
                };
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"초기화 실패:\n{ex.Message}",
                    "초기화 오류",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }
    }
}
