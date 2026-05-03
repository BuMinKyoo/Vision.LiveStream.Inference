using System;
using System.IO;
using System.Windows;
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
                // 추론 엔진 1개를 로드해서 도메인 어댑터들에게 공유.
                // (지금은 정적 이미지 도메인만 쓰지만, RTSP 도메인도 같은 엔진을 받게 됨)
                _engine = new YoloInferenceEngine(modelPath);
                var snapshotDetector = new SnapshotDetector(_engine);
                DataContext = new MainViewModel(snapshotDetector);

                Closed += (_, _) => _engine?.Dispose();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"검출기 초기화 실패:\n{ex.Message}",
                    "초기화 오류",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }
        }
    }
}
