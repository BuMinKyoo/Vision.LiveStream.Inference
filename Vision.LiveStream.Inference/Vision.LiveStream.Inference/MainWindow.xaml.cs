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
        private readonly YoloInferenceEngine? _cpuEngine;
        private readonly YoloInferenceEngine? _gpuEngine;
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
                _cpuEngine = new YoloInferenceEngine(modelPath, InferenceDevice.Cpu);

                // GPU 엔진 초기화 실패(CUDA 미설치 등)해도 앱은 CPU 모드로 계속 실행
                YoloInferenceEngine? gpuEngine = null;
                try
                {
                    gpuEngine = new YoloInferenceEngine(modelPath, InferenceDevice.Gpu);
                    _gpuEngine = gpuEngine;
                }
                catch (Exception ex)
                {
                    MessageBox.Show(
                        $"GPU 초기화 실패 - CPU 모드로만 실행됩니다.\n\n{ex.Message}",
                        "GPU 경고",
                        MessageBoxButton.OK,
                        MessageBoxImage.Warning);
                }

                var snapshotDetector = new SnapshotDetector(_cpuEngine);
                var cpuRtspDetector = new RtspFrameDetector(_cpuEngine);
                var gpuRtspDetector = new RtspFrameDetector(_gpuEngine ?? _cpuEngine); // GPU 실패 시 CPU로 폴백

                _snapshotVm = new SnapshotViewModel(snapshotDetector);
                _rtspVm = new RtspViewModel(cpuRtspDetector, gpuRtspDetector);

                DataContext = new ShellViewModel(_snapshotVm, _rtspVm);

                Closed += (_, _) =>
                {
                    _snapshotVm?.Dispose();
                    _rtspVm?.Dispose();
                    _cpuEngine?.Dispose();
                    _gpuEngine?.Dispose();
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
