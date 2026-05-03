using System;
using System.IO;
using System.Windows;
using Vision.LiveStream.Inference.Services;
using Vision.LiveStream.Inference.ViewModels;

namespace Vision.LiveStream.Inference
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly YoloV8Detector? _detector;

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
                _detector = new YoloV8Detector(modelPath);
                DataContext = new MainViewModel(_detector);

                Closed += (_, _) => _detector?.Dispose();
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
