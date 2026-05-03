# Vision.LiveStream.Inference

WPF (.NET 10) + ONNX Runtime 기반 **실시간 RTSP 영상 객체 검출** 학습 프로젝트.
선행 프로젝트 [`Vision.OnnxTester`](https://github.com/) (정적 이미지 + YOLOv8) 의 검출 엔진을 베이스로,
RTSP 스트림을 받아 실시간으로 추론하고 화면에 박스를 그리는 것이 목표.

> **학습 목표 (Step 4):** 영상 수신 스레드 / AI 추론 스레드 / UI 렌더링(Dispatcher) 스레드를
> 완벽히 분리해서, 어느 한쪽이 막혀도 화면이 끊기지 않게 만든다.

---

## 1. 개발 환경

| 항목 | 버전 / 비고 |
|---|---|
| OS | Windows 11 |
| IDE | Visual Studio 2022 (또는 Rider) |
| .NET SDK | .NET 10 (`net10.0-windows`) |
| 언어 | C# (Nullable enable, ImplicitUsings enable) |
| UI | WPF |

### NuGet 패키지 (이미 csproj 에 포함됨)

| 패키지 | 용도 |
|---|---|
| `Microsoft.ML.OnnxRuntime` | ONNX 모델 추론 (CPU). GPU 쓰려면 `.Gpu` 로 교체 |
| `SixLabors.ImageSharp` | 이미지 픽셀 조작 (리사이즈, 정규화, HWC→CHW) |

---

## 2. 폴더 구조

```
Vision.LiveStream.Inference/                 <-- 저장소 루트
├── README.md
├── Tester/                                  <-- 가상 CCTV 환경 (서버 + 송출기 + 샘플 영상)
│   ├── cameraTest/
│   │   ├── Video1.mp4 / Video2.mp4 / Video3.mp4   샘플 CCTV 영상
│   │   ├── ffmpeg.exe                              스트림 송출기
│   │   ├── run_cameras_tcp.bat                     TCP 모드 다중 카메라 실행
│   │   └── run_cameras_udp.bat                     UDP 모드 다중 카메라 실행
│   └── mediamtx_v1.18.1_windows_amd64/
│       ├── mediamtx.exe                            RTSP 서버 (분배기)
│       └── mediamtx.yml                            서버 설정
└── Vision.LiveStream.Inference/             <-- 솔루션 + WPF 프로젝트
    └── Vision.LiveStream.Inference/
        ├── Assets/
        │   ├── Models/yolov8n.onnx
        │   └── TestImages/*.jpg
        ├── Common/   (RelayCommand, AsyncRelayCommand, BaseViewModel)
        ├── Models/   (Detection)
        ├── Services/ (CocoLabels, ImagePreprocessor, YoloV8Detector)
        ├── ViewModels/ (MainViewModel)
        └── MainWindow.xaml(.cs)
```

> `Tester/` 폴더의 `mediamtx.exe`, `ffmpeg.exe`, 샘플 mp4 는 학습 편의를 위해 저장소에 함께 포함.

---

## 3. 가상 CCTV 환경 구축 (간추림)

> 아래 절차는 PC 한 대 안에서 **RTSP 서버 → 송출기 → 클라이언트** 구조를 그대로 재현하는 흐름.

### Step 1. MediaMTX (RTSP 서버) 실행

PC 를 RTSP 분배기로 만들어 주는 서버.

- 다운로드: <https://github.com/bluenviron/mediamtx/releases> 에서 `mediamtx_vX.X.X_windows_amd64.zip`
- 본 저장소에는 `Tester/mediamtx_v1.18.1_windows_amd64/` 에 이미 포함됨
- 실행: `mediamtx.exe` 더블클릭
- 성공 로그: `[RTSP] listener opened on :8554`
- ⚠ 이 콘솔 창은 **끄지 말고 켜둘 것** (서버가 죽음)

### Step 2. FFmpeg (송출기) 준비

mp4 파일을 디먹싱 → RTSP 패킷으로 다시 먹싱해서 서버로 쏴주는 도구.

- 다운로드: <https://www.gyan.dev/ffmpeg/builds/> → `ffmpeg-master-latest-win64-gpl.zip`
- 압축의 `bin/ffmpeg.exe` 만 꺼내서 사용
- 본 저장소에는 `Tester/cameraTest/ffmpeg.exe` 로 이미 포함됨

### Step 3. 단일 카메라 송출 (수동 명령어)

`Tester/cameraTest/` 안에서 cmd 또는 PowerShell 열고:

```cmd
ffmpeg -re -stream_loop -1 -i Video1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1
```

옵션 의미

- `-re` : 원본 프레임레이트로 재생 (실시간 흉내)
- `-stream_loop -1` : 무한 반복
- `-c copy` : 재인코딩 없이 패킷만 복사 → CPU 거의 안 씀 (스트림 카피의 위력)
- `-f rtsp rtsp://localhost:8554/cam1` : RTSP 로 서버에 송출

`frame=... fps=...` 가 쭉 올라오면 정상.

### Step 4. VLC 로 수신 검증

C# 앱을 짜기 전에 상용 플레이어로 먼저 받아보자.

- VLC: <https://www.videolan.org/>
- VLC 실행 → **미디어 → 네트워크 스트림 열기** → `rtsp://localhost:8554/cam1` 입력 → 재생
- 영상이 뜨면 환경 구축 OK

---

## 4. 다중 카메라 일괄 실행 — `run_cameras_tcp.bat` / `run_cameras_udp.bat`

여러 채널(cam1, cam2, cam3 …) 을 한 번에 띄우려고 만든 배치 파일.
`Tester/cameraTest/` 폴더 안에 있는 `Video*.mp4` 파일을 자동으로 카운트해서
같은 개수만큼 `cam1`, `cam2`, `cam3` … 로 송출한다.

### 사용 방법

1. `mediamtx.exe` 가 떠 있는 상태인지 먼저 확인 (Step 1 참고)
2. `Tester/cameraTest/` 폴더로 이동
3. 둘 중 하나 더블클릭
   - `run_cameras_tcp.bat` — RTSP 트랜스포트를 **TCP** 로 강제
   - `run_cameras_udp.bat` — 기본(UDP) 트랜스포트 사용
4. 콘솔에 `Found N video files. Starting N cameras in background...` 출력
5. **창은 끄지 말 것** — 닫으면 송출도 끊김
6. 종료할 때는 콘솔에서 **아무 키나 누르면** `taskkill /F /IM ffmpeg.exe` 로 일괄 종료

### 송출되는 RTSP 주소

`Video1.mp4` → `rtsp://localhost:8554/cam1`
`Video2.mp4` → `rtsp://localhost:8554/cam2`
`Video3.mp4` → `rtsp://localhost:8554/cam3`
… (파일 개수만큼)

### TCP 와 UDP 의 차이 (간단히)

| 모드 | 특징 | 언제 쓰나 |
|---|---|---|
| **UDP** (`run_cameras_udp.bat`) | 기본값. 빠르지만 패킷 유실 가능 | LAN 내부 / 지연이 우선일 때 |
| **TCP** (`run_cameras_tcp.bat`) | `-rtsp_transport tcp` 강제. 손실에 강하고 NAT/방화벽에 잘 통과 | 네트워크가 불안정 / 영상 깨짐 발생 시 |

> 둘 다 동작이 안 되면 보통 **MediaMTX 가 안 떠있거나** Windows 방화벽이 8554 를 막은 경우.

### 동작 원리 (배치 파일 내부)

- `Video*.mp4` 패턴으로 파일 카운트
- `FOR /L` 루프로 `start /B` 해서 ffmpeg 를 백그라운드 실행 (창 안 뜸)
- 로그는 `> NUL 2>&1` 로 휴지통행 (콘솔 깨끗)
- 종료 키 입력 시 `taskkill /F /IM ffmpeg.exe` 로 모든 ffmpeg 프로세스 정리

---

## 5. 다음 작업 (Step 4 본 진행)

- [ ] RTSP 프레임 수신 스레드 (`OpenCvSharp` 또는 `LibVLCSharp` 검토)
- [ ] AI 추론 스레드 (현재 `YoloV8Detector` 재활용 + 프레임용 `Preprocess(byte[])` 오버로드 추가)
- [ ] UI 렌더링은 `Dispatcher.BeginInvoke` 로만
- [ ] 큐 기반 백프레셔(최신 프레임만 유지) 처리

---
