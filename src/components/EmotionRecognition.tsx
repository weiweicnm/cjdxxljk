import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Settings, CheckCircle2, AlertCircle, Loader2, Smile, Camera, Video, StopCircle } from 'lucide-react';
import * as ort from 'onnxruntime-web';
import { clsx } from 'clsx';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

export function EmotionRecognition() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  
  // 模型与预处理配置 - 默认值已对齐 Ultralytics 标准训练
  const [labels, setLabels] = useState('惊讶,害怕,厌恶,开心,难过,生气,中性');
  const [inputSize, setInputSize] = useState('320');
  const [channels, setChannels] = useState<'3' | '1'>('3');
  const [normalization, setNormalization] = useState<'imagenet' | 'none'>('none');

  const [mode, setMode] = useState<'image' | 'camera'>('image');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const isProcessingRef = useRef<boolean>(false);

  const [prediction, setPrediction] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    let interval: any;
    if (isCameraRunning && session && mode === 'camera') {
      interval = setInterval(async () => {
        if (isProcessingRef.current || !videoRef.current || videoRef.current.readyState < 2) return;
        
        isProcessingRef.current = true;
        try {
          const res = await performInference(videoRef.current);
          setPrediction(res);
        } catch (e) {
          console.error("摄像头推理出错", e);
        } finally {
          isProcessingRef.current = false;
        }
      }, 300);
    }
    return () => clearInterval(interval);
  }, [isCameraRunning, session, mode, inputSize, channels, normalization, labels]);

  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setModelStatus('loading');
    setErrorMessage('');
    setPrediction(null);

    try {
      const buffer = await file.arrayBuffer();
      const newSession = await ort.InferenceSession.create(buffer, { 
        executionProviders: ['wasm'] // 先禁用webgl避免精度问题，确认可用后再开启
      });
      setSession(newSession);
      setModelStatus('ready');
      setImageSrc(null);
    } catch (err: any) {
      console.error(err);
      setModelStatus('error');
      setErrorMessage(err.message || '模型加载失败，请确保上传的是有效的 .onnx 格式文件。');
    }
  };

  useEffect(() => {
    const loadDefaultModel = async () => {
      try {
        setModelStatus('loading');
        const response = await fetch('/best.onnx');
        
        if (!response.ok) {
          throw new Error(`模型文件未找到或网络错误: ${response.status}`);
        }

        const buffer = await response.arrayBuffer();
        const session = await ort.InferenceSession.create(buffer, { 
          executionProviders: ['wasm']
        });
        
        setSession(session);
        setModelStatus('ready');
        setErrorMessage('');
        console.log("✅ 默认模型自动加载成功");
      } catch (err: any) {
        console.error("❌ 自动加载模型失败:", err);
        setModelStatus('error');
        setErrorMessage(
          '默认模型加载失败。请检查控制台错误，' +
          '或手动上传模型文件'
        );
      }
    };

    loadDefaultModel();
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setImageSrc(event.target?.result as string);
      setPrediction(null);
    };
    reader.readAsDataURL(file);
  };

  // ========== 核心推理函数 - 已全部对齐 Ultralytics 标准 ==========
  const performInference = async (source: CanvasImageSource): Promise<string> => {
    if (!session) throw new Error("模型未加载");

    const targetSize = Number(inputSize);
    const channelNum = Number(channels);

    const canvas = document.createElement('canvas');
    canvas.width = targetSize;
    canvas.height = targetSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error("无法创建画布处理图像");

    // 获取源图像尺寸
    let sWidth = typeof source.width === 'number' ? source.width : 0;
    let sHeight = typeof source.height === 'number' ? source.height : 0;
    if (source instanceof HTMLVideoElement) {
      sWidth = source.videoWidth;
      sHeight = source.videoHeight;
    }

    // ===== 修复1：标准 Letterbox 缩放 + 官方填充色 114 =====
    if (sWidth > 0 && sHeight > 0) {
      const ratio = Math.min(targetSize / sWidth, targetSize / sHeight);
      const newW = Math.round(sWidth * ratio);
      const newH = Math.round(sHeight * ratio);
      const padX = (targetSize - newW) / 2;
      const padY = (targetSize - newH) / 2;

      // Ultralytics 默认填充灰度值 114
      ctx.fillStyle = 'rgb(114, 114, 114)';
      ctx.fillRect(0, 0, targetSize, targetSize);
      ctx.drawImage(source, padX, padY, newW, newH);
    } else {
      ctx.drawImage(source, 0, 0, targetSize, targetSize);
    }

    const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;
    const float32Data = new Float32Array(channelNum * targetSize * targetSize);

    // ImageNet 标准化参数 (RGB 顺序)
    const meanRGB = [0.485, 0.456, 0.406];
    const stdRGB = [0.229, 0.224, 0.225];

    // ===== 修复2：RGB 通道顺序（与 PyTorch 训练输入一致） =====
    for (let i = 0; i < targetSize * targetSize; i++) {
      const offset = i * 4;
      const r = imgData[offset] / 255.0;
      const g = imgData[offset + 1] / 255.0;
      const b = imgData[offset + 2] / 255.0;

      if (channelNum === 3) {
        if (normalization === 'imagenet') {
          float32Data[i * 3]     = (r - meanRGB[0]) / stdRGB[0];
          float32Data[i * 3 + 1] = (g - meanRGB[1]) / stdRGB[1];
          float32Data[i * 3 + 2] = (b - meanRGB[2]) / stdRGB[2];
        } else {
          // 默认：仅除以 255，与 Ultralytics 训练一致
          float32Data[i * 3]     = r;
          float32Data[i * 3 + 1] = g;
          float32Data[i * 3 + 2] = b;
        }
      } else {
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        float32Data[i] = gray;
      }
    }

    // 构建 NCHW 格式输入张量
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor('float32', float32Data, [1, channelNum, targetSize, targetSize]);
    
    // 运行推理
    const results = await session.run({ [inputName]: tensor });
    const outputName = session.outputNames[0];
    const outputTensor = results[outputName];
    const outputData = outputTensor.data as Float32Array;
    const dims = outputTensor.dims; // [1, 11, 2100]
    const labelsArray = labels.split(',').map(label => label.trim());
    const numClasses = labelsArray.length;
    const numPredictions = dims[2];

    const confThreshold = 0.25;
    let bestResult = { score: -1, label: '', boxIndex: -1 };

    // ===== 修复3：标准 YOLO 后处理 + Sigmoid 激活 =====
    // 输出格式：[1, 4 + numClasses, numPredictions]
    // 0~3: 边界框坐标，4~10: 7类情绪 logits
    for (let i = 0; i < numPredictions; i++) {
      let maxClassScore = -Infinity;
      let classId = -1;

      // 遍历7个情绪类别
      for (let cIdx = 0; cIdx < numClasses; cIdx++) {
        const classScoreIndex = (4 + cIdx) * numPredictions + i;
        const rawLogit = outputData[classScoreIndex];
        // Sigmoid 激活：将 logits 转为 0~1 概率
        const score = 1 / (1 + Math.exp(-rawLogit));

        if (score > maxClassScore) {
          maxClassScore = score;
          classId = cIdx;
        }
      }

      // 只保留置信度达标的锚框，过滤背景干扰
      if (maxClassScore > confThreshold && maxClassScore > bestResult.score) {
        bestResult.score = maxClassScore;
        bestResult.label = labelsArray[classId];
        bestResult.boxIndex = i;
      }
    }

    // 调试日志：打印最优结果的各类别详细分数
    if (bestResult.boxIndex >= 0) {
      console.log("=== 最优锚框各类别分数 ===");
      labelsArray.forEach((label, idx) => {
        const raw = outputData[(4 + idx) * numPredictions + bestResult.boxIndex];
        const prob = 1 / (1 + Math.exp(-raw));
        console.log(`${label}: ${prob.toFixed(4)} (raw: ${raw.toFixed(2)})`);
      });
    }
    console.log(`[Debug] Best Result: ${bestResult.label}, Score: ${bestResult.score.toFixed(4)}`);

    if (bestResult.score > confThreshold) {
      return `${bestResult.label} (${(bestResult.score * 100).toFixed(1)}%)`;
    } else {
      return "未识别出情绪";
    }
  };
  
  const runImagePrediction = async () => {
    if (!session || !imageRef.current) return;

    setIsPredicting(true);
    setErrorMessage('');
    setPrediction(null);

    try {
      const result = await performInference(imageRef.current);
      setPrediction(result);
    } catch (err: any) {
      console.error(err);
      setErrorMessage('预测失败: ' + (err.message || '未知错误'));
      setModelStatus('error');
    } finally {
      setIsPredicting(false);
    }
  };

  const startCamera = async () => {
    setErrorMessage('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraRunning(true);
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(console.error);
        };
      }
    } catch (err: any) {
      setErrorMessage('无法访问摄像头，请检查您的浏览器权限。');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraRunning(false);
    isProcessingRef.current = false;
  };

  const handleModeChange = (newMode: 'image' | 'camera') => {
    if (newMode === 'image') {
      stopCamera();
    }
    setMode(newMode);
    setPrediction(null);
    setErrorMessage('');
  };

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <header className="mb-8">
        <h2 className="text-3xl font-bold text-sky-900">人脸情绪识别 (基于 PyTorch)</h2>
        <p className="text-sky-600/80 mt-2">支持上传您自己的 ONNX 模型，分析静态图片或调用摄像头进行实时面部情绪分析</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-sky-50 space-y-6">
          <div className="flex items-center gap-3 border-b border-sky-50 pb-4">
            <Settings className="w-6 h-6 text-sky-600" />
            <h3 className="text-xl font-semibold text-sky-900">1. 模型配置 (ONNX)</h3>
          </div>

          <div className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-sky-800 mb-2">
                上传 ONNX 模型文件 (.onnx)
              </label>
              <label className="cursor-pointer bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-200 border-dashed px-5 py-6 rounded-2xl flex flex-col items-center justify-center gap-3 transition-colors">
                <UploadCloud className="w-6 h-6 text-sky-400" />
                <span className="font-medium text-sm">选择 .onnx 文件...</span>
                <input type="file" accept=".onnx" className="hidden" onChange={handleModelUpload} />
              </label>
            </div>

            {modelStatus === 'loading' && (
              <div className="flex items-center gap-2 text-sky-600 text-sm bg-sky-50 p-3 rounded-xl border border-sky-100">
                <Loader2 className="w-4 h-4 animate-spin" /> 模型加载中...
              </div>
            )}
            {modelStatus === 'ready' && (
              <div className="flex items-center gap-2 text-teal-600 text-sm bg-teal-50 p-3 rounded-xl border border-teal-100">
                <CheckCircle2 className="w-5 h-5" /> 模型加载成功，可以开始识别
              </div>
            )}
            {modelStatus === 'error' && (
              <div className="flex items-start gap-2 text-rose-600 text-sm bg-rose-50 p-3 rounded-xl border border-rose-100">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                <span>{errorMessage || '模型加载失败，请检查控制台或手动上传模型。'}</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-sky-800 mb-1">输入分辨率</label>
                <input
                  type="number"
                  value={inputSize}
                  onChange={(e) => setInputSize(e.target.value)}
                  className="w-full bg-white border border-sky-100 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
                  placeholder="320"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-sky-800 mb-1">图片通道</label>
                <select
                  value={channels}
                  onChange={(e) => setChannels(e.target.value as '3' | '1')}
                  className="w-full bg-white border border-sky-100 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
                >
                  <option value="3">RGB (3通道)</option>
                  <option value="1">灰度图 (1通道)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-sky-800 mb-1">归一化方式</label>
              <select
                value={normalization}
                onChange={(e) => setNormalization(e.target.value as 'imagenet' | 'none')}
                className="w-full bg-white border border-sky-100 rounded-xl px-3 py-2 text-sm focus:outline-none focus:border-sky-500"
              >
                <option value="none">仅缩放到 0 ~ 1（Ultralytics 默认）</option>
                <option value="imagenet">ImageNet 标准化</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-sky-800 mb-2">
                分类标签 (按输出节点顺序，中文逗号分隔)
              </label>
              <input
                type="text"
                value={labels}
                onChange={(e) => setLabels(e.target.value)}
                className="w-full bg-white border border-sky-100 rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 shadow-sm text-sm"
              />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-3xl shadow-sm border border-sky-50 space-y-6">
          <div className="flex items-center justify-between border-b border-sky-50 pb-4">
            <div className="flex items-center gap-3">
              <Smile className="w-6 h-6 text-sky-600" />
              <h3 className="text-xl font-semibold text-sky-900">2. 进行情绪识别</h3>
            </div>
            <div className="flex bg-sky-50 p-1 rounded-xl">
              <button
                onClick={() => handleModeChange('image')}
                className={clsx(
                  "px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200", 
                  mode === 'image' ? 'bg-white text-sky-800 shadow-sm' : 'text-sky-600 hover:text-sky-800'
                )}
              >
                图片上传
              </button>
              <button
                onClick={() => handleModeChange('camera')}
                className={clsx(
                  "px-4 py-1.5 rounded-lg text-sm font-medium transition-all duration-200", 
                  mode === 'camera' ? 'bg-white text-sky-800 shadow-sm' : 'text-sky-600 hover:text-sky-800'
                )}
              >
                实时摄像头
              </button>
            </div>
          </div>

          <div className="space-y-6">
            {mode === 'image' && (
              <>
                <label className={`cursor-pointer border border-sky-200 border-dashed rounded-2xl flex flex-col items-center justify-center gap-3 transition-colors overflow-hidden relative ${imageSrc ? 'bg-black/5' : 'bg-sky-50 hover:bg-sky-100 py-12'}`}>
                  {imageSrc ? (
                    <img ref={imageRef} src={imageSrc} alt="Upload preview" className="max-h-64 object-contain" crossOrigin="anonymous" />
                  ) : (
                    <>
                      <ImageIcon className="w-8 h-8 text-sky-400" />
                      <span className="font-medium text-sky-700">点击上传人脸图片</span>
                    </>
                  )}
                  <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </label>

                <button
                  onClick={runImagePrediction}
                  disabled={!session || !imageSrc || isPredicting}
                  className="w-full bg-sky-600 hover:bg-sky-700 disabled:bg-sky-300 text-white py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2 shadow-sm"
                >
                  {isPredicting ? (
                    <><Loader2 className="w-5 h-5 animate-spin" /> 正在推理分析...</>
                  ) : (
                    '开始分析情绪'
                  )}
                </button>
              </>
            )}

            {mode === 'camera' && (
              <>
                <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-800 aspect-video flex flex-col items-center justify-center shadow-inner">
                  <video 
                    ref={videoRef} 
                    className="w-full h-full object-cover" 
                    playsInline 
                    autoPlay 
                    muted 
                    style={{ transform: 'scaleX(-1)' }}
                  />
                  {!isCameraRunning && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80 bg-slate-900/50 backdrop-blur-sm z-10">
                      <Camera className="w-12 h-12 mb-3 text-sky-400 opacity-90" />
                      <p className="font-medium tracking-wide">摄像头已关闭</p>
                    </div>
                  )}
                  {isCameraRunning && !session && (
                    <div className="absolute top-4 left-4 flex items-center gap-2 text-xs font-medium text-white bg-black/50 px-3 py-1.5 rounded-full backdrop-blur-md">
                      <AlertCircle className="w-4 h-4 text-amber-400" />
                      请先上传模型开始分析
                    </div>
                  )}
                </div>
                
                <button
                  onClick={isCameraRunning ? stopCamera : startCamera}
                  className={clsx(
                    "w-full py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2 shadow-sm",
                    isCameraRunning ? "bg-rose-100 text-rose-700 hover:bg-rose-200 border border-rose-200" : "bg-sky-600 text-white hover:bg-sky-700"
                  )}
                >
                  {isCameraRunning ? <><StopCircle className="w-5 h-5"/> 停止识别</> : <><Video className="w-5 h-5"/> 开启摄像头</>}
                </button>
              </>
            )}

            {prediction && (
              <div className="bg-sky-50 border border-sky-100 rounded-2xl p-6 text-center space-y-2 animate-in zoom-in-95 duration-300">
                <div className="text-sm text-sky-600 font-medium tracking-wide">AI 识别结果</div>
                <div className="text-3xl font-bold text-sky-900">{prediction}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}