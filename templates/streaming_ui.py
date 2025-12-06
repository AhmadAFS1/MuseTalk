from fastapi.responses import HTMLResponse

def get_streaming_ui_html() -> str:
    """Returns the HTML content for the MSE streaming interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>MuseTalk MSE Streaming</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        @media (max-width: 968px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        .panel {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        
        .panel h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="number"]:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        input[type="file"]:hover {
            background: #f0f4ff;
        }
        
        .batch-info {
            margin-top: 8px;
            padding: 10px;
            background: #e0e7ff;
            border-radius: 6px;
            font-size: 0.85rem;
            line-height: 1.6;
        }
        
        .batch-info strong {
            color: #667eea;
            display: block;
            margin-bottom: 4px;
        }
        
        .batch-info .metric {
            display: flex;
            justify-content: space-between;
            margin: 3px 0;
        }
        
        .batch-info .label {
            color: #666;
        }
        
        .batch-info .value {
            color: #333;
            font-weight: 600;
        }
        
        .batch-info .value.good {
            color: #4CAF50;
        }
        
        .batch-info .value.warn {
            color: #FF9800;
        }
        
        .batch-options {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        
        .batch-option {
            position: relative;
        }
        
        .batch-option input[type="radio"] {
            display: none;
        }
        
        .batch-option label {
            display: block;
            padding: 15px 10px;
            text-align: center;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 0.9rem;
        }
        
        .batch-option label:hover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .batch-option input[type="radio"]:checked + label {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
            font-weight: 700;
        }
        
        .batch-option .batch-label {
            display: block;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }
        
        .batch-option .batch-meta {
            display: block;
            font-size: 0.7rem;
            opacity: 0.8;
        }
        
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .status-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }
        
        .status-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .time-stats {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 0.9rem;
        }
        
        .time-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .time-stat-label {
            color: #666;
            font-size: 0.8rem;
            margin-bottom: 4px;
        }
        
        .time-stat-value {
            color: #667eea;
            font-weight: 700;
            font-size: 1.2rem;
            font-family: 'Courier New', monospace;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .chunk-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .chunk-item {
            background: #f8f9fa;
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .chunk-item a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 5px;
            transition: background 0.2s;
        }
        
        .chunk-item a:hover {
            background: #e0e7ff;
        }
        
        #videoPlayer {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            background: #000;
            display: block;
        }
        
        .hidden {
            display: none;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 MuseTalk MSE Streaming</h1>
            <p>Zero-buffering, zero-flicker real-time video generation</p>
        </div>
        
        <div class="content">
            <!-- Left Panel -->
            <div class="panel">
                <h2>📤 Upload & Configure</h2>
                
                <form id="uploadForm">
                    <div class="form-group">
                        <label for="avatar_id">🎭 Avatar ID</label>
                        <input type="text" id="avatar_id" value="test_avatar" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="audio_file">🎵 Audio File</label>
                        <input type="file" id="audio_file" accept="audio/*,.mpga,.wav,.mp3" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="fps">🎞️ Frame Rate (FPS)</label>
                        <select id="fps">
                        <option value="10" selected>10 fps (Fastest)</option>
                            <option value="15" selected>15 fps (Fast)</option>
                            <option value="20">20 fps (Balanced)</option>
                            <option value="25">25 fps (High Quality)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="chunk_duration">⏱️ Chunk Duration (seconds)</label>
                        <input type="number" id="chunk_duration" value="1" min="0.5" max="3" step="0.5">
                    </div>
                    
                    <div class="form-group">
                        <label>🔢 Batch Size (GPU Memory vs Concurrency)</label>
                        <div class="batch-options">
                            <div class="batch-option">
                                <input type="radio" name="batch_size" id="batch_1" value="1">
                                <label for="batch_1">
                                    <span class="batch-label">1</span>
                                    <span class="batch-meta">2GB<br>9 users</span>
                                </label>
                            </div>
                            <div class="batch-option">
                                <input type="radio" name="batch_size" id="batch_2" value="2" checked>
                                <label for="batch_2">
                                    <span class="batch-label">2</span>
                                    <span class="batch-meta">3.5GB<br>5 users</span>
                                </label>
                            </div>
                            <div class="batch-option">
                                <input type="radio" name="batch_size" id="batch_4" value="4">
                                <label for="batch_4">
                                    <span class="batch-label">4</span>
                                    <span class="batch-meta">6GB<br>3 users</span>
                                </label>
                            </div>
                            <div class="batch-option">
                                <input type="radio" name="batch_size" id="batch_8" value="8">
                                <label for="batch_8">
                                    <span class="batch-label">8</span>
                                    <span class="batch-meta">10GB<br>1 user</span>
                                </label>
                            </div>
                        </div>
                        <div class="batch-info" id="batchInfo">
                            <strong>📊 Batch Size 2 (Recommended)</strong>
                            <div class="metric">
                                <span class="label">VRAM per request:</span>
                                <span class="value">3.5 GB</span>
                            </div>
                            <div class="metric">
                                <span class="label">Max concurrent users:</span>
                                <span class="value good">5 streams</span>
                            </div>
                            <div class="metric">
                                <span class="label">Speed per chunk (2s):</span>
                                <span class="value">~2.1 seconds</span>
                            </div>
                            <div class="metric">
                                <span class="label">Best for:</span>
                                <span class="value">Balanced speed & capacity</span>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn" id="submitBtn">
                        🚀 Start Generation
                    </button>
                </form>
            </div>
            
            <!-- Right Panel -->
            <div class="panel">
                <h2>📊 Progress & Preview</h2>
                
                <div class="status-box">
                    <h3 id="statusText">Ready to start</h3>
                    
                    <div class="time-stats" id="timeStats" style="display: none;">
                        <div class="time-stat">
                            <span class="time-stat-label">⏱️ Elapsed</span>
                            <span class="time-stat-value" id="elapsedTime">00:00</span>
                        </div>
                        <div class="time-stat">
                            <span class="time-stat-label">⚡ Speed</span>
                            <span class="time-stat-value" id="processingSpeed">--</span>
                        </div>
                        <div class="time-stat">
                            <span class="time-stat-label">🎯 ETA</span>
                            <span class="time-stat-value" id="estimatedTime">--</span>
                        </div>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress" style="width: 0%">0%</div>
                    </div>
                    <p id="statusDetail">Upload an audio file to begin</p>
                </div>
                
                <div id="videoSection" class="hidden">
                    <h3 style="margin-bottom: 10px;">🎥 Live Preview</h3>
                    <video id="videoPlayer" controls></video>
                    <p id="videoInfo" style="margin-top: 10px; text-align: center; color: #666;">Waiting for chunks...</p>
                </div>
                
                <div id="chunksSection" class="hidden">
                    <h3 style="margin-bottom: 10px;">📦 Generated Chunks</h3>
                    <div class="chunk-list" id="chunkList"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Batch size configurations
        const batchConfigs = {
            1: {
                vram: '2.0 GB',
                maxConcurrent: '9 streams',
                speed: '~3.3 seconds',
                chunkSpeed: 110,
                bestFor: 'Maximum concurrency (mobile apps)',
                recommendation: '✅ Best for many users'
            },
            2: {
                vram: '3.5 GB',
                maxConcurrent: '5 streams',
                speed: '~2.1 seconds',
                chunkSpeed: 70,
                bestFor: 'Balanced speed & capacity',
                recommendation: '✅ Recommended default'
            },
            4: {
                vram: '6.0 GB',
                maxConcurrent: '3 streams',
                speed: '~1.4 seconds',
                chunkSpeed: 45,
                bestFor: 'Faster generation',
                recommendation: '⚡ Very fast per-stream'
            },
            8: {
                vram: '10.0 GB',
                maxConcurrent: '1 stream',
                speed: '~0.9 seconds',
                chunkSpeed: 30,
                bestFor: 'Maximum speed (single user)',
                recommendation: '⚠️ Only 1 concurrent user'
            }
        };
        
        // Update batch info when selection changes
        document.querySelectorAll('input[name="batch_size"]').forEach(function(radio) {
            radio.addEventListener('change', function() {
                updateBatchInfo(this.value);
            });
        });
        
        function updateBatchInfo(batchSize) {
            const config = batchConfigs[batchSize];
            const batchInfo = document.getElementById('batchInfo');
            
            const concurrentClass = parseInt(batchSize) <= 2 ? 'good' : 'warn';
            
            batchInfo.innerHTML = `
                <strong>📊 Batch Size ${batchSize} ${config.recommendation ? '- ' + config.recommendation : ''}</strong>
                <div class="metric">
                    <span class="label">VRAM per request:</span>
                    <span class="value">${config.vram}</span>
                </div>
                <div class="metric">
                    <span class="label">Max concurrent users:</span>
                    <span class="value ${concurrentClass}">${config.maxConcurrent}</span>
                </div>
                <div class="metric">
                    <span class="label">Speed per chunk (2s):</span>
                    <span class="value">${config.speed}</span>
                </div>
                <div class="metric">
                    <span class="label">Best for:</span>
                    <span class="value">${config.bestFor}</span>
                </div>
            `;
        }
        
        let state = {
            chunks: [],
            requestId: '',
            totalChunks: 0,
            startTime: null,
            elapsedTimer: null,
            fps: 15
        };
        
        let mediaSource = null;
        let sourceBuffer = null;
        let pendingChunks = [];
        let isAppending = false;
        let isFirstChunk = true;
        
        const form = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const statusText = document.getElementById('statusText');
        const statusDetail = document.getElementById('statusDetail');
        const progress = document.getElementById('progress');
        const chunkList = document.getElementById('chunkList');
        const chunksSection = document.getElementById('chunksSection');
        const videoSection = document.getElementById('videoSection');
        const videoPlayer = document.getElementById('videoPlayer');
        const videoInfo = document.getElementById('videoInfo');
        const timeStats = document.getElementById('timeStats');
        const elapsedTime = document.getElementById('elapsedTime');
        const processingSpeed = document.getElementById('processingSpeed');
        const estimatedTime = document.getElementById('estimatedTime');
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
        }
        
        function updateElapsedTime() {
            if (!state.startTime) return;
            
            const elapsed = (Date.now() - state.startTime) / 1000;
            elapsedTime.textContent = formatTime(elapsed);
            
            if (state.chunks.length > 0) {
                const speed = state.chunks.length / elapsed;
                processingSpeed.textContent = speed.toFixed(2) + ' c/s';
                
                if (state.totalChunks > 0 && speed > 0) {
                    const remaining = state.totalChunks - state.chunks.length;
                    const eta = remaining / speed;
                    estimatedTime.textContent = formatTime(eta);
                }
            }
        }
        
        function startTimer() {
            state.startTime = Date.now();
            timeStats.style.display = 'flex';
            state.elapsedTimer = setInterval(updateElapsedTime, 1000);
        }
        
        function stopTimer() {
            if (state.elapsedTimer) {
                clearInterval(state.elapsedTimer);
                state.elapsedTimer = null;
            }
        }
        
        function initMediaSource() {
            mediaSource = new MediaSource();
            videoPlayer.src = URL.createObjectURL(mediaSource);
            
            mediaSource.addEventListener('sourceopen', function() {
                console.log('✅ MediaSource opened');
                
                try {
                    sourceBuffer = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
                    sourceBuffer.mode = 'sequence';
                    
                    sourceBuffer.addEventListener('updateend', function() {
                        isAppending = false;
                        processNextChunk();
                    });
                    
                    sourceBuffer.addEventListener('error', function(e) {
                        console.error('❌ SourceBuffer error:', e);
                        statusText.textContent = '❌ SourceBuffer error';
                    });
                    
                    console.log('✅ SourceBuffer created (mode=sequence)');
                } catch (e) {
                    console.error('❌ Failed to create SourceBuffer:', e);
                    statusText.textContent = '❌ ' + e.message;
                }
            });
            
            mediaSource.addEventListener('sourceended', function() {
                console.log('✅ MediaSource ended');
            });
        }
        
        function processNextChunk() {
            if (isAppending || pendingChunks.length === 0 || !sourceBuffer) {
                return;
            }
            
            isAppending = true;
            const chunk = pendingChunks.shift();
            
            try {
                sourceBuffer.appendBuffer(chunk);
                
                if (isFirstChunk) {
                    isFirstChunk = false;
                    videoPlayer.play().then(function() {
                        console.log('▶️ Playback started');
                        videoInfo.textContent = 'Playing...';
                    }).catch(function(e) {
                        console.warn('⚠️ Autoplay blocked:', e);
                        videoInfo.textContent = 'Click to play';
                    });
                }
            } catch (e) {
                console.error('❌ Append error:', e);
                isAppending = false;
            }
        }
        
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            state = { 
                chunks: [], 
                requestId: '', 
                totalChunks: 0, 
                startTime: null,
                elapsedTimer: null,
                fps: parseInt(document.getElementById('fps').value)
            };
            
            pendingChunks = [];
            isAppending = false;
            isFirstChunk = true;
            
            chunkList.innerHTML = '';
            chunksSection.classList.add('hidden');
            videoSection.classList.add('hidden');
            timeStats.style.display = 'none';
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner"></span>Processing...';
            statusText.textContent = 'Uploading audio...';
            progress.style.width = '0%';
            progress.textContent = '0%';
            
            initMediaSource();
            
            const formData = new FormData();
            formData.append('audio_file', document.getElementById('audio_file').files[0]);
            
            const batchSize = document.querySelector('input[name="batch_size"]:checked').value;
            
            const params = new URLSearchParams({
                avatar_id: document.getElementById('avatar_id').value,
                batch_size: batchSize,
                fps: state.fps,
                chunk_duration: document.getElementById('chunk_duration').value
            });
            
            try {
                const response = await fetch('/generate/stream?' + params.toString(), {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status + ': ' + response.statusText);
                }
                
                startTimer();
                statusText.textContent = 'Generating video chunks (batch_size=' + batchSize + ')...';
                chunksSection.classList.remove('hidden');
                videoSection.classList.remove('hidden');
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                while (true) {
                    const result = await reader.read();
                    if (result.done) {
                        console.log('✅ Stream complete');
                        
                        if (!sourceBuffer.updating && pendingChunks.length === 0) {
                            mediaSource.endOfStream();
                        }
                        break;
                    }
                    
                    buffer += decoder.decode(result.value, { stream: true });
                    
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop() || '';
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                await handleSSEEvent(data);
                            } catch (e) {
                                console.warn('Failed to parse SSE:', line, e);
                            }
                        }
                    }
                }
            } catch (error) {
                stopTimer();
                statusText.textContent = '❌ Error occurred';
                statusDetail.textContent = error.message;
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Start Generation';
            }
        });
        
        async function handleSSEEvent(data) {
            if (data.event === 'chunk') {
                state.chunks.push(data.url);
                state.requestId = data.url.split('/')[2];
                state.totalChunks = data.total_chunks || state.chunks.length;
                
                const percent = Math.round((state.chunks.length / state.totalChunks) * 100);
                progress.style.width = percent + '%';
                progress.textContent = percent + '%';
                
                statusText.textContent = 'Chunk ' + state.chunks.length + ' of ' + state.totalChunks;
                statusDetail.textContent = 'Processing... ' + state.chunks.length + '/' + state.totalChunks + ' chunks ready';
                
                const chunkDiv = document.createElement('div');
                chunkDiv.className = 'chunk-item';
                chunkDiv.innerHTML = '<span>✅ Chunk ' + (data.index + 1) + '</span>' +
                    '<a href="' + data.url + '" target="_blank">Download</a>';
                chunkList.appendChild(chunkDiv);
                
                console.log('📥 Fetching chunk: ' + data.url);
                const response = await fetch(data.url);
                const arrayBuffer = await response.arrayBuffer();
                
                pendingChunks.push(arrayBuffer);
                processNextChunk();
            }
            else if (data.event === 'complete') {
                stopTimer();
                
                const totalTime = (Date.now() - state.startTime) / 1000;
                statusText.textContent = '✅ Generation Complete!';
                statusDetail.textContent = 'All ' + state.chunks.length + ' chunks generated in ' + formatTime(totalTime);
                progress.style.width = '100%';
                progress.textContent = '100%';
                
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Start Generation';
            }
            else if (data.event === 'error') {
                stopTimer();
                statusText.textContent = '❌ Error';
                statusDetail.textContent = data.message;
                submitBtn.disabled = false;
                submitBtn.textContent = '🚀 Start Generation';
            }
        }
    </script>
</body>
</html>
    """


async def streaming_ui_endpoint():
    """FastAPI endpoint for streaming UI"""
    return HTMLResponse(content=get_streaming_ui_html())