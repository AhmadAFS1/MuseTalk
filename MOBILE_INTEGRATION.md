# 📱 Mobile App Integration Guide

## Overview

The mobile player (`/player/mobile`) is designed to be embedded in iOS/Android apps via WebView with multi-user session isolation.

---

## 🔑 Key Features

- **Session Isolation**: Each user gets unique `session_id` for separate streaming
- **Zero UI Clutter**: Just the video player (no controls, no branding)
- **Mobile-Optimized**: Auto-play, `playsinline`, touch-friendly
- **Bi-directional Communication**: WebView ↔ Native App via `postMessage`

---

## 🚀 Integration Steps

### 1. **Embed WebView in Native App**

**iOS (Swift)**
```swift
import WebKit

class AvatarViewController: UIViewController, WKScriptMessageHandler {
    var webView: WKWebView!
    var sessionId: String = UUID().uuidString
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let contentController = WKUserContentController()
        contentController.add(self, name: "nativeApp")
        config.userContentController = contentController
        
        webView = WKWebView(frame: view.bounds, configuration: config)
        view.addSubview(webView)
        
        let url = URL(string: "https://your-server.com/player/mobile?session_id=\(sessionId)")!
        webView.load(URLRequest(url: url))
    }
    
    // Receive messages from WebView
    func userContentController(_ userContentController: WKUserContentController, 
                              didReceive message: WKScriptMessage) {
        if let dict = message.body as? [String: Any],
           let type = dict["type"] as? String {
            
            if type == "PLAYER_READY" {
                print("✅ Player ready")
                sendAudioToPlayer()
            }
        }
    }
    
    func sendAudioToPlayer() {
        guard let audioData = getAudioData() else { return }
        
        let js = """
        window.postMessage({
            type: 'START_STREAM',
            avatar_id: 'test_avatar',
            batch_size: '2',
            fps: '15',
            chunk_duration: '2',
            audioBlob: new Blob([\(audioData)], {type: 'audio/mpeg'})
        }, '*');
        """
        
        webView.evaluateJavaScript(js)
    }
}
```

**Android (Kotlin)**
```kotlin
import android.webkit.WebView
import android.webkit.JavascriptInterface

class AvatarActivity : AppCompatActivity() {
    private lateinit var webView: WebView
    private val sessionId = UUID.randomUUID().toString()
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        webView = WebView(this)
        webView.settings.apply {
            javaScriptEnabled = true
            mediaPlaybackRequiresUserGesture = false
        }
        
        webView.addJavascriptInterface(NativeInterface(), "nativeApp")
        
        val url = "https://your-server.com/player/mobile?session_id=$sessionId"
        webView.loadUrl(url)
        
        setContentView(webView)
    }
    
    inner class NativeInterface {
        @JavascriptInterface
        fun onPlayerReady(sessionId: String) {
            runOnUiThread {
                sendAudioToPlayer()
            }
        }
    }
    
    private fun sendAudioToPlayer() {
        val audioData = getAudioData() // Your audio as Base64
        
        val js = """
        window.postMessage({
            type: 'START_STREAM',
            avatar_id: 'test_avatar',
            batch_size: '2',
            fps: '15',
            chunk_duration: '2',
            audioBlob: base64ToBlob('$audioData', 'audio/mpeg')
        }, '*');
        """
        
        webView.evaluateJavaScript(js, null)
    }
}
```

---

## 📡 Communication Protocol

### **From Native App → WebView**

```javascript
window.postMessage({
    type: 'START_STREAM',
    avatar_id: 'user_123_avatar',
    batch_size: '2',          // 1, 2, 4, or 8
    fps: '15',                // 10, 15, 20, or 25
    chunk_duration: '2',      // Seconds per chunk
    audioBlob: <Blob>         // Audio file as Blob
}, '*');
```

### **From WebView → Native App**

```javascript
window.parent.postMessage({
    type: 'PLAYER_READY',
    sessionId: 'abc123...'
}, '*');
```

---

## 🔒 Multi-User Session Management

### **Server-Side** (already implemented)

- Each request to `/generate/stream` creates isolated chunk directory:
  ```
  chunks/
    ├── user1_avatar_req_abc12345/
    │   ├── chunk_0000.mp4
    │   └── chunk_0001.mp4
    └── user2_avatar_req_def67890/
        ├── chunk_0000.mp4
        └── chunk_0001.mp4
  ```

### **Client-Side**

- Each WebView instance has unique `session_id` in URL
- Native app tracks sessions per user:
  ```swift
  var activeSessions: [String: WKWebView] = [:]
  
  func createPlayerForUser(userId: String) -> WKWebView {
      let sessionId = UUID().uuidString
      let webView = createWebView(sessionId: sessionId)
      activeSessions[userId] = webView
      return webView
  }
  ```

---

## ⚡ Performance Tips

1. **Batch Size Selection**
   - Mobile (3G/4G): `batch_size=1` (max concurrency)
   - WiFi: `batch_size=2` (balanced)
   - Single user: `batch_size=4` (fastest)

2. **FPS Optimization**
   - Mobile data: `fps=10` (lower bandwidth)
   - Normal: `fps=15` (recommended)
   - High quality: `fps=25` (smooth but heavy)

3. **Chunk Duration**
   - Fast network: `chunk_duration=1` (low latency)
   - Default: `chunk_duration=2` (balanced)
   - Slow network: `chunk_duration=3` (fewer requests)

---

## 🧪 Testing Multi-User

**Simulate 5 concurrent users:**

```bash
# Terminal 1
curl -X POST "http://localhost:8000/generate/stream?avatar_id=user1&batch_size=2" \
  -F "audio_file=@test1.mp3"

# Terminal 2
curl -X POST "http://localhost:8000/generate/stream?avatar_id=user2&batch_size=2" \
  -F "audio_file=@test2.mp3"

# ... up to 5 concurrent requests
```

**Check GPU memory allocation:**
```bash
curl http://localhost:8000/stats
```

---

## 🔧 Troubleshooting

### **Black screen on mobile**
- Check `playsinline` attribute
- Disable `mediaPlaybackRequiresUserGesture` in WebView settings

### **Audio not playing**
- iOS: Enable audio session in native app
- Android: Add `INTERNET` permission in manifest

### **Chunks not loading**
- Verify server is accessible from mobile network
- Check CORS headers in API responses

---

## 📊 Example: Full Flow

1. User opens app → Native code creates WebView with unique `session_id`
2. WebView loads `/player/mobile?session_id=abc123`
3. WebView sends `PLAYER_READY` → Native app
4. User records audio → Native app sends `START_STREAM` with audio blob
5. WebView calls `/generate/stream` → Server creates `chunks/abc123/`
6. Chunks stream to WebView → Auto-play via MSE
7. User gets instant playback (no buffering)

---

## 🚀 Production Checklist

- [ ] Use HTTPS for WebView URLs
- [ ] Implement session cleanup (delete chunks after TTL)
- [ ] Add retry logic for network failures
- [ ] Monitor concurrent user limits (adjust batch size)
- [ ] Add analytics for playback events
- [ ] Test on low-end devices (Android 6+, iOS 12+)

---

## 📞 Support

For integration help, check existing endpoints:
- Health: `GET /health`
- Stats: `GET /stats`
- Docs: `GET /docs`