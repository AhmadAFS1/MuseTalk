# 📱 Mobile Integration Guide (Session-Based)

## Overview

**New approach:** Session-based API - no `postMessage` needed!

---

## 🚀 Simple 3-Step Flow

### **Step 1: Create Session** (when user opens chat)

```swift
// iOS Example
func openAvatarChat(avatarId: String, userId: String) {
    let url = URL(string: "https://your-server.com/sessions/create?avatar_id=\(avatarId)&user_id=\(userId)")!
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        guard let data = data else { return }
        
        let session = try? JSONDecoder().decode(SessionResponse.self, from: data)
        
        DispatchQueue.main.async {
            self.loadPlayer(url: session.player_url)
            self.sessionId = session.session_id
        }
    }.resume()
}
```

### **Step 2: Load Player in WebView**

```swift
func loadPlayer(url: String) {
    let webView = WKWebView(frame: view.bounds)
    webView.load(URLRequest(url: URL(string: url)!))
    view.addSubview(webView)
}
```

### **Step 3: Send Audio**

```swift
func sendAudio(audioData: Data) {
    let url = URL(string: "https://your-server.com/sessions/\(sessionId)/stream")!
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    
    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"audio_file\"; filename=\"audio.mp3\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: audio/mpeg\r\n\r\n".data(using: .utf8)!)
    body.append(audioData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    
    request.httpBody = body
    
    URLSession.shared.dataTask(with: request).resume()
}
```

---

## 🎯 That's It!

- ✅ No `postMessage` complexity
- ✅ WebView auto-receives chunks via SSE
- ✅ Fully isolated sessions per user
- ✅ Automatic cleanup after 1 hour

---

## 🧪 Testing Multi-User

**Terminal 1: User 1**
```bash
# Create session
curl -X POST "http://localhost:8000/sessions/create?avatar_id=test_avatar&user_id=user1"
# Returns: {"session_id": "abc123", "player_url": "/player/session/abc123"}

# Send audio
curl -X POST "http://localhost:8000/sessions/abc123/stream" \
  -F "audio_file=@audio1.mp3"
```

**Terminal 2: User 2** (concurrent)
```bash
curl -X POST "http://localhost:8000/sessions/create?avatar_id=test_avatar&user_id=user2"
# Returns: {"session_id": "def456", ...}

curl -X POST "http://localhost:8000/sessions/def456/stream" \
  -F "audio_file=@audio2.mp3"
```

Both users get independent streams!

---

## 📊 Session Management

**Check session status:**
```bash
curl http://localhost:8000/sessions/abc123/status
```

**Delete session (when user closes chat):**
```bash
curl -X DELETE http://localhost:8000/sessions/abc123
```

**View all sessions:**
```bash
curl http://localhost:8000/sessions/stats
```

---

## 🔒 Multi-User Isolation

Each session has:
- ✅ Unique `session_id`
- ✅ Separate chunk queue
- ✅ Independent streaming state
- ✅ Avatar config (batch_size, fps)

Server handles up to **5 concurrent streams** automatically!