import AVFoundation
import Foundation

/// Any audio file AVFoundation reads (WAV, m4a, mp3, …) → mono float32 at 16 kHz, the log-mel model's input.
/// OmniVoice and Kokoro TTS write 24 kHz WAVs; this is the resample the server does with librosa.
public func loadWaveform16k(url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let srcFormat = file.processingFormat
    guard let src = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
        throw BundleError.io("cannot allocate a buffer for \(url.lastPathComponent)")
    }
    try file.read(into: src)
    if srcFormat.sampleRate == 16000, srcFormat.channelCount == 1, srcFormat.commonFormat == .pcmFormatFloat32,
       let ch = src.floatChannelData {
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(src.frameLength)))
    }
    guard let dstFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
          let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
        throw BundleError.format("cannot convert \(url.lastPathComponent) (\(srcFormat)) to 16 kHz mono")
    }
    let capacity = AVAudioFrameCount(Double(src.frameLength) * 16000.0 / srcFormat.sampleRate) + 4096
    guard let dst = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: capacity) else {
        throw BundleError.io("cannot allocate the 16 kHz buffer")
    }
    var handedOver = false
    var convError: NSError? = nil
    let status = converter.convert(to: dst, error: &convError) { _, outStatus in
        if handedOver { outStatus.pointee = .endOfStream; return nil }
        handedOver = true
        outStatus.pointee = .haveData
        return src
    }
    if status == .error { throw convError ?? BundleError.format("audio conversion failed for \(url.lastPathComponent)") }
    guard let ch = dst.floatChannelData else { throw BundleError.io("no channel data after conversion") }
    return Array(UnsafeBufferPointer(start: ch[0], count: Int(dst.frameLength)))
}
