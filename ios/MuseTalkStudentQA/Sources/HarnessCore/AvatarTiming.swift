import Foundation

/// The source animation clock and inference/display clock are independent. Reusing a 24 fps source frame at
/// 25 fps does not reuse the generated face: each render tick still receives its own audio window and inference.
public struct AvatarTimebase {
    public let sourceFPS: Int
    public let renderFPS: Int

    public init(sourceFPS: Int, renderFPS: Int) throws {
        guard (1...60).contains(sourceFPS), (1...60).contains(renderFPS) else {
            throw BundleError.format("source/render fps must be between 1 and 60")
        }
        self.sourceFPS = sourceFPS
        self.renderFPS = renderFPS
    }

    /// Display ticks covering a source interval. Durations not divisible by a display tick round up by <1 tick.
    public func renderFrameCount(sourceFrames: Int) -> Int {
        max(1, (sourceFrames * renderFPS + sourceFPS - 1) / sourceFPS)
    }

    /// Sample animation by elapsed render time. 240 source frames at 24 fps wrap after 250 ticks at 25 fps.
    public func sourceCycle(renderFrame: Int, sourceFrames: Int) -> Int {
        let frame = max(0, renderFrame)
        let source = (frame / renderFPS) * sourceFPS + ((frame % renderFPS) * sourceFPS) / renderFPS
        return source % max(1, sourceFrames)
    }
}
