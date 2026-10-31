import SwiftUI

struct LipsyncView: View {
    @ObservedObject var player: LipsyncPlayer

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Color.black
                if let img = player.image {
                    Image(decorative: img, scale: 1, orientation: .up)
                        .resizable()
                        .scaledToFit()
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "person.wave.2").font(.system(size: 44)).foregroundStyle(.secondary)
                        Text(player.hasAvatar ? "Latina · MuseTalk on the Neural Engine" : "no avatar in the bundle")
                            .foregroundStyle(.secondary)
                    }
                }
                if player.state == .loading || player.state == .preroll {
                    ProgressView().tint(.white).scaleEffect(1.5)
                }
            }
            .aspectRatio(512.0 / 832.0, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal)

            Text(player.status).font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center).padding(.horizontal)
            if !player.stats.isEmpty {
                Text(player.stats).font(.caption.monospaced()).multilineTextAlignment(.center).padding(.horizontal)
            }
            HStack {
                if player.state == .playing || player.state == .preroll || player.state == .loading {
                    Button(role: .destructive) { player.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                        .buttonStyle(.bordered)
                } else {
                    Button { player.start() } label: { Label("Play", systemImage: "play.fill") }
                        .buttonStyle(.borderedProminent)
                        .disabled(!player.hasAvatar)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.top)
        .navigationTitle("Lipsync")
    }
}
