import SwiftUI

@main
struct DictateDemoApp: App {
    @State private var viewModel = DictateViewModel()

    var body: some Scene {
        MenuBarExtra {
            DictateMenuView(viewModel: viewModel)
                .task {
                    // Load models on first menu open
                    if !viewModel.modelLoaded && !viewModel.isLoading {
                        await viewModel.loadModels()
                    }
                }
        } label: {
            HStack(spacing: 2) {
                Image(systemName: viewModel.isRecording ? "mic.fill" : "mic")
                if viewModel.isLoading {
                    Text("...")
                        .font(.caption2)
                }
            }
        }
        .menuBarExtraStyle(.window)

        WindowGroup("Dictate", id: "dictate-hud") {
            DictateHUDView(viewModel: viewModel)
                .frame(minWidth: 400, minHeight: 200)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: 450, height: 300)
        .defaultPosition(.topTrailing)
    }
}
