// JLVisionCapture.swift
// JL Vision — iPhone Companion App
//
// Architecture: Option 3 — Video + AR metadata + AI results
//   ARKit camera + tracking  →  Apple Vision body pose (19 joints, no model download)
//   + LiDAR depth (iPhone 12/13/14/15 Pro only)  →  TCP JSON stream to coach workstation
//
// Requirements: iOS 16+, Xcode 15+
// Frameworks:   ARKit, Vision, Network, AVFoundation
// Entitlements: com.apple.developer.networking.multicast (if using Bonjour discovery)
//
// Usage:
//   1. Create a new iOS App target in Xcode
//   2. Add this file and set deployment target to iOS 16.0
//   3. Add NSCameraUsageDescription to Info.plist
//   4. Build and run on a physical iPhone (ARKit requires real hardware)
//   5. Enter the coach workstation's IP and tap Connect
//   6. The workstation receives pose frames on TCP port 8765

import UIKit
import ARKit
import Vision
import Network
import simd

// MARK: - Payload types

struct JLFrame: Codable {
    let timestamp: Double
    let frameId: Int
    let source: String
    let hasLiDAR: Bool
    let cameraPose: CameraPose
    var keypoints: [String: [Double]]   // name → [y_norm, x_norm, confidence]
    var depth: [String: Double]          // joint name → metres (LiDAR only)
}

struct CameraPose: Codable {
    let position: [Float]    // [x, y, z] world metres
    let rotation: [Float]    // quaternion [qx, qy, qz, qw]
}

// MARK: - ARKit + Vision session

class JLCaptureSession: NSObject, ARSessionDelegate {

    // Public state
    var onStatusChange: ((String) -> Void)?
    private(set) var isConnected = false

    // Internals
    private let arSession   = ARSession()
    private var connection: NWConnection?
    private var frameId     = 0
    private let sendQueue   = DispatchQueue(label: "jl.send", qos: .userInteractive)

    // Vision
    private let poseRequest = VNDetectHumanBodyPoseRequest()
    private let visionQueue = VNSequenceRequestHandler()

    // Apple Vision joint → MoveNet name (y coordinate is FLIPPED: Vision uses bottom-left origin)
    private let jointMap: [(VNHumanBodyPoseObservation.JointName, String)] = [
        (.nose,           "nose"),
        (.leftEye,        "left_eye"),    (.rightEye,        "right_eye"),
        (.leftEar,        "left_ear"),    (.rightEar,        "right_ear"),
        (.leftShoulder,   "left_shoulder"), (.rightShoulder, "right_shoulder"),
        (.leftElbow,      "left_elbow"),  (.rightElbow,      "right_elbow"),
        (.leftWrist,      "left_wrist"),  (.rightWrist,      "right_wrist"),
        (.leftHip,        "left_hip"),    (.rightHip,        "right_hip"),
        (.leftKnee,       "left_knee"),   (.rightKnee,       "right_knee"),
        (.leftAnkle,      "left_ankle"),  (.rightAnkle,      "right_ankle"),
    ]

    // MARK: Connect

    func connect(host: String, port: UInt16 = 8765) {
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host),
            port: NWEndpoint.Port(rawValue: port)!
        )
        let params = NWParameters.tcp
        params.serviceClass = .responsiveData
        connection = NWConnection(to: endpoint, using: params)
        connection?.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                switch state {
                case .ready:
                    self?.isConnected = true
                    self?.onStatusChange?("Connected")
                case .failed(let err):
                    self?.isConnected = false
                    self?.onStatusChange?("Error: \(err.localizedDescription)")
                case .cancelled:
                    self?.isConnected = false
                    self?.onStatusChange?("Disconnected")
                default:
                    self?.onStatusChange?("Connecting…")
                }
            }
        }
        connection?.start(queue: sendQueue)
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
    }

    // MARK: ARKit

    func startCapture() {
        let config = ARWorldTrackingConfiguration()
        // Person segmentation + depth (depth only on Pro LiDAR models)
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentationWithDepth) {
            config.frameSemantics = .personSegmentationWithDepth
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            config.frameSemantics = .personSegmentation
        }
        config.videoFormat = ARWorldTrackingConfiguration.supportedVideoFormats
            .first(where: { $0.framesPerSecond == 60 })
            ?? ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing
            ?? ARWorldTrackingConfiguration.supportedVideoFormats.first!
        arSession.delegate = self
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopCapture() {
        arSession.pause()
    }

    // MARK: ARSessionDelegate — called on every camera frame

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameId += 1
        // Process at ~30 fps regardless of capture rate
        guard frameId % 2 == 0 else { return }

        let pixelBuffer = frame.capturedImage
        let imgWidth  = CVPixelBufferGetWidth(pixelBuffer)
        let imgHeight = CVPixelBufferGetHeight(pixelBuffer)

        // Run Apple Vision human body pose on the frame
        do {
            try visionQueue.perform([poseRequest], on: pixelBuffer,
                                    orientation: .right)
        } catch { return }

        guard let observation = poseRequest.results?.first else { return }

        // Build keypoints dict (Vision → MoveNet names, y-axis flipped)
        var keypoints: [String: [Double]] = [:]
        for (visionJoint, movenetName) in jointMap {
            guard let pt = try? observation.recognizedPoint(visionJoint),
                  pt.confidence > 0.15 else { continue }
            // Vision bottom-left → top-left: flip y
            keypoints[movenetName] = [
                Double(1.0 - pt.location.y),   // y_norm
                Double(pt.location.x),           // x_norm
                Double(pt.confidence)
            ]
        }

        guard !keypoints.isEmpty else { return }

        // Camera world transform
        let t = frame.camera.transform
        let pose = CameraPose(
            position: [t.columns.3.x, t.columns.3.y, t.columns.3.z],
            rotation: [t.columns.0.z, t.columns.1.z, t.columns.2.z, t.columns.3.w]
        )

        // LiDAR depth for key joints (Pro models only)
        var depth: [String: Double] = [:]
        let depthJoints = ["left_ankle", "right_ankle", "left_knee", "right_knee",
                           "left_hip", "right_hip"]
        if let depthMap = frame.sceneDepth?.depthMap {
            let dW = CVPixelBufferGetWidth(depthMap)
            let dH = CVPixelBufferGetHeight(depthMap)
            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(depthMap) {
                let floats = base.assumingMemoryBound(to: Float32.self)
                for joint in depthJoints {
                    if let kp = keypoints[joint] {
                        // kp is [y_norm, x_norm, conf] — map to depth buffer coords
                        let px = max(0, min(dW - 1, Int(kp[1] * Double(dW))))
                        let py = max(0, min(dH - 1, Int(kp[0] * Double(dH))))
                        let d  = floats[py * dW + px]
                        if d > 0.1 && d < 15.0 { depth[joint] = Double(d) }
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        }

        let payload = JLFrame(
            timestamp:  frame.timestamp,
            frameId:    frameId,
            source:     UIDevice.current.name,
            hasLiDAR:   frame.sceneDepth != nil,
            cameraPose: pose,
            keypoints:  keypoints,
            depth:      depth
        )

        sendFrame(payload)
    }

    // MARK: Send

    private func sendFrame(_ frame: JLFrame) {
        guard isConnected, let conn = connection else { return }
        guard var json = try? String(data: JSONEncoder().encode(frame), encoding: .utf8) else { return }
        json += "\n"
        guard let data = json.data(using: .utf8) else { return }
        conn.send(content: data, completion: .idempotent)
    }
}

// MARK: - Simple SwiftUI / UIKit entry point

import SwiftUI

@main
struct JLVisionCaptureApp: App {
    var body: some Scene {
        WindowGroup { CaptureView() }
    }
}

struct CaptureView: View {
    @StateObject private var vm = CaptureViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            ARViewContainer(session: vm.capture.arSession)
                .ignoresSafeArea()

            VStack {
                // ── Status bar ────────────────────────────────────────
                HStack {
                    Circle()
                        .fill(vm.isConnected ? Color.green : Color.orange)
                        .frame(width: 10, height: 10)
                    Text(vm.statusText)
                        .foregroundColor(.white)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                    Spacer()
                    Text(vm.hasLiDAR ? "LiDAR ✓" : "No LiDAR")
                        .foregroundColor(vm.hasLiDAR ? .cyan : .gray)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .background(Color.black.opacity(0.55))

                Spacer()

                // ── Connection panel ──────────────────────────────────
                VStack(spacing: 10) {
                    TextField("Coach Mac/PC IP  (e.g. 192.168.1.10)", text: $vm.hostIP)
                        .keyboardType(.numbersAndPunctuation)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)
                        .padding(.horizontal)

                    Button(vm.isConnected ? "Disconnect" : "Connect & Stream") {
                        vm.toggleConnection()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(vm.isConnected ? .red : .cyan)
                }
                .padding()
                .background(.ultraThinMaterial)
                .cornerRadius(16)
                .padding(.horizontal, 20)
                .padding(.bottom, 32)
            }
        }
        .onAppear { vm.start() }
    }
}

// MARK: - ViewModel

class CaptureViewModel: ObservableObject {
    let capture = JLCaptureSession()

    @Published var statusText  = "Not connected"
    @Published var isConnected = false
    @Published var hostIP      = ""
    var hasLiDAR: Bool { ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) }

    func start() {
        capture.onStatusChange = { [weak self] msg in
            self?.statusText  = msg
            self?.isConnected = self?.capture.isConnected ?? false
        }
        capture.startCapture()
    }

    func toggleConnection() {
        if isConnected {
            capture.disconnect()
        } else {
            guard !hostIP.isEmpty else { statusText = "Enter IP address"; return }
            capture.connect(host: hostIP)
        }
    }
}

// MARK: - ARKit scene view

struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = session
        view.automaticallyUpdatesLighting = true
        view.rendersCameraGrain = false
        view.debugOptions = []   // set [.showWorldOrigin] to debug
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
