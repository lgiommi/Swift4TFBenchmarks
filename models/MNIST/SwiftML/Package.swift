// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftML",
    platforms: [
        .macOS(.v10_15),
    ],
    products: [
        .executable(name: "swift-ml", targets: ["SwiftML"]),
    ],
    dependencies: [
        //.package(url: "https://github.com/apple/swift-argument-parser", from: "0.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "0.2.0"),
        .package(url: "https://github.com/mxcl/Path.swift", from: "1.2.0"),
        .package(url: "https://github.com/JustHTTP/Just", from: "0.7.2"),
        //.package(url: "https://github.com/tensorflow/swift-models", .revision("3acbabd23a8d9b09aaf6d3c38391d0cbed7ce7b9")),
        // example of using local package, for that we need its location
        // and git revision hash string
        .package(url: "../../../../swift-models", .revision("3bd96d22cca19b1024540815089ac908474df567")),
    ],
    targets: [
        .target(
            name: "SwiftML",
            dependencies: [
                "Just",
                .product(name: "Datasets", package: "swift-models"),
                .product(name: "TrainingLoop", package: "swift-models"),
                .product(name: "Path", package: "Path.swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]),
        .testTarget(
            name: "SwiftMLTests",
            dependencies: ["SwiftML"]),
    ]
)
