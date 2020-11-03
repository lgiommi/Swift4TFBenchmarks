import ArgumentParser

// for help, please see https://github.com/apple/swift-argument-parser
struct SwiftML: ParsableCommand {
    @Option(name: .shortAndLong, help: "The number of epochs for train (default 500)")
    var epochs: Int?
    @Option(name: .shortAndLong, help: "The batch size (default 32)")
    var batchSize: Int?
    @Argument(help: "Perform ML action (train|test)")
    var action: String
    mutating func run() throws {
        if action == "LeNet" {
            LeNetTrainMNIST(epochs ?? 5, batchSize ?? 128)
        } else {
            print("unsupported action \(action)")
        }
    }
}

SwiftML.main()

