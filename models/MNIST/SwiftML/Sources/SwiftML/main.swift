import ArgumentParser

// for help, please see https://github.com/apple/swift-argument-parser
struct SwiftML: ParsableCommand {
    @Option(name: .shortAndLong, help: "The number of epochs for train (default 500)")
    var epochs: Int?
    @Option(name: .shortAndLong, help: "The learning rate for train (default 0.1)")
    var learningRate: Float?
    @Option(name: .shortAndLong, help: "The batch size (default 32)")
    var batchSize: Int?
    @Option(name: .shortAndLong, help: "Name of the output file with results")
    var out: String?
    @Argument(help: "Perform ML action (train|test)")
    var action: String
    mutating func run() throws {
        if action == "LeNet" {
            LeNetTrainMNIST(epochs ?? 5, learningRate ?? 0.1, batchSize ?? 128)
        }
        if action == "LeNet_v2" {
            LeNet_v2TrainMNIST(epochs ?? 5, batchSize ?? 128)
        }
        else {
            print("unsupported action \(action)")
        }
    }
}

SwiftML.main()

