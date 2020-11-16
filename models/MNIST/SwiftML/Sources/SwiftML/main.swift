import ArgumentParser

// for help, please see https://github.com/apple/swift-argument-parser
struct SwiftML: ParsableCommand {
    @Option(name: .shortAndLong, help: "The name of the file containing parameters for the model")
    var paramsFile: String?
    var out: String?
    @Argument(help: "Perform ML action (train|test)")
    var action: String
    mutating func run() throws {
        if action == "LeNet" {
            LeNetTrainMNIST(paramsFile ?? "params.json")
        }
        else {
            print("unsupported action \(action)")
        }
    }
}

SwiftML.main()

