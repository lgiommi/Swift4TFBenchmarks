import Foundation

// import Python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

// helper function to run shell command of the specified string
// "/bin/ls".shell("-lh")
@available(macOS 10.13, *)
public extension String {
    @discardableResult
    func shell(_ args: String...) -> String
    {
        let (task,pipe) = (Process(),Pipe())
        task.executableURL = URL(fileURLWithPath: self)
        (task.arguments,task.standardOutput) = (args,pipe)
        do    { try task.run() }
        catch { print("Unexpected error: \(error).") }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: String.Encoding.utf8) ?? ""
    }
}

// helper function to download files
func downloadData(from sourceString: String, to destinationString: String) {
    let fileManager = FileManager.default
    if fileManager.fileExists(atPath: destinationString) {
        print("File \(destinationString) exists")
        return
    }

    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
}

// helper function to inspect our data
func inspectData(fname: String, num: Int = 5) {
    print("inspect: \(fname)")
    let f = Python.open(fname)
    for _ in 0..<num {
        print(Python.next(f).strip())
    }
    f.close()
}

// function for reading local json parameter files
func readLocalFile(forName paramsFile: String) -> [Any] {
    let url = URL(string: "file://\(paramsFile)")!
    //let url = Bundle.main.url(forResource: paramsFile, withExtension: "json")!
    do {
        let jsonData = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String:Any]
        let epochCount : Int = json["epochs"] as! Int
        let learningRate : Double = json["lr"] as! Double
        let batchSize : Int = json["batch_size"] as! Int
        let out : String = json["out"] as! String
        let plots : String = json["plots"] as! String
        print("learning rate: \(learningRate)\tbatch size: \(batchSize)\tepochs: \(epochCount)\toutput file: \(out)\tplots file: \(plots)")
        return([epochCount,learningRate,batchSize,out])
    }
    catch {
        print(error)
    }
    return([])
}

// copy local files from one path to another
func copyFiles (fromString : String, toString : String) {
    let fileManager = FileManager.default
    let toURL = URL(string: "file://\(toString)/topolino.json")!
    let fromURL = URL(string: "file://\(fromString)")!
    do{
        if fileManager.fileExists(atPath: "\(toString)/topolino.json") {
            try! fileManager.removeItem(atPath: "\(toString)/topolino.json")
        }
        try fileManager.copyItem(at: fromURL, to: toURL)
        print("copied")
    } catch let error {
        NSLog("Error in copying Data.plist: \(error)") // see the above quoted error message from here
    }
}
