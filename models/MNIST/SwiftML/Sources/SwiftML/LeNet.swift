// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Datasets
import TensorFlow
import TrainingLoop
import CoreFoundation
import Foundation

func readLocalFile(forName paramsFile: String) -> [Any] {
    let url = Bundle.main.url(forResource: paramsFile, withExtension: "json")!
    do {
        let jsonData = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: jsonData) as! [String:Any]

        let epochCount : Int = json["epochs"] as! Int
        let learningRate : Double = json["lr"] as! Double
        let batchSize : Int = json["batchSize"] as! Int
        let out : String = json["out"] as! String
        return([epochCount,learningRate,batchSize,out])
    }
    catch {
        print(error)
    }
    return([])
}

func LeNetTrainMNIST(_ paramsFile : String = "params") {

    // Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
    // device on macOS instead of X10.
#if os(macOS)
      let device = Device.defaultTFEager
#else
      let device = Device.defaultXLA
#endif
    
    let BundlePath = Bundle.main.resourcePath!
    print("The \(paramsFile) file  is stored in the following path \(BundlePath)")
    
    let paramsFile = paramsFile.replacingOccurrences(of: ".json", with: "")
    let results = readLocalFile(forName: paramsFile)
    print(results)
    let epochCount = results[0] as! Int
    let learningRate = results[1] as! Double
    let batchSize = results[2] as! Int
    let out = results[3] as! String
    
    let dataset = MNIST(batchSize: batchSize, on: device)

    // The   model, equivalent to `LeNet` in `ImageClassificationModels`.
    var classifier = Sequential {
        Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
        AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
        Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
        AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
        Flatten<Float>()
        Dense<Float>(inputSize: 400, outputSize: 40, activation: relu)
        Dense<Float>(inputSize: 40, outputSize: 20, activation: relu)
        Dense<Float>(inputSize: 20, outputSize: 10)
    }
    
    let optimizer = SGD(for: classifier, learningRate: Float(learningRate))

    let trainingProgress = TrainingProgress()
    var trainingLoop = TrainingLoop(
      training: dataset.training,
      validation: dataset.validation,
      optimizer: optimizer,
      lossFunction: softmaxCrossEntropy,
      callbacks: [trainingProgress.update])
    
    let startTime = CFAbsoluteTimeGetCurrent()
    try! trainingLoop.fit(&classifier, epochs: epochCount, on: device)
    let trainTime = CFAbsoluteTimeGetCurrent() - startTime
    print("Training time: \(Double(trainTime)) s.")
    
    // plot our accuracies and losses
    let acc = trainingProgress.accuracies
    let loss = trainingProgress.losses
    var n_batches : [Int] = [60000,10000]
    n_batches = n_batches.map { Double($0)/Double(batchSize) }.map { Int(floor($0)) }
    n_batches[1]+=1
    var accT_list : [Double] = []
    var accV_list : [Double] = []
    var lossT_list : [Double] = []
    var lossV_list : [Double] = []
    for i in 0..<acc.count {
      if i != 0 && ((i-n_batches[0]) % (n_batches[0] + n_batches[1]) == 0 || i == n_batches[0]) {
        accT_list.append(Double(acc[i-1]))
        lossT_list.append(Double(loss[i-1]))
      }
      if i != 0 && i % (n_batches[0]+n_batches[1]) == 0 {
        accV_list.append(Double(acc[i-1]))
        lossV_list.append(Double(loss[i-1]))
      }
    }
    
    accV_list.append(Double(acc[acc.count-1]))
    lossV_list.append(Double(loss[loss.count-1]))
    
    let dictionary: [String: Any] = ["Swift": ["loss": lossT_list, "accuracy": accT_list, "val_loss": lossV_list, "val_accuracy": accV_list, "trainTime": trainTime]]
    
    do {
        let fileURL = try FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent(out)

        try JSONSerialization.data(withJSONObject: dictionary)
            .write(to: fileURL)
    } catch {
        print(error)
    }

    
    print("Accuracy Training: \(accT_list) ")
    print("Accuracy Validation: \(accV_list) ")
    print("Loss Training: \(lossT_list) ")
    print("Loss Validation: \(lossV_list) ")
    //plot(acc: acc, loss: loss, fname: "mnist.pdf")
}

