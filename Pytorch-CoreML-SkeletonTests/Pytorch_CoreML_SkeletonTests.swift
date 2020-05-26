//
//  Pytorch_CoreML_SkeletonTests.swift
//  Pytorch-CoreML-SkeletonTests
//
//  Created by Gerald on 5/25/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

import XCTest
import CoreML
import AVFoundation

@testable import Pytorch_CoreML_Skeleton

class Pytorch_CoreML_SkeletonTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func test_wav__spectrogram() {
        let model = wave__spec()
        typealias NetworkInput = wave__specInput
        typealias NetworkOutput = wave__specOutput
        
        // read in the expected model output from JSON
        let bundle = Bundle(for: Pytorch_CoreML_SkeletonTests.self)
        let path = bundle.path(forResource: "spec_out.bonjour", ofType: "json")
        let data = try! Data(contentsOf: URL(fileURLWithPath: path!))
        let expected_spectrogram: [[NSNumber]] = try! JSONSerialization.jsonObject(with: data) as! [[NSNumber]]

        print( "expected spec: \(expected_spectrogram.count) \(expected_spectrogram[0].count)")

        // read the input shapes of our model
        let inputName = "input.1"
        let inputConstraint: MLFeatureDescription = model.model.modelDescription
            .inputDescriptionsByName[inputName]!

        let input_batch_size: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[0])! )
        let input_samples: Int = Int(truncating: (inputConstraint.multiArrayConstraint?.shape[1])! )

        // read the same WAV file used in PyTorch
        let testBundle = Bundle(for: type(of: self))
        guard let filePath = testBundle.path(forResource: "bonjour", ofType: "wav") else {
              fatalError( "error opening bonjour.wav" )
        }
          
        // Read wav file
        var wav_file:AVAudioFile!
        do {
            let fileUrl = URL(fileURLWithPath: filePath )
            wav_file = try AVAudioFile( forReading:fileUrl )
        } catch {
            fatalError("Could not open wav file.")
        }

        print("wav file length: \(wav_file.length)")

        let buffer = AVAudioPCMBuffer(pcmFormat: wav_file.processingFormat,
                                      frameCapacity: UInt32(wav_file.length))
        do {
            try wav_file.read(into:buffer!)
        } catch{
            fatalError("Error reading buffer.")
        }
          
        guard let bufferData = buffer?.floatChannelData![0] else {
            fatalError("Can not get a float handle to buffer")
        }
        
        // allocate a ML Array & copy samples over
        let array_shape = [input_batch_size as NSNumber, input_samples as NSNumber]
        let audioData = try! MLMultiArray(shape: array_shape, dataType: MLMultiArrayDataType.float32 )
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(audioData.dataPointer))
        for i in 0..<input_samples {
            ptr[i] = Float32(bufferData[i])
        }

        // create the input dictionary as { 'input.1' : [<wave floats>] }
        let inputs: [String: Any] = [
            inputName: audioData,
        ]
        // container for ML Model inputs
        let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)
               
        // Send the waveform samples into the model to generate the spectrogram
        let raw_outputs = try! model.model.prediction(from: provider)
               
        // convert raw dictionary into our model's output class
        let outputs = NetworkOutput( features: raw_outputs )
        // the output we're interested in is "_14"
        let output_spectrogram: MLMultiArray = outputs._14
        print( "outputs: \(output_spectrogram.shape)") // [1, 1, 61, 513]

        // sanity check the shapes of our output
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[2] ) == expected_spectrogram.count,
            "incorrect shape[2]! \(output_spectrogram.shape[2]) \(expected_spectrogram.count)" )
        XCTAssertTrue( Int( truncating: output_spectrogram.shape[3] ) == expected_spectrogram[0].count,
            "incorrect shape[3]! \(output_spectrogram.shape[3]) \(expected_spectrogram[0].count)" )

        // compare every element of our spectrogram with those from the JSON file
        for i in 0..<expected_spectrogram.count {
            let spec_row = expected_spectrogram[i] as [NSNumber]

            for j in 0..<spec_row.count {
                let test_idx: [NSNumber] = [ NSNumber(value: i), NSNumber(value: j) ]
                let val = output_spectrogram[ test_idx ].floatValue
                XCTAssertLessThan( abs( val - spec_row[j].floatValue ), 0.15,
                                   "spec vals different at \(i) \(j)!" )
            }
        }

    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
